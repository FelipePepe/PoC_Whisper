"""
Módulo para transcribir audio con diarización de hablantes
Combina Whisper (transcripción) con pyannote.audio (identificación de hablantes)
"""
import whisper
import os
from pathlib import Path
from typing import Optional
import torch
from dotenv import load_dotenv
import tempfile

# Cargar variables de entorno desde .env
load_dotenv()

# Limitar a 1 hilo por procesador físico (proteger llamadas que pueden fallar en entornos ya inicializados)
NUM_CORES = os.cpu_count() or 4
try:
    torch.set_num_threads(NUM_CORES)  # 1 thread por core
except Exception:
    pass
try:
    torch.set_num_interop_threads(1)  # Mínimo para interoperabilidad
except Exception:
    pass


def normalize_audio_for_diarization(audio_path: str) -> str:
    """
    Normaliza el audio a formato WAV con 16kHz mono para compatibilidad con pyannote.
    
    Args:
        audio_path (str): Ruta al archivo de audio original
    
    Returns:
        str: Ruta al archivo temporal normalizado
    """
    print("Normalizando audio para diarización...")
    
    # Cargar audio con torchaudio (importar aquí para evitar coste en importación del módulo)
    try:
        import torchaudio
    except Exception:
        raise RuntimeError("torchaudio no está disponible en este entorno para normalizar audio")

    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convertir a mono si es necesario
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resamplear a 16kHz si es necesario
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Guardar en archivo temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    torchaudio.save(temp_file.name, waveform, 16000)
    
    print(f"Audio normalizado guardado en: {temp_file.name}")
    return temp_file.name


def transcribe_with_speaker_diarization(
    audio_path: str,
    hf_token: str,
    model_size: str = "base",
    language: Optional[str] = None,
    num_speakers: Optional[int] = None
) -> list:
    """
    Transcribe un archivo de audio identificando quién habla en cada momento.
    
    Args:
        audio_path (str): Ruta al archivo de audio
        hf_token (str): Token de HuggingFace para acceder a pyannote
        model_size (str): Tamaño del modelo Whisper ('tiny', 'base', 'small', 'medium', 'large')
        language (str): Idioma del audio (ej: 'es', 'en'). Si es None, se detecta automáticamente
        num_speakers (int): Número de hablantes (opcional, si se conoce de antemano)
    
    Returns:
        list: Lista de segmentos con texto, hablante y timestamps
    """
    # Verificar que el archivo existe
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"El archivo {audio_path} no existe")
    
    # Normalizar audio para pyannote
    normalized_audio = normalize_audio_for_diarization(audio_path)
    
    print("Paso 1/3: Identificando hablantes con pyannote.audio...")
    
    # Cargar pipeline de diarización (importar dentro de la función para pruebas y entornos ligeros)
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    
    # Usar GPU si está disponible
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    
    # Realizar diarización
    diarization_params = {}
    if num_speakers:
        diarization_params['num_speakers'] = num_speakers
    
    diarization = pipeline(normalized_audio, **diarization_params)
    
    # Limpiar archivo temporal
    try:
        os.unlink(normalized_audio)
    except Exception:
        pass
    
    print(f"Paso 2/3: Transcribiendo audio con Whisper '{model_size}'...")
    
    # Cargar modelo Whisper
    model = whisper.load_model(model_size)
    
    # Opciones de transcripción
    options = {"word_timestamps": True}
    if language:
        options['language'] = language
    
    # Realizar transcripción
    result = model.transcribe(audio_path, **options)
    
    print("Paso 3/3: Combinando transcripción con identificación de hablantes...")
    
    # Combinar diarización con transcripción
    segments_with_speakers = []
    
    for segment in result['segments']:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        
        # Encontrar el hablante más probable para este segmento
        speaker = get_speaker_for_segment(diarization, start_time, end_time)
        
        segments_with_speakers.append({
            'start': start_time,
            'end': end_time,
            'speaker': speaker,
            'text': text
        })
    
    return segments_with_speakers


def get_speaker_for_segment(diarization, start_time, end_time):
    """
    Determina qué hablante es más probable para un segmento dado.
    
    Args:
        diarization: Resultado de la diarización de pyannote
        start_time (float): Tiempo de inicio del segmento
        end_time (float): Tiempo de fin del segmento
    
    Returns:
        str: Identificador del hablante (ej: "SPEAKER_00")
    """
    # Calcular overlap con cada hablante
    speaker_times = {}
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start < end_time and turn.end > start_time:
            # Hay overlap
            overlap_start = max(turn.start, start_time)
            overlap_end = min(turn.end, end_time)
            overlap_duration = overlap_end - overlap_start
            
            if speaker not in speaker_times:
                speaker_times[speaker] = 0
            speaker_times[speaker] += overlap_duration
    
    # Retornar el hablante con mayor tiempo de overlap
    if speaker_times:
        return max(speaker_times, key=speaker_times.get)
    else:
        return "UNKNOWN"


def format_transcription_by_speaker(segments: list) -> str:
    """
    Formatea la transcripción agrupando el texto por hablante.
    
    Args:
        segments (list): Lista de segmentos con speaker, start, end, text
    
    Returns:
        str: Transcripción formateada por hablante
    """
    output = []
    current_speaker = None
    current_text = []
    
    for seg in segments:
        speaker = seg['speaker']
        
        if speaker != current_speaker:
            # Cambio de hablante
            if current_speaker is not None:
                output.append(f"\n{current_speaker}:\n{''.join(current_text)}\n")
            
            current_speaker = speaker
            current_text = [seg['text']]
        else:
            current_text.append(seg['text'])
    
    # Agregar el último hablante
    if current_speaker is not None:
        output.append(f"\n{current_speaker}:\n{''.join(current_text)}\n")
    
    return ''.join(output)


def save_diarized_transcription(segments: list, output_path: str, format_type: str = "grouped"):
    """
    Guarda la transcripción con diarización en un archivo.
    
    Args:
        segments (list): Lista de segmentos con speaker, start, end, text
        output_path (str): Ruta del archivo de salida
        format_type (str): "grouped" (agrupado por hablante) o "timestamped" (con timestamps)
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        if format_type == "grouped":
            f.write(format_transcription_by_speaker(segments))
        else:  # timestamped
            for seg in segments:
                f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}:\n")
                f.write(f"{seg['text']}\n\n")
    
    print(f"Transcripción con diarización guardada en: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python diarize.py <archivo_audio> [hf_token] [modelo] [idioma] [num_speakers]")
        print("\nArgumentos:")
        print("  archivo_audio: Ruta al archivo de audio")
        print("  hf_token: Token de HuggingFace (opcional si está en .env)")
        print("  modelo: Tamaño del modelo Whisper (default: base)")
        print("  idioma: Código de idioma (ej: es, en) (default: auto-detectar)")
        print("  num_speakers: Número de hablantes si se conoce (opcional)")
        print("\nEjemplo con token en .env:")
        print("  python diarize.py audio.mp3 base es 3")
        print("\nEjemplo con token explícito:")
        print("  python diarize.py audio.mp3 hf_xxxxxxxxxxxxx base es 3")
        print("\nNOTA: Define HF_TOKEN en archivo .env (ver .env.example)")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Intentar obtener el token desde variables de entorno o argumentos
    token = os.getenv('HF_TOKEN')
    arg_offset = 0
    
    # Si el segundo argumento no parece un modelo, asumimos que es el token
    if len(sys.argv) > 2 and sys.argv[2].startswith('hf_'):
        token = sys.argv[2]
        arg_offset = 1
    
    if not token:
        print("❌ ERROR: Token de HuggingFace no encontrado")
        print("\nOpciones:")
        print("1. Define HF_TOKEN en archivo .env (recomendado)")
        print("2. Pásalo como argumento: python diarize.py audio.mp3 hf_xxxxx")
        print("\nPasos para obtener el token:")
        print("- https://huggingface.co/settings/tokens")
        print("- https://huggingface.co/pyannote/speaker-diarization-3.1")
        sys.exit(1)
    
    model = sys.argv[2 + arg_offset] if len(sys.argv) > 2 + arg_offset else "base"
    lang = sys.argv[3 + arg_offset] if len(sys.argv) > 3 + arg_offset else None
    num_spk = int(sys.argv[4 + arg_offset]) if len(sys.argv) > 4 + arg_offset else None
    
    try:
        # Resolve the transcribe function from sys.modules if available so tests
        # that monkeypatch the imported module's attribute are respected when
        # executing the module via runpy.run_module.
        import sys as _sys
        _mod = _sys.modules.get('src.diarize')
        _transcribe_fn = None
        if _mod is not None and hasattr(_mod, 'transcribe_with_speaker_diarization'):
            _transcribe_fn = getattr(_mod, 'transcribe_with_speaker_diarization')

        if _transcribe_fn is None:
            _transcribe_fn = transcribe_with_speaker_diarization

        segments = _transcribe_fn(
            audio_file,
            token,
            model,
            lang,
            num_spk
        )
        
        print("\n" + "="*60)
        print("TRANSCRIPCIÓN CON IDENTIFICACIÓN DE HABLANTES")
        print("="*60)
        print(format_transcription_by_speaker(segments))
        
        # Guardar versión agrupada y timestamped en el mismo directorio que el
        # archivo de entrada para que las pruebas y usuarios encuentren los
        # resultados junto al audio original.
        audio_path = Path(audio_file)
        output_file_grouped = audio_path.parent / f"{audio_path.stem}_diarized_grouped.txt"
        save_diarized_transcription(segments, str(output_file_grouped), "grouped")

        # Guardar versión con timestamps
        output_file_timestamped = audio_path.parent / f"{audio_path.stem}_diarized_timestamped.txt"
        save_diarized_transcription(segments, str(output_file_timestamped), "timestamped")
        
        # Estadísticas
        speakers = {seg['speaker'] for seg in segments}
        print("\nEstadísticas:")
        print(f"  - Hablantes identificados: {len(speakers)}")
        print(f"  - Hablantes: {', '.join(sorted(speakers))}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
