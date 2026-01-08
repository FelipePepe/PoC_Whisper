"""
Ejemplo de uso del módulo de diarización (identificación de hablantes)
"""
from .diarize import transcribe_with_speaker_diarization, format_transcription_by_speaker, save_diarized_transcription
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


def ejemplo_diarizacion():
    """
    Ejemplo de transcripción con identificación de hablantes
    """
    print("=" * 60)
    print("EJEMPLO: Transcripción con Diarización de Hablantes")
    print("=" * 60)
    print()
    
    # Leer el token desde .env
    HF_TOKEN = os.getenv('HF_TOKEN')
    
    if not HF_TOKEN:
        print("❌ ERROR: Token de HuggingFace no encontrado")
        print()
        print("Pasos para configurar:")
        print("1. Copia el archivo .env.example a .env:")
        print("   cp .env.example .env")
        print()
        print("2. Edita .env y añade tu token:")
        print("   HF_TOKEN=hf_xxxxxxxxxxxxx")
        print()
        print("3. Obtén el token en:")
        print("   - https://huggingface.co/settings/tokens")
        print("   - Acepta condiciones: https://huggingface.co/pyannote/speaker-diarization-3.1")
        return
    
    # Archivo de audio a transcribir
    audio_file = "audio.mp3"
    
    try:
        print(f"Procesando: {audio_file}")
        print()
        
        # Transcribir con identificación de hablantes
        segments = transcribe_with_speaker_diarization(
            audio_path=audio_file,
            hf_token=HF_TOKEN,
            model_size="base",
            language="es",
            num_speakers=None  # Auto-detectar número de hablantes
        )
        
        print()
        print("=" * 60)
        print("RESULTADO:")
        print("=" * 60)
        print(format_transcription_by_speaker(segments))
        
        # Guardar archivos
        save_diarized_transcription(segments, "transcripcion_grouped.txt", "grouped")
        save_diarized_transcription(segments, "transcripcion_timestamped.txt", "timestamped")
        
        # Estadísticas
        speakers = {seg['speaker'] for seg in segments}
        print()
        print("Estadísticas:")
        print(f"  - Total de segmentos: {len(segments)}")
        print(f"  - Hablantes identificados: {len(speakers)}")
        print(f"  - Hablantes: {', '.join(sorted(speakers))}")
        
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo '{audio_file}'")
        print("Por favor, coloca un archivo de audio en el directorio del proyecto")
        print()
        print("O ejecuta directamente desde línea de comandos:")
        print(f"  python diarize.py {audio_file} {HF_TOKEN} base es")
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    ejemplo_diarizacion()
