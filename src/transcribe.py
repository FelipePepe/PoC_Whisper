"""
Módulo para transcribir audio a texto usando Whisper de OpenAI
"""
import whisper
import os
from pathlib import Path
from typing import Optional


def transcribe_audio(audio_path: str, model_size: str = "base", language: Optional[str] = None) -> dict:
    """
    Transcribe un archivo de audio a texto usando Whisper.
    
    Args:
        audio_path (str): Ruta al archivo de audio
        model_size (str): Tamaño del modelo ('tiny', 'base', 'small', 'medium', 'large')
        language (str): Idioma del audio (ej: 'es', 'en'). Si es None, se detecta automáticamente
    
    Returns:
        dict: Diccionario con el texto transcrito y metadatos
    """
    # Verificar que el archivo existe
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"El archivo {audio_path} no existe")
    
    print(f"Cargando modelo Whisper '{model_size}'...")
    model = whisper.load_model(model_size)
    
    print(f"Transcribiendo '{audio_path}'...")
    
    # Opciones de transcripción
    options = {}
    if language:
        options['language'] = language
    
    # Realizar la transcripción
    result = model.transcribe(audio_path, **options)
    
    return result


def transcribe_with_timestamps(audio_path: str, model_size: str = "base", language: Optional[str] = None) -> list:
    """
    Transcribe un archivo de audio y devuelve segmentos con timestamps.
    
    Args:
        audio_path (str): Ruta al archivo de audio
        model_size (str): Tamaño del modelo ('tiny', 'base', 'small', 'medium', 'large')
        language (str): Idioma del audio (ej: 'es', 'en'). Si es None, se detecta automáticamente
    
    Returns:
        list: Lista de segmentos con texto y timestamps
    """
    result = transcribe_audio(audio_path, model_size, language)
    
    segments = []
    for segment in result['segments']:
        segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text']
        })
    
    return segments


def save_transcription(text: str, output_path: str):
    """
    Guarda la transcripción en un archivo de texto.
    
    Args:
        text (str): Texto transcrito
        output_path (str): Ruta del archivo de salida
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Transcripción guardada en: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python transcribe.py <archivo_audio> [modelo] [idioma]")
        print("Ejemplo: python transcribe.py audio.mp3 base es")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "base"
    lang = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        result = transcribe_audio(audio_file, model, lang)
        print("\n" + "="*50)
        print("TRANSCRIPCIÓN:")
        print("="*50)
        print(result["text"])
        
        # Guardar en archivo
        output_file = Path(audio_file).stem + "_transcripcion.txt"
        save_transcription(result["text"], output_file)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
