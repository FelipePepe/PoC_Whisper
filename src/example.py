"""
Ejemplo de uso del módulo de transcripción Whisper
"""
from .transcribe import transcribe_audio, transcribe_with_timestamps, save_transcription

# Archivo de audio de ejemplo
AUDIO_FILE = "audio.mp3"


def ejemplo_transcripcion_simple():
    """
    Ejemplo básico: transcribir un archivo de audio
    """
    print("Ejemplo 1: Transcripción simple")
    print("-" * 50)
    
    # NOTA: Reemplaza 'audio.mp3' con tu archivo de audio
    audio_file = AUDIO_FILE
    
    try:
        # Transcribir usando el modelo 'base'
        resultado = transcribe_audio(audio_file, model_size="base")
        
        print("Texto transcrito:")
        print(resultado["text"])
        print()
        
        # Guardar en archivo
        save_transcription(resultado["text"], "transcripcion.txt")
        
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo '{audio_file}'")
        print("Por favor, coloca un archivo de audio en el directorio del proyecto")
        print("Formatos soportados: mp3, wav, m4a, mp4, webm, etc.")


def ejemplo_transcripcion_con_timestamps():
    """
    Ejemplo avanzado: transcribir con segmentos y timestamps
    """
    print("\nEjemplo 2: Transcripción con timestamps")
    print("-" * 50)
    
    audio_file = AUDIO_FILE
    
    try:
        # Transcribir con timestamps
        segmentos = transcribe_with_timestamps(audio_file, model_size="base")
        
        print("Segmentos con timestamps:")
        for seg in segmentos:
            start = seg['start']
            end = seg['end']
            text = seg['text']
            print(f"[{start:.2f}s - {end:.2f}s]: {text}")
        
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo '{audio_file}'")


def ejemplo_transcripcion_con_idioma():
    """
    Ejemplo: transcribir especificando el idioma
    """
    print("\nEjemplo 3: Transcripción especificando idioma español")
    print("-" * 50)
    
    audio_file = AUDIO_FILE
    
    try:
        # Transcribir en español usando el modelo 'small' para mejor precisión
        resultado = transcribe_audio(audio_file, model_size="base", language="es")
        
        print("Texto transcrito (español):")
        print(resultado["text"])
        
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo '{audio_file}'")


if __name__ == "__main__":
    print("=" * 50)
    print("PoC Whisper - Ejemplos de Transcripción")
    print("=" * 50)
    print()
    
    # Ejecutar ejemplos
    ejemplo_transcripcion_simple()
    
    # Descomentar para ejecutar más ejemplos:
    # ejemplo_transcripcion_con_timestamps()
    # ejemplo_transcripcion_con_idioma()
    
    print("\n" + "=" * 50)
    print("Ejemplos completados")
    print("=" * 50)
