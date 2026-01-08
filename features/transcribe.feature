Feature: Transcripción de audio
  Para convertir audio a texto usando Whisper
  Como desarrollador quiero asegurar los comportamientos clave de transcripción

  Background:
    Given un entorno con el venv activado

  @unit
  Scenario: Archivo de audio no existe
    Given no existe el archivo "audio_inexistente.wav"
    When intento transcribir el archivo "audio_inexistente.wav" usando el módulo de transcripción
    Then recibiré un error indicando que el archivo no existe

  @unit
  Scenario: Transcribir con modelo y idioma especificados
    Given existe un archivo de audio válido "ejemplo.wav"
    When transcribo el archivo "ejemplo.wav" con el modelo "tiny" y el idioma "es"
    Then la salida debe contener la clave "text" con la transcripción

  @unit
  Scenario: Obtener segmentos con timestamps
    Given el modelo devuelve segmentos con timestamps
    When solicito la transcripción con timestamps para "ejemplo.wav"
    Then recibiré una lista de segmentos con campos "start", "end" y "text"

  @unit
  Scenario: Guardar la transcripción en un fichero
    Given tengo el texto "hola mundo"
    When guardo la transcripción en "salida.txt"
    Then el fichero "salida.txt" debe existir y contener "hola mundo"
