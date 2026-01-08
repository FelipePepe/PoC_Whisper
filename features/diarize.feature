Feature: Diarización de hablantes
  Para identificar hablantes y combinar con la transcripción
  Como desarrollador quiero definir los escenarios clave de la pipeline de diarización

  Background:
    Given un entorno con el venv activado

  @integration
  Scenario: Normalizar audio para pyannote
    Given existe un archivo de audio multicanal "multi.wav"
    When normalizo el audio para diarización
    Then obtengo un archivo temporal WAV muestreado a 16kHz

  @integration
  Scenario: Determinar hablante para un segmento
    Given una estructura de diarización con dos tracks
    When pido el hablante para el segmento entre 0.5 y 1.5
    Then deberia recibir el identificador del hablante más probable

  @integration
  Scenario: Transcribir con diarización sin token HF
    Given existe un archivo de audio válido "ejemplo.wav"
    And no está definido `HF_TOKEN` en el entorno
    When intento transcribir con identificación de hablantes
    Then recibo un error sobre token de HuggingFace faltante

  @integration
  Scenario: Guardar transcripción diarizada en formatos agrupado y con timestamps
    Given una lista de segmentos con hablantes y texto
    When guardo la transcripción en modo "grouped" y en modo "timestamped"
    Then se crean los ficheros de salida con el contenido esperado
