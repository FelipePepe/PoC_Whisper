Feature: End-to-end smoke flows
  Validar flujos completos de alto nivel como smoke tests

  Background:
    Given un entorno con el venv activado

  @e2e
  Scenario: Flujo completo transcribe -> diarize (modo smoke)
    Given existe un archivo de audio válido "e2e_example.wav"
    When transcribo el archivo "e2e_example.wav" con el modelo "tiny" y el idioma "es"
    Then la salida debe contener la clave "text" con la transcripción
    Given no está definido `HF_TOKEN` en el entorno
    When intento transcribir con identificación de hablantes
    Then recibo un error sobre token de HuggingFace faltante

  @e2e
  Scenario: Guardar transcripción diarizada (smoke)
    Given existe un archivo de audio válido "e2e_example.wav"
    Given una lista de segmentos con hablantes y texto
    When guardo la transcripción en modo "grouped"
    Then se crean los ficheros de salida con el contenido esperado
