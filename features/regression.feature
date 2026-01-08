Feature: Regression tests
  Asegurar que cambios no rompen comportamientos ya soportados

  Background:
    Given un entorno con el venv activado

  @regression
  Scenario: Revisión de guardado y lectura de transcripción
    Given tengo el texto "registro de regresión"
    When guardo la transcripción en "reg_out.txt"
    Then el fichero "reg_out.txt" debe existir y contener "registro de regresión"

  @regression
  Scenario: Normalización y asignación de hablante estable
    Given existe un archivo de audio multicanal "reg_multi.wav"
    When normalizo el audio para diarización
    Then obtengo un archivo temporal WAV muestreado a 16kHz
    Given una estructura de diarización con dos tracks
    When pido el hablante para el segmento entre 0.5 y 1.5
    Then deberia recibir el identificador del hablante más probable
