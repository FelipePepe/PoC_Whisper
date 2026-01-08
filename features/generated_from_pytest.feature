Feature: Escenarios generados desde pytest
  Estos escenarios reflejan de forma legible los tests existentes en pytest

  Background:
    Given un entorno con el venv activado

  @unit
  Scenario: transcribe_main no args y éxito
    Given el módulo de transcripción se ejecuta sin argumentos
    When invoco el entrypoint de `src.transcribe` con argumento de audio válido
    Then debería imprimir la transcripción y guardar el fichero "_transcripcion.txt"

  @unit
  Scenario: transcribe_audio fichero no existe
    Given no existe el archivo "nonexistent_file.wav"
    When llamo a `transcribe_audio` con ese fichero
    Then recibo un `FileNotFoundError`

  @unit
  Scenario: transcribe con modelo mockeado devuelve texto
    Given un modelo Whisper mockeado que devuelve texto
    When transcribo un audio con `transcribe_audio`
    Then la respuesta contiene la clave "text"

  @unit
  Scenario: transcribe con timestamps devuelve segmentos
    Given `transcribe_audio` devuelve segmentos con timestamps
    When solicito `transcribe_with_timestamps`
    Then recibo una lista con objetos que contienen "start", "end" y "text"

  @unit
  Scenario: save_transcription escribe fichero
    Given el texto "hola mundo"
    When llamo a `save_transcription` con "salida.txt"
    Then el fichero "salida.txt" debe existir con el texto

  @unit
  Scenario: example.py - transcripción simple maneja éxito y error de fichero
    Given el ejemplo usa `AUDIO_FILE` apuntando a un archivo
    When ejecuto `ejemplo_transcripcion_simple`
    Then imprime la transcripción y llama a `save_transcription` o muestra error si falta el fichero

  @unit
  Scenario: example.py - transcripción con timestamps
    Given `transcribe_with_timestamps` devuelve segmentos
    When ejecuto `ejemplo_transcripcion_con_timestamps`
    Then se imprimen los segmentos con formato de timestamps

  @unit
  Scenario: example_diarization maneja token ausente
    Given NO está definido `HF_TOKEN`
    When ejecuto `ejemplo_diarizacion`
    Then imprime instrucciones para configurar el token

  @integration
  Scenario: diarize.normalize_audio para pyannote requiere torchaudio
    Given no está disponible `torchaudio`
    When llamo a `normalize_audio_for_diarization`
    Then se lanza `RuntimeError` informando la ausencia de torchaudio

  @integration
  Scenario: transcribe_with_speaker_diarization combina diarización y transcripción
    Given pyannote y whisper están mockeados
    When ejecuto `transcribe_with_speaker_diarization`
    Then devuelve segmentos anotados con `speaker`, `start`, `end`, `text`

  @integration
  Scenario: diarize __main__ maneja token en argumento y crea ficheros en el directorio del audio
    Given un audio temporal y token pasado como argumento hf_xxx
    When ejecuto el módulo `src.diarize` como script
    Then se crean los ficheros `_diarized_grouped.txt` y `_diarized_timestamped.txt` junto al audio

  @integration
  Scenario: diarize __main__ informa y sale si falta token
    Given no hay `HF_TOKEN` y no se pasa token por argumento
    When ejecuto el módulo `src.diarize` como script
    Then el proceso termina mostrando el mensaje de token faltante

  @unit
  Scenario: get_speaker_for_segment asigna UNKNOWN cuando no hay solapamiento
    Given una diarización sin solapamiento para el segmento consultado
    When pido el hablante para ese segmento
    Then la función devuelve "UNKNOWN"

  @unit
  Scenario: format_transcription_by_speaker agrupa correctamente por hablante
    Given una lista de segmentos con cambios de hablante
    When llamo a `format_transcription_by_speaker`
    Then el texto resultante agrupa los bloques por hablante

  @unit
  Scenario: GUI - advertencia si no hay archivo seleccionado
    Given la UI sin `audio_file`
    When llamo a `start_transcription`
    Then se muestra una advertencia al usuario

  @unit
  Scenario: GUI - error si el archivo no existe
    Given `audio_file` apunta a ruta inexistente
    When llamo a `start_transcription`
    Then se muestra un error al usuario

  @unit
  Scenario: GUI - guardar resultado sin texto muestra advertencia
    Given el área de resultado está vacía
    When llamo a `save_result`
    Then aparece una advertencia y no se escribe fichero

  @unit
  Scenario: GUI - guardar resultado escribe fichero y muestra confirmación
    Given el área de resultado contiene texto
    When llamo a `save_result` y el fichero se crea correctamente
    Then se muestra un diálogo de confirmación y la barra de estado se actualiza

  @unit
  Scenario: GUI - cancelación pregunta al usuario y finaliza si confirma
    Given una transcripción en curso (`processing` True)
    When el usuario confirma la cancelación
    Then el estado `processing` queda en False y la UI se actualiza

  @e2e
  Scenario: flujo e2e smoke transcribe + diarize (token faltante)
    Given existe un audio de ejemplo
    When ejecuto la transcripción seguida de intento de diarización sin token
    Then la transcripción se produce y la diarización falla por token faltante
