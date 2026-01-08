import os
import tempfile
from behave import given, when, then

from src import transcribe


class DummyModel:
    def __init__(self, text="texto de prueba"):
        self._text = text

    def transcribe(self, audio_path, **opts):
        return {"text": self._text, "segments": [{"start": 0.0, "end": 1.0, "text": "hola"}]}



@given('no existe el archivo "{fname}"')
def step_no_file(context, fname):
    # Asegurar que el archivo no existe en cwd
    p = os.path.abspath(fname)
    if os.path.exists(p):
        os.unlink(p)
    context.audio_path = p


@when('intento transcribir el archivo "{fname}" usando el módulo de transcripción')
def step_try_transcribe_missing(context, fname):
    try:
        transcribe.transcribe_audio(fname)
        context.exc = None
    except Exception as e:
        context.exc = e


@then('recibiré un error indicando que el archivo no existe')
def step_assert_file_error(context):
    assert context.exc is not None
    assert isinstance(context.exc, FileNotFoundError)





@when('transcribo el archivo "{fname}" con el modelo "{model}" y el idioma "{lang}"')
def step_transcribe_with_model(context, fname, model, lang):
    # mockear la carga de modelo
    orig = transcribe.whisper.load_model
    transcribe.whisper.load_model = lambda size: DummyModel("resultado completo")
    try:
        context.result = transcribe.transcribe_audio(fname, model_size=model, language=lang)
    finally:
        transcribe.whisper.load_model = orig


@then('la salida debe contener la clave "text" con la transcripción')
def step_assert_text_key(context):
    assert isinstance(context.result, dict)
    assert 'text' in context.result


@when('solicito la transcripción con timestamps para "{fname}"')
def step_transcribe_timestamps(context, fname):
    # mockear load_model para devolver segmentos
    orig = transcribe.whisper.load_model
    transcribe.whisper.load_model = lambda size: DummyModel()
    try:
        context.segments = transcribe.transcribe_with_timestamps(fname)
    finally:
        transcribe.whisper.load_model = orig


@given('el modelo devuelve segmentos con timestamps')
def step_model_returns_segments(context):
    # marcado informativo para el escenario; la función de transcripción se mockea en el When
    context.model_returns_segments = True


@then('recibiré una lista de segmentos con campos "start", "end" y "text"')
def step_assert_segments(context):
    assert isinstance(context.segments, list)
    seg = context.segments[0]
    assert 'start' in seg and 'end' in seg and 'text' in seg


@given('tengo el texto "{text}"')
def step_have_text(context, text):
    context.save_text = text


@when('guardo la transcripción en "{out}"')
def step_save_transcription(context, out):
    p = os.path.abspath(out)
    transcribe.save_transcription(context.save_text, p)
    context.saved_path = p


@then('el fichero "{out}" debe existir y contener "{text}"')
def step_check_saved_file(context, out, text):
    p = os.path.abspath(out)
    assert os.path.exists(p)
    content = open(p, 'r', encoding='utf-8').read()
    assert text in content


@given('el módulo de transcripción se ejecuta sin argumentos')
def step_module_no_args(context):
    # marker step for generated scenarios
    context.module_no_args = True


@when('invoco el entrypoint de `src.transcribe` con argumento de audio válido')
def step_invoke_transcribe_main(context):
    import runpy, sys, tempfile
    # create temporary audio file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tmp.write(b'RIFF')
    tmp.close()
    context.audio_path = tmp.name

    # patch whisper model to avoid heavy loading
    orig = transcribe.whisper.load_model
    transcribe.whisper.load_model = lambda size: DummyModel('texto e2e')

    old_argv = sys.argv[:]
    try:
        sys.argv = ['transcribe.py', context.audio_path]
        runpy.run_module('src.transcribe', run_name='__main__')
    finally:
        transcribe.whisper.load_model = orig
        sys.argv = old_argv


@then('debería imprimir la transcripción y guardar el fichero "_transcripcion.txt"')
def step_assert_transcribe_main_saved(context):
    from pathlib import Path
    out = Path(context.audio_path).stem + '_transcripcion.txt'
    assert Path(out).exists()


@given('un modelo Whisper mockeado que devuelve texto')
def step_given_mock_model(context):
    context.use_mock_model = True


@when('transcribo un audio con `transcribe_audio`')
def step_when_transcribe_audio(context):
    import tempfile, os
    # create temp audio
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tmp.write(b'RIFF')
    tmp.close()
    context.audio_path = tmp.name

    # patch model
    orig = transcribe.whisper.load_model
    transcribe.whisper.load_model = lambda size: DummyModel('texto mock')
    try:
        context.result = transcribe.transcribe_audio(context.audio_path)
    finally:
        transcribe.whisper.load_model = orig


@then('la respuesta contiene la clave "text"')
def step_then_result_has_text(context):
    assert isinstance(context.result, dict)
    assert 'text' in context.result


@given('`transcribe_audio` devuelve segmentos con timestamps')
def step_transcribe_audio_returns_segments(context):
    context.model_returns_segments = True


@when('solicito `transcribe_with_timestamps`')
def step_request_transcribe_with_timestamps(context):
    import tempfile
    # make temp audio
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tmp.write(b'RIFF')
    tmp.close()
    context.audio_path = tmp.name

    # patch model
    orig = transcribe.whisper.load_model
    transcribe.whisper.load_model = lambda size: DummyModel()
    try:
        context.segments = transcribe.transcribe_with_timestamps(context.audio_path)
    finally:
        transcribe.whisper.load_model = orig


@then('recibo una lista con objetos que contienen "start", "end" y "text"')
def step_then_receive_segments(context):
    assert isinstance(context.segments, list)
    seg = context.segments[0]
    assert 'start' in seg and 'end' in seg and 'text' in seg


@given('el texto "{text}"')
def step_given_text_simple(context, text):
    context.save_text = text


@when('llamo a `save_transcription` con "{out}"')
def step_call_save_transcription(context, out):
    import os
    p = os.path.abspath(out)
    transcribe.save_transcription(context.save_text, p)
    context.saved_path = p


@then('el fichero "{out}" debe existir con el texto')
def step_then_check_saved_simple(context, out):
    import os
    p = os.path.abspath(out)
    assert os.path.exists(p)
    content = open(p, 'r', encoding='utf-8').read()
    assert context.save_text in content


@given('el ejemplo usa `AUDIO_FILE` apuntando a un archivo')
def step_example_audio_file(context):
    import tempfile, os
    from src import example
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tmp.write(b'RIFF')
    tmp.close()
    example.AUDIO_FILE = tmp.name
    context.example_audio = tmp.name


@when('ejecuto `ejemplo_transcripcion_simple`')
def step_run_example_simple(context):
    from src import example
    # patch example functions to avoid heavy models
    orig_trans = getattr(example, 'transcribe_audio', None)
    orig_save = getattr(example, 'save_transcription', None)

    def fake_transcribe(audio_path, model_size='base', language=None):
        return {'text': 'texto ejemplo'}

    def fake_save(text, out):
        context.example_saved = True

    example.transcribe_audio = fake_transcribe
    example.save_transcription = fake_save

    try:
        example.ejemplo_transcripcion_simple()
    finally:
        if orig_trans is not None:
            example.transcribe_audio = orig_trans
        if orig_save is not None:
            example.save_transcription = orig_save


@then('imprime la transcripción y llama a `save_transcription` o muestra error si falta el fichero')
def step_example_assert_saved_or_error(context):
    # Accept success path where save called
    assert getattr(context, 'example_saved', False) is True


@given('`transcribe_with_timestamps` devuelve segmentos')
def step_transcribe_with_timestamps_returns(context):
    # Mark that the mocked transcribe_with_timestamps should return segments
    context.model_returns_segments = True


@when('ejecuto `ejemplo_transcripcion_con_timestamps`')
def step_run_example_timestamps(context):
    from src import example
    # create temp audio and point example to it
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tmp.write(b'RIFF')
    tmp.close()
    example.AUDIO_FILE = tmp.name

    # patch the transcribe_with_timestamps used by the example
    orig = getattr(example, 'transcribe_with_timestamps', None)

    def fake_transcribe_with_timestamps(path, model_size='base'):
        return [{'start': 0.0, 'end': 1.0, 'text': 'hola mundo'}]

    example.transcribe_with_timestamps = fake_transcribe_with_timestamps
    try:
        example.ejemplo_transcripcion_con_timestamps()
        context.example_segments_printed = True
    finally:
        if orig is not None:
            example.transcribe_with_timestamps = orig


@then('se imprimen los segmentos con formato de timestamps')
def step_assert_printed_timestamps(context):
    assert getattr(context, 'example_segments_printed', False) is True


@when('llamo a `transcribe_audio` con ese fichero')
def step_call_transcribe_audio(context):
    try:
        transcribe.transcribe_audio(context.audio_path)
        context.exc = None
    except Exception as e:
        context.exc = e


@then('recibo un `FileNotFoundError`')
def step_assert_file_not_found(context):
    assert context.exc is not None
    assert isinstance(context.exc, FileNotFoundError)
