from pathlib import Path


def test_ejemplo_transcripcion_con_timestamps_runs(monkeypatch, tmp_path):
    import src.example as example

    f = tmp_path / 'audio.mp3'
    f.write_bytes(b'RIFF')
    example.AUDIO_FILE = str(f)

    # patch transcribe_with_timestamps to return segments
    monkeypatch.setattr('src.example.transcribe_with_timestamps', lambda p, model_size='base': [{'start': 0.0, 'end': 1.0, 'text': 'hola'}])

    example.ejemplo_transcripcion_con_timestamps()


def test_ejemplo_transcripcion_con_idioma_runs(monkeypatch, tmp_path):
    import src.example as example

    f = tmp_path / 'audio.mp3'
    f.write_bytes(b'RIFF')
    example.AUDIO_FILE = str(f)

    monkeypatch.setattr('src.example.transcribe_audio', lambda a, model_size='base', language=None: {'text': 'hola en es'})

    example.ejemplo_transcripcion_con_idioma()


def test_run_example_module_main(monkeypatch, tmp_path):
    import runpy
    import src.example as example

    # prepare audio and patch functions used by example main
    f = tmp_path / 'audio.mp3'
    f.write_bytes(b'RIFF')
    example.AUDIO_FILE = str(f)

    monkeypatch.setattr('src.example.transcribe_audio', lambda a, model_size='base', language=None: {'text': 'MAIN'})
    monkeypatch.setattr('src.example.transcribe_with_timestamps', lambda p, model_size='base': [{'start':0.0,'end':1.0,'text':'x'}])

    # Running module as __main__ should not raise
    runpy.run_module('src.example', run_name='__main__')
