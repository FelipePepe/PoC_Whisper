import sys
from pathlib import Path


def test_ejemplo_transcripcion_simple_success(monkeypatch, tmp_path, capsys):
    from src import example

    # Preparar archivo de audio ficticio
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"RIFF")

    # Forzar que el módulo use nuestro archivo
    monkeypatch.setattr(example, 'AUDIO_FILE', str(audio))

    # Mockear transcribe_audio y save_transcription
    def fake_transcribe(audio_path, model_size="base", language=None):
        assert audio_path == str(audio)
        return {"text": "hola mundo"}

    saved = {}

    def fake_save(text, output_path):
        saved['path'] = output_path
        saved['text'] = text

    monkeypatch.setattr('src.example.transcribe_audio', fake_transcribe)
    monkeypatch.setattr('src.example.save_transcription', fake_save)

    # Ejecutar ejemplo
    example.ejemplo_transcripcion_simple()

    captured = capsys.readouterr()
    assert "hola mundo" in captured.out
    assert saved.get('text') == "hola mundo"


def test_ejemplo_transcripcion_simple_file_not_found(monkeypatch, tmp_path, capsys):
    from src import example

    # Usar un archivo que no existe
    audio = tmp_path / "no_existe.mp3"
    monkeypatch.setattr(example, 'AUDIO_FILE', str(audio))

    # Hacer que transcribe lance FileNotFoundError
    def raise_not_found(audio_path, model_size="base", language=None):
        raise FileNotFoundError()

    monkeypatch.setattr('src.example.transcribe_audio', raise_not_found)

    example.ejemplo_transcripcion_simple()
    captured = capsys.readouterr()
    assert "ERROR: No se encontró el archivo" in captured.out


def test_ejemplo_transcripcion_con_timestamps(monkeypatch, tmp_path, capsys):
    from src import example

    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(example, 'AUDIO_FILE', str(audio))

    def fake_with_timestamps(audio_path, model_size="base"):
        return [
            {'start': 0.0, 'end': 1.0, 'text': 'uno'},
            {'start': 1.0, 'end': 2.5, 'text': 'dos'},
        ]

    monkeypatch.setattr('src.example.transcribe_with_timestamps', fake_with_timestamps)

    example.ejemplo_transcripcion_con_timestamps()
    captured = capsys.readouterr()
    assert "[0.00s - 1.00s]: uno" in captured.out or "[0.00s - 1.00s]" in captured.out


def test_ejemplo_transcripcion_con_idioma_success(monkeypatch, tmp_path, capsys):
    from src import example

    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(example, 'AUDIO_FILE', str(audio))

    def fake_transcribe(audio_path, model_size="base", language=None):
        assert language == 'es'
        return {"text": "hola español"}

    monkeypatch.setattr('src.example.transcribe_audio', fake_transcribe)

    example.ejemplo_transcripcion_con_idioma()
    captured = capsys.readouterr()
    assert "hola español" in captured.out


def test_ejemplo_transcripcion_con_idioma_file_not_found(monkeypatch, tmp_path, capsys):
    from src import example

    audio = tmp_path / "no.mp3"
    monkeypatch.setattr(example, 'AUDIO_FILE', str(audio))

    def raise_not_found(audio_path, model_size="base", language=None):
        raise FileNotFoundError()

    monkeypatch.setattr('src.example.transcribe_audio', raise_not_found)
    example.ejemplo_transcripcion_con_idioma()
    captured = capsys.readouterr()
    assert "ERROR: No se encontró" in captured.out
import os
from importlib import reload


def test_ejemplo_transcripcion_simple_handles_missing(monkeypatch, tmp_path):
    # Import the module fresh
    import src.example as example

    # Ensure AUDIO_FILE points to a non-existent file
    example.AUDIO_FILE = str(tmp_path / "no_existe.mp3")

    # Call the example; it should handle FileNotFoundError and not raise
    example.ejemplo_transcripcion_simple()


def test_ejemplo_transcripcion_with_mocked_transcribe(monkeypatch, tmp_path):
    import src.example as example

    called = {}

    def fake_transcribe(audio_file, model_size="base", language=None):
        called['audio'] = audio_file
        return {"text": "hola mundo", "segments": []}

    # Patch the transcribe function used by the example module
    monkeypatch.setattr('src.example.transcribe_audio', fake_transcribe)

    # Point AUDIO_FILE to a temp file so the example tries to transcribe
    f = tmp_path / "audio.mp3"
    f.write_bytes(b'RIFF')
    example.AUDIO_FILE = str(f)

    # Should run without raising
    example.ejemplo_transcripcion_simple()
    assert called.get('audio') == str(f)


def test_ejemplo_transcripcion_with_timestamps(monkeypatch, tmp_path):
    import src.example as example

    # Patch transcribe_with_timestamps to return deterministic segments
    monkeypatch.setattr('src.example.transcribe_with_timestamps', lambda p, model_size='base': [{'start': 0.0, 'end': 1.0, 'text': 'hola'}])

    f = tmp_path / 'audio.mp3'
    f.write_bytes(b'RIFF')
    example.AUDIO_FILE = str(f)

    # Should run without error
    example.ejemplo_transcripcion_con_timestamps()


def test_ejemplo_transcripcion_con_idioma(monkeypatch, tmp_path):
    import src.example as example

    # Patch transcribe_audio to return controlled text
    monkeypatch.setattr('src.example.transcribe_audio', lambda a, model_size='base', language=None: {'text': 'hola en español'})

    f = tmp_path / 'audio.mp3'
    f.write_bytes(b'RIFF')
    example.AUDIO_FILE = str(f)

    example.ejemplo_transcripcion_con_idioma()
