import os
import tempfile
import pytest

from src import transcribe


class RecordingModel:
    def __init__(self):
        self.called_with = None

    def transcribe(self, audio_path, **opts):
        # record the options passed for assertions
        self.called_with = {'audio_path': audio_path, 'opts': opts}
        return {"text": "ok", "segments": []}


def test_transcribe_passes_language_option(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    rec = RecordingModel()
    monkeypatch.setattr(transcribe.whisper, "load_model", lambda size: rec)

    # call with explicit language
    res = transcribe.transcribe_audio(str(audio), model_size="tiny", language="es")
    assert res["text"] == "ok"
    assert rec.called_with is not None
    assert rec.called_with['opts'].get('language') == 'es'


def test_transcribe_without_language(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    rec = RecordingModel()
    monkeypatch.setattr(transcribe.whisper, "load_model", lambda size: rec)

    res = transcribe.transcribe_audio(str(audio), model_size="tiny", language=None)
    assert res["text"] == "ok"
    # language key should not be present
    assert 'language' not in rec.called_with['opts']


def test_transcribe_load_model_failure(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    def bad_load(size):
        raise RuntimeError("fail load")

    monkeypatch.setattr(transcribe.whisper, "load_model", bad_load)

    with pytest.raises(RuntimeError):
        transcribe.transcribe_audio(str(audio))


def test_transcribe_main_creates_output(monkeypatch, tmp_path):
    # prepare audio and model
    audio = tmp_path / 'in.mp3'
    audio.write_bytes(b'RIFF')

    class M:
        def transcribe(self, ap, **opts):
            return {'text': 'resultado', 'segments': []}

    monkeypatch.setattr(transcribe.whisper, 'load_model', lambda s: M())

    import runpy, sys
    old_argv = sys.argv
    try:
        sys.argv = ['src.transcribe', str(audio), 'tiny', 'es']
        runpy.run_module('src.transcribe', run_name='__main__')
        # The module writes output to cwd; ensure file exists in cwd
        expected = f"{audio.stem}_transcripcion.txt"
        assert os.path.exists(expected)
        # cleanup
        os.unlink(expected)
    finally:
        sys.argv = old_argv


def test_transcribe_main_missing_args_exits():
    import runpy, sys
    old_argv = sys.argv
    try:
        sys.argv = ['src.transcribe']
        with pytest.raises(SystemExit):
            runpy.run_module('src.transcribe', run_name='__main__')
    finally:
        sys.argv = old_argv
