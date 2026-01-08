import os
import tempfile
import builtins
import pytest

from src import transcribe


class DummyModel:
    def __init__(self, text="hello world"):
        self._text = text

    def transcribe(self, audio_path, **opts):
        return {
            "text": self._text,
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
                {"start": 1.0, "end": 2.0, "text": "world"},
            ]
        }


def test_transcribe_audio_file_not_found():
    with pytest.raises(FileNotFoundError):
        transcribe.transcribe_audio("nonexistent_file.wav")


def test_transcribe_with_mocked_model(monkeypatch, tmp_path):
    # create a fake audio file
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    # mock whisper.load_model
    monkeypatch.setattr(transcribe.whisper, "load_model", lambda size: DummyModel("full text"))

    res = transcribe.transcribe_audio(str(audio), model_size="tiny", language="es")
    assert res["text"] == "full text"


def test_transcribe_with_timestamps_uses_segments(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(transcribe.whisper, "load_model", lambda size: DummyModel())

    segments = transcribe.transcribe_with_timestamps(str(audio))
    assert isinstance(segments, list)
    from pytest import approx
    assert segments[0]["start"] == approx(0.0)
    assert segments[1]["text"] == "world"


def test_save_transcription_writes_file(tmp_path):
    out = tmp_path / "out.txt"
    transcribe.save_transcription("hola mundo", str(out))
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "hola mundo"
