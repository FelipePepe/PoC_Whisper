from pathlib import Path
from src.transcribe import save_transcription, transcribe_with_timestamps


def test_save_transcription(tmp_path):
    out = tmp_path / "out.txt"
    save_transcription("hola mundo", str(out))
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "hola mundo"


def test_transcribe_with_timestamps_monkeypatched(monkeypatch):
    # Simulate transcribe_audio returning segments
    def fake_transcribe(audio_path, model_size="base", language=None):
        return {
            "segments": [
                {"start": 0.0, "end": 1.2, "text": "hola"},
                {"start": 1.3, "end": 2.0, "text": "mundo"},
            ]
        }

    monkeypatch.setattr("src.transcribe.transcribe_audio", fake_transcribe)

    segments = transcribe_with_timestamps("fake.mp3")
    assert isinstance(segments, list)
    assert segments[0]["text"] == "hola"
    # Compare floats with tolerance
    assert abs(segments[1]["start"] - 1.3) < 1e-6
