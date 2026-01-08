import os
import runpy
import sys
from pathlib import Path


def test_transcribe_with_speaker_diarization_deletes_temp(monkeypatch, tmp_path):
    from src import diarize

    # create a temp file to act as normalized audio
    temp = tmp_path / 'norm.wav'
    temp.write_bytes(b'RIFF')

    # Monkeypatch normalize_audio_for_diarization to return our temp path
    monkeypatch.setattr(diarize, 'normalize_audio_for_diarization', lambda p: str(temp))

    # Ensure HF_TOKEN is present
    monkeypatch.setenv('HF_TOKEN', 'hf_fake')

    # Monkeypatch diarize.whisper to avoid real whisper audio decoding
    class DummyModel:
        def transcribe(self, audio_path, **opts):
            return {"text": "hola", "segments": [{"start": 0.0, "end": 1.0, "text": "hola"}]}

    fake_whisper = type('W', (), {'load_model': lambda size: DummyModel()})
    monkeypatch.setattr(diarize, 'whisper', fake_whisper)

    # Create a real audio input file
    audio = tmp_path / 'audio.mp3'
    audio.write_bytes(b'RIFF')

    segments = diarize.transcribe_with_speaker_diarization(str(audio), 'hf_fake', model_size='tiny')

    assert isinstance(segments, list)
    # temp should have been removed by the function
    assert not temp.exists()


def test_main_exits_when_no_token(monkeypatch):
    # Ensure HF_TOKEN is not set
    if 'HF_TOKEN' in os.environ:
        del os.environ['HF_TOKEN']

    # Run module as script with no args; it should sys.exit(1)
    monkeypatch.setattr(sys, 'argv', ['diarize.py'])
    import pytest
    with pytest.raises(SystemExit) as exc:
        runpy.run_module('src.diarize', run_name='__main__')
    assert exc.value.code == 1
