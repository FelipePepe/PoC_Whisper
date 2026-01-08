import os
import types
from pathlib import Path


def test_pipeline_to_called_and_num_speakers(monkeypatch, tmp_path):
    from src import diarize

    # create a temp normalized audio
    tmp = tmp_path / 'norm.wav'
    tmp.write_bytes(b'RIFF')

    monkeypatch.setattr(diarize, 'normalize_audio_for_diarization', lambda p: str(tmp))

    # fake pipeline object that records calls
    recorded = {}

    class FakePipeline:
        def __init__(self):
            # no-op initializer for fake pipeline
            return

        def to(self, device):
            recorded['to'] = str(device)

        def __call__(self, audio_path, **kwargs):
            recorded['called_with'] = kwargs
            # return object with itertracks
            class D:
                def itertracks(self, yield_label=True):
                    class Turn:
                        def __init__(self):
                            self.start = 0
                            self.end = 1

                    yield (Turn(), None, 'S1')

            return D()

    import sys as _sys
    fake_audio_mod = types.ModuleType('pyannote.audio')
    # Provide Pipeline with from_pretrained
    fake_audio_mod.Pipeline = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakePipeline())
    fake_pkg = types.ModuleType('pyannote')
    fake_pkg.audio = fake_audio_mod
    _sys.modules['pyannote'] = fake_pkg
    _sys.modules['pyannote.audio'] = fake_audio_mod

    # make torch.cuda.is_available return True
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    # monkeypatch whisper model to avoid ffmpeg
    class DummyModel:
        def transcribe(self, audio_path, **opts):
            return {'segments': [{'start': 0.0, 'end': 1.0, 'text': 'hi'}], 'text': 'hi'}

    monkeypatch.setattr(diarize, 'whisper', types.SimpleNamespace(load_model=lambda s: DummyModel()))

    audio = tmp_path / 'audio.mp3'
    audio.write_bytes(b'RIFF')

    segments = diarize.transcribe_with_speaker_diarization(str(audio), 'hf', model_size='tiny', num_speakers=2)

    assert isinstance(segments, list)
    assert 'to' in recorded
    assert 'num_speakers' in recorded.get('called_with', {})
