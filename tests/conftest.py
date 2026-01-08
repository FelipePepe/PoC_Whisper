import os
import sys
from pathlib import Path
import types

import pytest

# Ensure project root is on sys.path so `import src` works in tests
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def fake_whisper(monkeypatch):
    """Provide a lightweight fake for whisper.load_model to avoid heavy imports."""
    fake = types.SimpleNamespace()

    class DummyModel:
        def __init__(self, text="texto"):
            self._text = text

        def transcribe(self, audio_path, **opts):
            return {"text": self._text, "segments": [{"start": 0.0, "end": 1.0, "text": "hola"}]}

    fake.load_model = lambda size: DummyModel()
    monkeypatch.setitem(sys.modules, 'whisper', fake)
    yield fake


@pytest.fixture(autouse=True)
def fake_pyannote(monkeypatch):
    """Provide a fake pyannote Pipeline to avoid network/large models in tests."""
    fake = types.SimpleNamespace()

    class DummyPipeline:
        def __init__(self):
            # No initialization required for the fake pipeline
            return

        def __call__(self, audio, **kwargs):
            # Return an object with itertracks method
            class DummyDiar:
                def __init__(self):
                    class Turn:
                        def __init__(self, s, e):
                            self.start = s
                            self.end = e

                    self._tracks = [(Turn(0, 1), None, 'SPEAKER_00')]

                def itertracks(self, yield_label=True):
                    for t in self._tracks:
                        yield t

            return DummyDiar()

    fake.Pipeline = types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyPipeline())
    monkeypatch.setitem(sys.modules, 'pyannote', types.SimpleNamespace(audio=fake))
    monkeypatch.setitem(sys.modules, 'pyannote.audio', fake)
    yield fake


@pytest.fixture(autouse=True)
def fake_torchaudio(monkeypatch):
    """Provide a simple torchaudio replacement used in normalization tests."""
    import types
    fake = types.SimpleNamespace()

    def load(path):
        # Return dummy waveform and sample_rate
        import torch
        return torch.zeros((1, 160)), 16000

    def save(path, waveform, sr):
        with open(path, 'wb') as f:
            f.write(b'RIFF')

    fake.load = load
    fake.save = save
    fake.transforms = types.SimpleNamespace(Resample=lambda a, b: (lambda x: x))
    monkeypatch.setitem(sys.modules, 'torchaudio', fake)
    yield fake
import sys
from pathlib import Path

# Ensure project root is on sys.path so `import src` works in tests
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
