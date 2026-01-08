import builtins
import runpy
import sys
from types import ModuleType


def test_normalize_audio_torchaudio_missing(monkeypatch, tmp_path):
    from src.diarize import normalize_audio_for_diarization

    # Create a dummy audio file
    audio = tmp_path / 'a.mp3'
    audio.write_bytes(b'RIFF')

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'torchaudio':
            raise ImportError('no torchaudio')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    try:
        import pytest
        with pytest.raises(RuntimeError):
            normalize_audio_for_diarization(str(audio))
    finally:
        monkeypatch.setattr(builtins, '__import__', real_import)


def test_transcribe_with_speaker_diarization_forces_pipeline_to_cuda(monkeypatch, tmp_path):
    # Ensure torch.cuda.is_available returns True
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    # Prepare fake pyannote module with Pipeline that has .to()
    fake_module = ModuleType('pyannote.audio')

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            # Mark called
            self._moved = True

        def __call__(self, audio, **kwargs):
            # Return an object with itertracks
            class D:
                def itertracks(self, yield_label=True):
                    t = type('T', (), {'start': 0.0, 'end': 0.5})()
                    yield (t, None, 'S1')
            return D()

    fake_module.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, 'pyannote.audio', fake_module)

    # Patch normalize to return a temp file and whisper model
    monkeypatch.setattr('src.diarize.normalize_audio_for_diarization', lambda p: str(tmp_path / 'n.wav'))

    class FakeModel:
        def transcribe(self, audio_path, **options):
            return {'segments': [{'start': 0.0, 'end': 0.5, 'text': 'x'}]}

    # Inject a fake whisper module into the src.diarize module to avoid real model loading / CUDA init
    fake_whisper = ModuleType('whisper')
    fake_whisper.load_model = lambda m: FakeModel()
    import importlib
    md = importlib.import_module('src.diarize')
    monkeypatch.setattr(md, 'whisper', fake_whisper, raising=False)

    # Run
    tmp_audio = tmp_path / 'audio.mp3'
    tmp_audio.write_bytes(b'RIFF')

    from src.diarize import transcribe_with_speaker_diarization
    segs = transcribe_with_speaker_diarization(str(tmp_audio), 'hf_FAKE')
    assert isinstance(segs, list)


def test_main_no_args_exits(monkeypatch):
    # When invoked with no args, module should exit with usage
    monkeypatch.setattr(sys, 'argv', ['diarize.py'])
    import pytest
    with pytest.raises(SystemExit):
        runpy.run_module('src.diarize', run_name='__main__')


def test_main_missing_token_exits(monkeypatch, tmp_path):
    # When no HF token provided anywhere, should exit with error
    audio = tmp_path / 'a.mp3'
    audio.write_bytes(b'RIFF')
    monkeypatch.setenv('HF_TOKEN', '')
    monkeypatch.setattr(sys, 'argv', ['diarize.py', str(audio)])
    import pytest
    # Ensure transcribe is not actually called
    monkeypatch.setattr('src.diarize.transcribe_with_speaker_diarization', lambda *a, **k: [])
    with pytest.raises(SystemExit):
        runpy.run_module('src.diarize', run_name='__main__')
