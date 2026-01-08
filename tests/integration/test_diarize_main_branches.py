import runpy
import importlib
import sys
import types
from pathlib import Path


def test_diarize_main_with_hf_arg_and_multiple_speakers(monkeypatch, tmp_path):
    # Prepare audio
    audio = tmp_path / 'audio.mp3'
    audio.write_bytes(b'RIFF')

    # Ensure HF token passed as second arg (starts with hf_)
    monkeypatch.setattr(sys, 'argv', ['diarize.py', str(audio), 'hf_ABC', 'tiny', 'es', '2'])

    # Patch transcribe_with_speaker_diarization to return segments with two speakers
    from src import diarize
    monkeypatch.setattr(diarize, 'transcribe_with_speaker_diarization', lambda a, t, m, l, n: [
        {'start': 0.0, 'end': 1.0, 'speaker': 'S1', 'text': 'one'},
        {'start': 1.0, 'end': 2.0, 'speaker': 'S2', 'text': 'two'},
    ])

    # Run module main; should complete and generate files
    runpy.run_module('src.diarize', run_name='__main__')

    grouped = tmp_path / (audio.stem + '_diarized_grouped.txt')
    timestamped = tmp_path / (audio.stem + '_diarized_timestamped.txt')
    assert grouped.exists()
    assert timestamped.exists()


def test_diarize_main_exception_path(monkeypatch, tmp_path):
    # Prepare audio
    audio = tmp_path / 'audio.mp3'
    audio.write_bytes(b'RIFF')

    # ensure HF_TOKEN is not set in env for this test
    import os
    if 'HF_TOKEN' in os.environ:
        del os.environ['HF_TOKEN']

    # Patch argv to include audio but no token in env
    monkeypatch.setattr(sys, 'argv', ['diarize.py', str(audio)])

    # Patch transcribe_with_speaker_diarization to raise
    from src import diarize
    def raise_err(a, t, m, l, n):
        raise RuntimeError('fail')

    monkeypatch.setattr(diarize, 'transcribe_with_speaker_diarization', raise_err)

    import pytest
    with pytest.raises(SystemExit):
        runpy.run_module('src.diarize', run_name='__main__')


def test_reload_diarize_handles_torch_threads_exceptions(monkeypatch):
    # Simulate torch where set_num_threads raises
    import sys as _sys
    orig_torch = _sys.modules.get('torch')
    orig_module = _sys.modules.get('src.diarize')

    fake_torch = types.ModuleType('torch')
    def bad_set(n):
        raise RuntimeError('no threads')
    fake_torch.set_num_threads = bad_set
    fake_torch.set_num_interop_threads = bad_set
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    try:
        _sys.modules['torch'] = fake_torch
        if 'src.diarize' in _sys.modules:
            del _sys.modules['src.diarize']
        # Import fresh module; should handle exceptions during thread setting
        m = importlib.import_module('src.diarize')
        assert hasattr(m, 'transcribe_with_speaker_diarization')
    finally:
        # restore
        if orig_module is not None:
            _sys.modules['src.diarize'] = orig_module
        else:
            _sys.modules.pop('src.diarize', None)
        if orig_torch is not None:
            _sys.modules['torch'] = orig_torch
        else:
            _sys.modules.pop('torch', None)
