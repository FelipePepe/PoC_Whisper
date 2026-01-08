import runpy
import sys
from pathlib import Path


def test_transcribe_main_no_args_exits(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['transcribe.py'])
    import pytest
    with pytest.raises(SystemExit):
        runpy.run_module('src.transcribe', run_name='__main__')


def test_transcribe_main_success(monkeypatch, tmp_path, capsys):
    # Prepare input audio
    audio = tmp_path / 'a.mp3'
    audio.write_bytes(b'RIFF')
    monkeypatch.setattr(sys, 'argv', ['transcribe.py', str(audio)])

    # Patch transcribe_audio to avoid heavy model
    def fake_transcribe(audio_path, model_size='base', language=None):
        return {'text': 'ok main'}

    monkeypatch.setattr('src.transcribe.transcribe_audio', fake_transcribe)

    runpy.run_module('src.transcribe', run_name='__main__')
    captured = capsys.readouterr()
    assert 'TRANSCRIPCIÓN' in captured.out
    # Check output file exists in cwd
    out = Path.cwd() / (audio.stem + '_transcripcion.txt')
    if out.exists():
        out.unlink()


def test_transcribe_main_exception_triggers_exit(monkeypatch):
    import pytest
    monkeypatch.setattr(sys, 'argv', ['transcribe.py', 'somefile.mp3'])

    def raise_err(*a, **k):
        raise RuntimeError('boom')

    monkeypatch.setattr('src.transcribe.transcribe_audio', raise_err)
    with pytest.raises(SystemExit):
        runpy.run_module('src.transcribe', run_name='__main__')
