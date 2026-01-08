import os
from pathlib import Path


def test_ejemplo_diarizacion_no_token(monkeypatch, capsys):
    from src import example_diarization

    # Ensure no HF_TOKEN
    monkeypatch.delenv('HF_TOKEN', raising=False)
    example_diarization.ejemplo_diarizacion()
    captured = capsys.readouterr()
    assert 'Token de HuggingFace no encontrado' in captured.out


def test_ejemplo_diarizacion_file_not_found(monkeypatch, tmp_path, capsys):
    from src import example_diarization

    # Set token
    monkeypatch.setenv('HF_TOKEN', 'hf_FAKE')

    # Make transcribe raise FileNotFoundError
    def raise_fn(*a, **k):
        raise FileNotFoundError()

    monkeypatch.setattr('src.example_diarization.transcribe_with_speaker_diarization', raise_fn)

    example_diarization.ejemplo_diarizacion()
    captured = capsys.readouterr()
    assert 'No se encontró el archivo' in captured.out or 'ERROR' in captured.out


def test_ejemplo_diarizacion_success(monkeypatch, tmp_path, capsys):
    from src import example_diarization

    monkeypatch.setenv('HF_TOKEN', 'hf_FAKE')

    segments = [ {'speaker': 'S1', 'start': 0.0, 'end': 1.0, 'text': 'hola'} ]

    called = {'saved': []}

    def fake_transcribe(*a, **k):
        return segments

    def fake_save(segments_arg, path, fmt=None):
        called['saved'].append((path, fmt))

    monkeypatch.setattr('src.example_diarization.transcribe_with_speaker_diarization', fake_transcribe)
    monkeypatch.setattr('src.example_diarization.save_diarized_transcription', fake_save)

    example_diarization.ejemplo_diarizacion()
    captured = capsys.readouterr()
    assert 'RESULTADO' in captured.out
    assert len(called['saved']) >= 1


def test_ejemplo_diarizacion_generic_exception(monkeypatch, tmp_path, capsys):
    from src import example_diarization

    monkeypatch.setenv('HF_TOKEN', 'hf_FAKE')

    def raise_any(*a, **k):
        raise ValueError('boom')

    monkeypatch.setattr('src.example_diarization.transcribe_with_speaker_diarization', raise_any)
    example_diarization.ejemplo_diarizacion()
    captured = capsys.readouterr()
    assert 'ERROR' in captured.out
import os


def test_ejemplo_diarizacion_returns_when_no_token(monkeypatch, tmp_path):
    # Ensure HF_TOKEN is not set
    if 'HF_TOKEN' in os.environ:
        del os.environ['HF_TOKEN']

    import src.example_diarization as ex_diar

    # Create a fake audio file referenced by the example to avoid FileNotFound handling
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b'RIFF')

    # Monkeypatch the file name constant
    ex_diar.AUDIO_FILE = str(audio_path)

    # The function should return early (no token) and not raise
    res = ex_diar.ejemplo_diarizacion()
    assert res is None


def test_ejemplo_diarizacion_with_mocked_pipeline(monkeypatch, tmp_path):
    # Provide a fake token and mock the heavy functions used by the example
    os.environ['HF_TOKEN'] = 'hf_fake'

    import src.example_diarization as ex_diar

    # Patch the transcribe_with_speaker_diarization to return a minimal result
    def fake_transcribe_with_speaker_diarization(audio_path, hf_token, model_size, language, num_speakers):
        return [
            {'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00', 'text': 'Hola'}
        ]

    monkeypatch.setattr('src.example_diarization.transcribe_with_speaker_diarization', fake_transcribe_with_speaker_diarization)

    # Create dummy audio file and run
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b'RIFF')
    ex_diar.AUDIO_FILE = str(audio_path)

    ex_diar.ejemplo_diarizacion()
