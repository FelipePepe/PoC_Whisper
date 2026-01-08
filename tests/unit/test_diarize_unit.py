import sys
import types
import os
import torch

import pytest

from src import diarize


def make_dummy_diar():
    class DummyTurn:
        def __init__(self, s, e):
            self.start = s
            self.end = e

    class DummyDiar:
        def __init__(self):
            self._tracks = [(DummyTurn(0.0, 1.0), None, 'SPEAKER_A'), (DummyTurn(1.0, 2.0), None, 'SPEAKER_B')]

        def itertracks(self, yield_label=True):
            for t in self._tracks:
                yield t

    return DummyDiar()


def test_get_speaker_for_segment_basic():
    diar = make_dummy_diar()
    # segment overlapping more with SPEAKER_A
    sp = diarize.get_speaker_for_segment(diar, 0.1, 0.9)
    assert sp == 'SPEAKER_A'


def test_get_speaker_for_segment_no_overlap():
    diar = make_dummy_diar()
    sp = diarize.get_speaker_for_segment(diar, 2.5, 3.0)
    assert sp == 'UNKNOWN'


def test_format_transcription_by_speaker():
    segments = [
        {'start': 0.0, 'end': 1.0, 'speaker': 'S1', 'text': 'Hello'},
        {'start': 1.0, 'end': 2.0, 'speaker': 'S1', 'text': ' world'},
        {'start': 2.0, 'end': 3.0, 'speaker': 'S2', 'text': 'Hi'},
    ]

    out = diarize.format_transcription_by_speaker(segments)
    assert 'S1' in out and 'Hello' in out and 'world' in out and 'S2' in out


def test_normalize_audio_for_diarization_mock(monkeypatch, tmp_path):
    # Create a fake torchaudio module and insert into sys.modules before calling
    fake_torchaudio = types.SimpleNamespace()

    # Prepare a waveform tensor: 2 channels, 160 samples
    waveform = torch.zeros((2, 160), dtype=torch.float32)
    sample_rate = 44100

    def fake_load(path):
        return waveform, sample_rate

    def fake_save(path, wave, sr):
        # write a small file so that the returned path exists
        with open(path, 'wb') as f:
            f.write(b'RIFF')

    fake_torchaudio.load = fake_load
    fake_torchaudio.save = fake_save
    fake_torchaudio.transforms = types.SimpleNamespace(Resample=lambda a, b: (lambda x: x))

    monkeypatch.setitem(sys.modules, 'torchaudio', fake_torchaudio)

    # Call normalize; it should return a path to a file that exists
    src_file = tmp_path / 'in.wav'
    src_file.write_bytes(b'RIFF')

    out = diarize.normalize_audio_for_diarization(str(src_file))
    assert os.path.exists(out)
    # cleanup
    os.unlink(out)
