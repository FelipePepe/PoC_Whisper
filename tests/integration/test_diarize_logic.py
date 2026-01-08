import tempfile
import os
import pytest
import torch

from src.diarize import get_speaker_for_segment, format_transcription_by_speaker, save_diarized_transcription


class DummyTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class DummyDiarization:
    def __init__(self, tracks):
        # tracks is list of tuples (turn, _, speaker)
        self._tracks = tracks

    def itertracks(self, yield_label=True):
        for t in self._tracks:
            yield t


def test_get_speaker_for_segment_overlaps():
    # speaker A covers 0-2, speaker B covers 2-4
    tracks = [ (DummyTurn(0,2), None, 'SPEAKER_00'), (DummyTurn(2,4), None, 'SPEAKER_01') ]
    diar = DummyDiarization(tracks)

    # segment that overlaps more with SPEAKER_00
    speaker = get_speaker_for_segment(diar, 0.5, 1.5)
    assert speaker == 'SPEAKER_00'

    # unknown when no overlap
    speaker2 = get_speaker_for_segment(diar, 4.5, 5.0)
    assert speaker2 == 'UNKNOWN'


def test_format_and_save_transcription(tmp_path):
    segments = [
        {'start':0.0, 'end':1.0, 'speaker':'SPEAKER_00', 'text':'Hola '},
        {'start':1.0, 'end':2.0, 'speaker':'SPEAKER_00', 'text':'mundo'},
        {'start':2.0, 'end':3.0, 'speaker':'SPEAKER_01', 'text':'Adios'},
    ]

    formatted = format_transcription_by_speaker(segments)
    assert 'SPEAKER_00' in formatted
    assert 'Hola ' in formatted

    out = tmp_path / 'diar.txt'
    save_diarized_transcription(segments, str(out), format_type='grouped')
    assert out.exists()
    content = out.read_text(encoding='utf-8')
    assert 'SPEAKER_01' in content
