import os
import sys
from types import ModuleType
from pathlib import Path


def test_get_speaker_for_segment_simple():
    from src.diarize import get_speaker_for_segment

    class Turn:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class FakeDiarization:
        def __init__(self, turns):
            self._turns = turns

        def itertracks(self, yield_label=True):
            for t, s in self._turns:
                yield (t, None, s)

    turns = [ (Turn(0.0, 1.0), 'S1'), (Turn(1.0, 3.0), 'S2') ]
    diar = FakeDiarization(turns)

    # Segment overlapping first turn
    sp = get_speaker_for_segment(diar, 0.2, 0.8)
    assert sp == 'S1'

    # Segment overlapping second turn
    sp2 = get_speaker_for_segment(diar, 1.2, 2.0)
    assert sp2 == 'S2'

    # No overlap
    sp3 = get_speaker_for_segment(diar, 3.5, 4.0)
    assert sp3 == 'UNKNOWN'


def test_format_and_save_transcription(tmp_path):
    from src.diarize import format_transcription_by_speaker, save_diarized_transcription

    segments = [
        {'speaker': 'S1', 'start': 0.0, 'end': 1.0, 'text': 'hola '},
        {'speaker': 'S1', 'start': 1.0, 'end': 2.0, 'text': 'mundo'},
        {'speaker': 'S2', 'start': 2.0, 'end': 3.0, 'text': 'adios'},
    ]

    out = format_transcription_by_speaker(segments)
    assert 'S1' in out and 'hola' in out and 'mundo' in out and 'S2' in out

    grouped = tmp_path / 'g.txt'
    timestamped = tmp_path / 't.txt'

    save_diarized_transcription(segments, str(grouped), 'grouped')
    save_diarized_transcription(segments, str(timestamped), 'timestamped')

    assert grouped.exists()
    assert timestamped.exists()

    text_g = grouped.read_text(encoding='utf-8')
    text_t = timestamped.read_text(encoding='utf-8')
    assert 'S1' in text_g
    assert '[0.00s - 1.00s]' in text_t or '0.00' in text_t


def test_transcribe_with_speaker_diarization_monkeypatched(monkeypatch, tmp_path):
    # This test monkeypatches pyannote and whisper to exercise the main flow
    from src import diarize

    # Create a small temp file to act as normalized audio and to be cleaned up
    normalized = tmp_path / 'norm.wav'
    normalized.write_bytes(b'RIFF')

    # Patch normalize_audio_for_diarization to return our temp file
    monkeypatch.setattr(diarize, 'normalize_audio_for_diarization', lambda p: str(normalized))

    # Create a fake pyannote.audio module with Pipeline
    fake_module = ModuleType('pyannote.audio')

    class FakeDiarization:
        def __init__(self):
            self._turns = [ (type('T', (), {'start': 0.0, 'end': 1.0})(), 'S1') ]

        def itertracks(self, yield_label=True):
            for t, s in self._turns:
                yield (t, None, s)

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, audio, **kwargs):
            return FakeDiarization()

    fake_module.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, 'pyannote.audio', fake_module)

    # Patch whisper.load_model to return a fake model
    class FakeModel:
        def transcribe(self, audio_path, **options):
            return {'segments': [ {'start': 0.0, 'end': 1.0, 'text': 'hola'} ] }

    monkeypatch.setattr('src.diarize.whisper.load_model', lambda m: FakeModel())

    # Run the function; should return list with speaker assigned
    tmp_audio = tmp_path / 'audio.mp3'
    tmp_audio.write_bytes(b'RIFF')

    segments = diarize.transcribe_with_speaker_diarization(str(tmp_audio), 'hf_FAKE', model_size='tiny')
    assert isinstance(segments, list)
    assert segments[0]['speaker'] in ('S1', 'UNKNOWN')
