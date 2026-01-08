class FakeTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeDiarization:
    def __init__(self, tracks):
        # tracks: list of (start,end,speaker)
        self._tracks = tracks

    def itertracks(self, yield_label=True):
        for s, e, speaker in self._tracks:
            yield FakeTurn(s, e), None, speaker


from src.diarize import get_speaker_for_segment


def test_get_speaker_for_segment():
    tracks = [
        (0.0, 1.0, 'SPEAKER_00'),
        (0.5, 2.0, 'SPEAKER_01'),
    ]
    diar = FakeDiarization(tracks)

    # Segment overlaps both; SPEAKER_01 has longer overlap in this range
    speaker = get_speaker_for_segment(diar, 0.8, 1.6)
    assert speaker == 'SPEAKER_01'
