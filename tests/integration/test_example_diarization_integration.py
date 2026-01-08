import os


def test_example_diarization_integration_no_token(tmp_path):
    # Integration-style test: ensure example handles missing HF_TOKEN gracefully
    if 'HF_TOKEN' in os.environ:
        del os.environ['HF_TOKEN']

    import src.example_diarization as ex_diar

    # Ensure audio file exists so the function reaches the token check
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b'RIFF')
    ex_diar.AUDIO_FILE = str(audio_path)

    # Should return early without raising
    assert ex_diar.ejemplo_diarizacion() is None
