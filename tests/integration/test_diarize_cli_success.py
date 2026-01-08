import runpy
import sys
import os
from pathlib import Path


def test_diarize_main_creates_output_files(monkeypatch, tmp_path):
    # Prepare temp working dir
    monkeypatch.chdir(tmp_path)

    audio = tmp_path / 'myaudio.mp3'
    audio.write_bytes(b'RIFF')

    # Ensure HF_TOKEN present
    monkeypatch.setenv('HF_TOKEN', 'hf_token')

    # Monkeypatch functions used inside main to deterministic behavior
    from src import diarize

    # replace transcribe_with_speaker_diarization to return deterministic segments
    monkeypatch.setattr(diarize, 'transcribe_with_speaker_diarization', lambda a, t, model_size='base', language=None, num_speakers=None: [
        {'start': 0.0, 'end': 1.0, 'speaker': 'S1', 'text': 'Hello'}
    ])

    # Run module as script with args: audio file and explicit token
    monkeypatch.setattr(sys, 'argv', ['diarize.py', str(audio), 'hf_token', 'tiny', 'es'])

    # Execute module; should complete without raising
    runpy.run_module('src.diarize', run_name='__main__')

    # Check output files created in tmp_path
    grouped = tmp_path / (audio.stem + '_diarized_grouped.txt')
    timestamped = tmp_path / (audio.stem + '_diarized_timestamped.txt')

    assert grouped.exists()
    assert timestamped.exists()
