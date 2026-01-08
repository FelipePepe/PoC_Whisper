import os
import tkinter as tk
from pathlib import Path

import pytest

from src.gui import WhisperGUI


def make_app(tmp_path):
    root = tk.Tk()
    # hide the root window in tests
    root.withdraw()
    app = WhisperGUI(root)
    return root, app


def test_show_result_and_clear():
    root, app = make_app(Path('.'))
    try:
        app.show_result('hola mundo')
        content = app.result_text.get(1.0, 'end').strip()
        assert 'hola mundo' in content

        app.clear_result()
        content2 = app.result_text.get(1.0, 'end').strip()
        assert content2 == ''
    finally:
        root.destroy()


def test_save_result_writes_file(monkeypatch, tmp_path):
    root, app = make_app(tmp_path)
    try:
        # Put some text in the result area
        app.show_result('texto de prueba')

        # Monkeypatch asksaveasfilename to return a temporary path
        out_path = tmp_path / 'out.txt'
        monkeypatch.setattr('tkinter.filedialog.asksaveasfilename', lambda **kw: str(out_path))

        # Monkeypatch messagebox calls to no-op
        monkeypatch.setattr('tkinter.messagebox.showinfo', lambda *a, **k: None)
        monkeypatch.setattr('tkinter.messagebox.showwarning', lambda *a, **k: None)
        monkeypatch.setattr('tkinter.messagebox.showerror', lambda *a, **k: None)

        app.save_result()

        assert out_path.exists()
        assert out_path.read_text(encoding='utf-8').strip() == 'texto de prueba'
    finally:
        root.destroy()


def test_start_transcription_validations(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    try:
        app = WhisperGUI(root)

        # No file selected -> should show warning (monkeypatch to avoid dialog)
        monkeypatch.setattr('tkinter.messagebox.showwarning', lambda *a, **k: None)
        app.audio_file = None
        app.start_transcription()

        # File selected but missing -> showerror
        monkeypatch.setattr('tkinter.messagebox.showerror', lambda *a, **k: None)
        app.audio_file = 'no_existe.wav'
        app.start_transcription()

        # Diarization without token -> showerror
        app.audio_file = __file__  # exists
        app.transcription_type.set('diarization')
        monkeypatch.setenv('HF_TOKEN', '')
        app.start_transcription()
        # Now simulate a successful diarization run by patching heavy functions
        monkeypatch.setenv('HF_TOKEN', 'hf_ok')
        from src import transcribe, diarize

        monkeypatch.setattr('src.gui.transcribe_audio', lambda a, m, l: {'text': 't'})
        monkeypatch.setattr('src.gui.transcribe_with_speaker_diarization', lambda a, t, m, l: [
            {'start': 0.0, 'end': 1.0, 'speaker': 'S1', 'text': 'Hi'}
        ])

        # Call process_audio directly to exercise diarization branch
        app.audio_file = __file__
        app.transcription_type.set('diarization')
        app.process_audio()
    finally:
        root.destroy()
