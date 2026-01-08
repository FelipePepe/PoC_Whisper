import os
import tkinter as tk
from pathlib import Path

import pytest

from src.gui import WhisperGUI


def make_app(tmp_path):
    root = tk.Tk()
    root.withdraw()
    app = WhisperGUI(root)
    return root, app


def test_select_file_updates_state(monkeypatch, tmp_path):
    root, app = make_app(tmp_path)
    try:
        # Create a fake file and patch filedialog
        f = tmp_path / 'audio.wav'
        f.write_bytes(b'RIFF')
        monkeypatch.setattr('tkinter.filedialog.askopenfilename', lambda **k: str(f))

        app.select_file()
        assert app.audio_file == str(f)
        assert 'audio.wav' in app.file_label.cget('text')
        assert 'Archivo seleccionado' in app.status_bar.cget('text')
    finally:
        root.destroy()


def test_process_audio_simple_executes_callbacks(monkeypatch, tmp_path):
    root, app = make_app(tmp_path)
    try:
        # prepare
        f = tmp_path / 'audio.mp3'
        f.write_bytes(b'RIFF')
        app.audio_file = str(f)
        app.transcription_type.set('simple')

        # monkeypatch transcribe_audio used inside gui
        monkeypatch.setattr('src.gui.transcribe_audio', lambda a, m, l: {'text': 'hello world'})

        # make root.after execute callbacks immediately
        app.root.after = lambda ms, func, *args: func(*args)

        app.process_audio()

        # callbacks should have run
        content = app.result_text.get(1.0, 'end').strip()
        assert 'hello world' in content
        assert app.processing is False
    finally:
        root.destroy()


def test_cancel_transcription_confirms_and_finishes(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    try:
        app = WhisperGUI(root)
        # set processing state
        app.processing = True
        app.cancel_btn.config(state=tk.NORMAL)

        monkeypatch.setattr('tkinter.messagebox.askyesno', lambda *a, **k: True)

        app.cancel_transcription()

        assert app.processing is False
        # Tkinter may return a special index object; compare to string
        assert str(app.cancel_btn['state']) == str(tk.DISABLED)
    finally:
        root.destroy()
