import os
import tempfile
import tkinter as tk

from src.gui import WhisperGUI


def test_gui_update_and_clear(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    gui = WhisperGUI(root)

    gui.update_status('Testing')
    assert 'Testing' in gui.status_bar.cget('text')

    gui.show_result('hello')
    assert 'hello' in gui.result_text.get(1.0, tk.END)

    gui.clear_result()
    assert gui.result_text.get(1.0, tk.END).strip() == ''

    gui.finish_processing()
    assert gui.processing is False


def test_save_result_writes_file(monkeypatch, tmp_path):
    root = tk.Tk()
    root.withdraw()
    gui = WhisperGUI(root)

    # insert some text
    gui.show_result('line one')

    # monkeypatch asksaveasfilename to return a tmp path
    dest = tmp_path / 'out.txt'
    monkeypatch.setattr('tkinter.filedialog.asksaveasfilename', lambda **kw: str(dest))
    # monkeypatch messagebox to avoid dialogs
    monkeypatch.setattr('tkinter.messagebox.showinfo', lambda *a, **k: None)

    gui.save_result()
    assert dest.exists()
    assert 'line one' in dest.read_text(encoding='utf-8')
