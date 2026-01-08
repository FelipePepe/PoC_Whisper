import tkinter as tk


def test_save_result_error_path(monkeypatch, tmp_path):
    root = tk.Tk()
    root.withdraw()
    try:
        from src.gui import WhisperGUI
        app = WhisperGUI(root)

        # set empty text -> warning branch
        app.result_text.delete(1.0, 'end')
        monkeypatch.setattr('tkinter.messagebox.showwarning', lambda *a, **k: None)
        app.save_result()

        # now set text and make asksaveasfilename return a path but writing fails
        app.show_result('texto')
        out = tmp_path / 'bad.txt'
        def fake_saveas(**kw):
            return str(out)

        monkeypatch.setattr('tkinter.filedialog.asksaveasfilename', fake_saveas)
        # monkeypatch open to throw when used by save_result
        import builtins
        def fake_open(*a, **k):
            raise IOError('disk full')

        monkeypatch.setattr(builtins, 'open', fake_open)
        monkeypatch.setattr('tkinter.messagebox.showerror', lambda *a, **k: None)

        app.save_result()

    finally:
        root.destroy()


def test_process_audio_handles_exceptions(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    try:
        from src.gui import WhisperGUI
        app = WhisperGUI(root)

        app.audio_file = __file__
        app.transcription_type.set('simple')

        # patch transcribe_audio to raise
        def raise_exc(a, m, l):
            raise RuntimeError('boom')
        monkeypatch.setattr('src.gui.transcribe_audio', raise_exc)

        # make after immediate
        app.root.after = lambda ms, func, *args: func(*args)

        called = {}
        monkeypatch.setattr('tkinter.messagebox.showerror', lambda title, msg: called.setdefault('err', msg))

        app.process_audio()

        assert 'err' in called
    finally:
        root.destroy()


def test_cancel_transcription_no_confirm(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    try:
        from src.gui import WhisperGUI
        app = WhisperGUI(root)
        app.processing = True
        monkeypatch.setattr('tkinter.messagebox.askyesno', lambda *a, **k: False)
        app.cancel_transcription()
        # processing should remain True because user cancelled the cancel
        assert app.processing is True
    finally:
        root.destroy()
