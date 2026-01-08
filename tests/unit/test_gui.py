import tkinter as tk
from pathlib import Path


def make_gui(monkeypatch):
    from src.gui import WhisperGUI
    root = tk.Tk()
    root.withdraw()
    gui = WhisperGUI(root)
    return gui, root


def test_start_transcription_no_file_shows_warning(monkeypatch):
    from src import gui as gui_mod
    gui, root = make_gui(monkeypatch)

    called = {}

    def fake_warn(title, msg):
        called['warn'] = (title, msg)

    monkeypatch.setattr(gui_mod, 'messagebox', type('M', (), {'showwarning': staticmethod(fake_warn)}))

    gui.audio_file = None
    gui.start_transcription()
    assert 'warn' in called
    root.destroy()


def test_start_transcription_nonexistent_file_shows_error(monkeypatch):
    from src import gui as gui_mod
    gui, root = make_gui(monkeypatch)

    called = {}

    def fake_error(title, msg):
        called['error'] = (title, msg)

    monkeypatch.setattr(gui_mod, 'messagebox', type('M', (), {'showerror': staticmethod(fake_error)}))

    gui.audio_file = '/no/such/file.mp3'
    gui.start_transcription()
    assert 'error' in called
    root.destroy()


def test_save_result_no_text_shows_warning(monkeypatch):
    from src import gui as gui_mod
    gui, root = make_gui(monkeypatch)

    called = {}

    def fake_warn(title, msg):
        called['warn'] = (title, msg)

    monkeypatch.setattr(gui_mod, 'messagebox', type('M', (), {'showwarning': staticmethod(fake_warn)}))

    # Ensure result text is empty
    gui.result_text.delete(1.0, tk.END)
    gui.save_result()
    assert 'warn' in called
    root.destroy()


def test_save_result_success_writes_file(monkeypatch, tmp_path):
    from src import gui as gui_mod
    gui, root = make_gui(monkeypatch)

    # Put some text
    gui.result_text.delete(1.0, tk.END)
    gui.result_text.insert(1.0, 'contenido')

    saved = tmp_path / 'out.txt'

    def fake_asksaveasfilename(**kwargs):
        return str(saved)

    info_called = {}

    def fake_info(title, msg):
        info_called['info'] = (title, msg)

    # Patch filedialog and messagebox
    monkeypatch.setattr(gui_mod.filedialog, 'asksaveasfilename', fake_asksaveasfilename)
    monkeypatch.setattr(gui_mod, 'messagebox', type('M', (), {'showinfo': staticmethod(fake_info)}))

    gui.save_result()
    assert saved.exists()
    assert 'info' in info_called
    root.destroy()


def test_cancel_transcription_confirms_and_finishes(monkeypatch):
    from src import gui as gui_mod
    gui, root = make_gui(monkeypatch)

    # Simulate user confirming cancellation
    monkeypatch.setattr(gui_mod.messagebox, 'askyesno', lambda t, m: True)

    gui.processing = True
    gui.cancel_transcription()
    assert gui.processing is False
    root.destroy()


def test_select_file_updates_label(monkeypatch, tmp_path):
    from src import gui as gui_mod
    gui, root = make_gui(monkeypatch)

    audio = tmp_path / 'a.mp3'
    audio.write_bytes(b'RIFF')

    monkeypatch.setattr(gui_mod.filedialog, 'askopenfilename', lambda **k: str(audio))
    gui.select_file()
    assert gui.audio_file == str(audio)
    assert 'a.mp3' in gui.file_label.cget('text')
    root.destroy()


def test_save_result_write_error_shows_error(monkeypatch, tmp_path):
    from src import gui as gui_mod
    gui, root = make_gui(monkeypatch)

    gui.result_text.delete(1.0, tk.END)
    gui.result_text.insert(1.0, 'contenido')

    saved = tmp_path / 'out.txt'

    monkeypatch.setattr(gui_mod.filedialog, 'asksaveasfilename', lambda **k: str(saved))

    def fake_open(*a, **k):
        raise OSError('disk full')

    errors = {}
    def fake_error(title, msg):
        errors['err'] = (title, msg)

    monkeypatch.setattr('builtins.open', fake_open)
    monkeypatch.setattr(gui_mod, 'messagebox', type('M', (), {'showerror': staticmethod(fake_error)}))

    gui.save_result()
    assert 'err' in errors
    root.destroy()


def test_process_audio_simple_updates_ui(monkeypatch, tmp_path):
    from src import gui as gui_mod
    gui, root = make_gui(monkeypatch)

    # Prepare audio file and set type to simple
    audio = tmp_path / 'a.mp3'
    audio.write_bytes(b'RIFF')
    gui.audio_file = str(audio)
    gui.transcription_type.set('simple')

    # Patch transcribe_audio
    monkeypatch.setattr('src.gui.transcribe_audio', lambda a, m, l: {'text': 'hola-ui'})

    # Make root.after call functions synchronously
    gui.root.after = lambda delay, func, *args: func(*args)

    gui.process_audio()
    assert 'hola-ui' in gui.result_text.get(1.0, 'end')
    root.destroy()


def test_process_audio_diarization_updates_ui(monkeypatch, tmp_path):
    from src import gui as gui_mod
    gui, root = make_gui(monkeypatch)

    audio = tmp_path / 'a.mp3'
    audio.write_bytes(b'RIFF')
    gui.audio_file = str(audio)
    gui.transcription_type.set('diarization')

    # Ensure HF_TOKEN present
    monkeypatch.setenv('HF_TOKEN', 'hf_FAKE')

    # Patch diarization funcs
    monkeypatch.setattr('src.gui.transcribe_with_speaker_diarization', lambda a, t, m, l: [ {'speaker':'S1','start':0,'end':1,'text':'x'} ])
    monkeypatch.setattr('src.gui.format_transcription_by_speaker', lambda segs: 'S1: x')

    gui.root.after = lambda delay, func, *args: func(*args)

    gui.process_audio()
    assert 'S1: x' in gui.result_text.get(1.0, 'end')
    root.destroy()


def test_start_transcription_diarization_no_token_shows_error(monkeypatch, tmp_path):
    from src import gui as gui_mod
    gui, root = make_gui(monkeypatch)

    audio = tmp_path / 'a.mp3'
    audio.write_bytes(b'RIFF')
    gui.audio_file = str(audio)
    gui.transcription_type.set('diarization')

    errors = {}
    def fake_error(title, msg):
        errors['err'] = (title, msg)

    monkeypatch.setenv('HF_TOKEN', '')
    monkeypatch.setattr(gui_mod, 'messagebox', type('M', (), {'showerror': staticmethod(fake_error)}))

    gui.start_transcription()
    assert 'err' in errors
    root.destroy()


def test_cancel_transcription_user_declines(monkeypatch):
    from src import gui as gui_mod
    gui, root = make_gui(monkeypatch)

    # User says No
    monkeypatch.setattr(gui_mod.messagebox, 'askyesno', lambda t, m: False)
    gui.processing = True
    gui.cancel_transcription()
    # Since user declined, processing should remain True
    assert gui.processing is True
    root.destroy()


def test_start_transcription_starts_thread(monkeypatch, tmp_path):
    from src import gui as gui_mod
    gui, root = make_gui(monkeypatch)

    audio = tmp_path / 'a.mp3'
    audio.write_bytes(b'RIFF')
    gui.audio_file = str(audio)
    gui.transcription_type.set('simple')

    started = {}

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self._target = target
            self.daemon = daemon

        def start(self):
            started['called'] = True

    monkeypatch.setattr('src.gui.threading.Thread', FakeThread)

    gui.start_transcription()
    assert gui.processing is True
    assert started.get('called') is True
    root.destroy()


def test_main_uses_tk(monkeypatch):
    # Replace tk.Tk with a fake that has mainloop
    from types import SimpleNamespace
    class FakeRoot(SimpleNamespace):
        def __init__(self):
            pass
        def title(self, *a, **k):
            return
        def geometry(self, *a, **k):
            return
        def resizable(self, *a, **k):
            return
        def columnconfigure(self, *a, **k):
            return
        def rowconfigure(self, *a, **k):
            return
        def mainloop(self):
            # do nothing
            return

    monkeypatch.setattr('src.gui.tk.Tk', FakeRoot)
    # Avoid building the whole UI inside WhisperGUI for this test
    monkeypatch.setattr('src.gui.WhisperGUI.setup_ui', lambda self: None)
    from src import gui as gui_mod
    gui_mod.main()
