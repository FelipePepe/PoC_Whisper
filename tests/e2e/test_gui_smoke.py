import sys
import os

# Basic smoke test placeholder for GUI e2e
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def test_gui_smoke():
    # Very small smoke test: import the GUI module
    import importlib
    importlib.import_module('src.gui')
    assert True
