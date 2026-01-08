#!/usr/bin/env python3
"""
Wrapper ligero para lanzar la interfaz gráfica (frontend).
Llama al módulo `src.gui` para mantener compatibilidad con ejecuciones previas.
"""
import runpy
import sys


if __name__ == "__main__":
    # Reenvía argumentos y ejecuta el módulo de la GUI
    sys.argv[0] = 'python -m src.gui'
    runpy.run_module('src.gui', run_name='__main__')
