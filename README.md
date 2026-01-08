# PoC Whisper — Transcription & Diarization

Proyecto PoC para transcribir audio usando OpenAI Whisper y realizar diarización de hablantes con `pyannote.audio`. Incluye una interfaz Tkinter, ejemplos y una suite de tests diseñada para ejecutarse sin descargar modelos grandes en CI.

Quick start

1. Crear y activar el entorno virtual (`.venv`):

```bash
source .venv/bin/activate
```

2. Ejecutar tests y ver cobertura:

```bash
pytest --cov=src --cov-report=term-missing
```

3. Ejecutar GUI de ejemplo:

```bash
python gui.py
```

Notas importantes

- Colocar variables sensibles (por ejemplo `HF_TOKEN`) en un archivo `.env` o en el entorno. No commitear tokens.
- Tests y BDD usan mocks para `whisper`, `pyannote.audio`, y `torchaudio`, de forma que CI puede ejecutar sin GPU ni descargas.

Estructura relevante

- `src/` — Código fuente principal (`transcribe.py`, `diarize.py`, `gui.py`).
- `tests/` — Pruebas unitarias, de integración y e2e.
- `features/` — Escenarios Gherkin y step definitions para `behave`.
- `CHANGELOG.md` — Historial de cambios (Unreleased).
