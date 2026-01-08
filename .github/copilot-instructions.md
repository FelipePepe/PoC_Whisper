# Copilot instructions for this repository

Breve: este repositorio es un PoC para transcribir audio con OpenAI Whisper y realizar
diarización de hablantes con `pyannote.audio`. El entorno de trabajo está en `.venv`.

Quick facts
- Python: 3.13 (virtualenv en `.venv`)
- Dependencias: ver `requirements.txt`
- Código fuente: `src/` (módulos principales: `transcribe.py`, `diarize.py`, `gui.py`, ejemplos)
- Scripts: `scripts/` (monitor), wrappers en raíz (`gui.py`, `run_gui`)
- Salidas: `outputs/`
- Tokens: HuggingFace token en `.env` como `HF_TOKEN` (no commitear)

When editing or adding code
- Keep changes minimal and focused; prefer small, safe edits.
- Use package-relative imports inside `src` (e.g. `from .transcribe import ...`).
- Do not modify `.venv` files. Do not add system-specific paths.
- Avoid large refactors unless asked; prefer adding wrappers or small helpers.
- Add tests: include unit, integration and/or end-to-end tests for new functionality. Place tests under `tests/` and group by type (e.g. `tests/unit`, `tests/integration`, `tests/e2e`).

How to run (developer machine)
```bash
# activar venv
source .venv/bin/activate

# GUI (wrapper preserves legacy entrypoint)
python gui.py
# o
./run_gui

# Módulos (modo paquete)
python -m src.transcribe <audio_file> [modelo] [idioma]
python -m src.diarize <audio_file> <HF_TOKEN> [modelo] [idioma]
```

Notes & cautions
- Diarización es intensiva: preferir GPU para rendimiento razonable.
- Keep `requirements.txt` and `docs/README.md` in sync when updating deps.
- Preserve user `.env` (do not commit secrets). Use `HF_TOKEN` in environment only.

If you need to change project structure
- Propose changes first and update `docs/README.md` with new run instructions.

CI and automated checks
- This repository includes a GitHub Actions workflow at `.github/workflows/ci.yml` that runs the test suites and enforces the per-suite coverage thresholds below. Push your branch to trigger CI.
- CI commands (same commands run in Actions):

```bash
# Unit suite (coverage enforced for `src/transcribe.py`)
pytest tests/unit --cov=src/transcribe.py --cov-fail-under=85 -q

# Integration suite (coverage enforced for `src/diarize.py`)
pytest tests/integration --cov=src/diarize.py --cov-fail-under=60 -q

# E2E suite (coverage enforced for `src/gui.py`)
pytest tests/e2e --cov=src/gui.py --cov-fail-under=10 -q
```

Testing & Coverage
- Add tests when adding or changing behaviour. Prefer unit tests for pure logic in `src/transcribe.py`, integration tests (mock heavy models when appropriate) for `src/diarize.py`, and small smoke e2e tests for `src/gui.py`.
- Coverage targets (per component / suite):
  - Unit tests: 85% 
  - Integration tests: 60% 
  - End-to-end (e2e) tests: 10% 
- Run a local, browsable coverage report when iterating:

```bash
pytest --cov=src --cov-report=term-missing --cov-report=html

Changelog
- Mantener un archivo `CHANGELOG.md` en la raíz del repositorio siguiendo la especificación "Keep a Changelog": https://keepachangelog.com/es-ES/1.0.0/.
- Use las secciones y convenciones recomendadas (por ejemplo: `Unreleased`, `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`).
- Actualice `CHANGELOG.md` con cada versión significativa o PR que cambie el comportamiento público del proyecto. Documente cambios agrupados por categoría y fecha.
- Ejemplo de inicio:

```markdown
## [Unreleased]
- Added: Nueva funcionalidad X.
```


```

Tips for testing heavy/machine-learning components
- Do not call large external models in unit tests. Use mocking or small, deterministic fixtures.
- For integration tests that exercise `pyannote.audio` pipelines, prefer recording short example audio and/or mocking the remote model responses. Keep heavy runs out of CI unless they run on GPU-enabled runners.
- Keep e2e GUI tests minimal and stable: prefer import/smoke checks or headless runs that avoid interactive dialogs.

Developer notes
- Use package-relative imports inside `src` so modules work with `python -m src.<module>`.
- If you add a new top-level script or wrapper, add a matching entry in `docs/README.md`.
- When adding new secrets (e.g. `HF_TOKEN`), document them in `docs/README.md` only — never commit real secrets.

