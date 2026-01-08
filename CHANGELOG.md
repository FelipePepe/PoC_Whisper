## [Unreleased]

- Added: Proof-of-concept transcription pipeline using OpenAI Whisper (`src/transcribe.py`).
- Added: Speaker diarization support and helpers (`src/diarize.py`).
- Added: Lightweight Tkinter GUI for local transcription and diarization (`src/gui.py`).
- Added: Example runner scripts (`src/example.py`, `src/example_diarization.py`).
- Added: Extensive test suites (unit, integration, e2e) with mocks for heavy ML deps.
- Added: Generated Gherkin scenarios based on pytest tests (`features/generated_from_pytest.feature`) and behave step definitions.
- Added: CI-friendly patterns: deferred heavy imports and fixtures to avoid downloading models during tests.

### Notes
- Tests and BDD runs use mocking for heavy dependencies (`whisper`, `pyannote.audio`, `torchaudio`) so CI can run without GPUs or large model downloads.
- Coverage: local per-file coverage targets reached (>=95%). See `pytest --cov=src` output for the exact numbers.
# Changelog

All notable changes to this project will be documented in this file.

The format is based on "Keep a Changelog" (https://keepachangelog.com/es-ES/1.0.0/).

## [Unreleased]
- Added: Inicialización del repositorio y PoC para transcripción y diarización.

## [0.1.0] - 2026-01-08
- Added: Primera versión pública con `src/transcribe.py`, `src/diarize.py`, `src/gui.py`.
- Added: Tests iniciales en `tests/unit`, `tests/integration`, `tests/e2e`.
- Added: CI workflow que ejecuta suites y verifica coverage por componente.
