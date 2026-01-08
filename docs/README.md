# Documentación del proyecto PoC_Whisper

Contenido breve de la documentación del repositorio.

Run & Development

- Activar entorno virtual: `source .venv/bin/activate`.
- Ejecutar tests: `pytest --cov=src --cov-report=term-missing`.
- Ejecutar BDD: `behave --no-capture --stop`.

CI considerations

- La suite de tests está diseñada para ejecutarse sin descargar modelos pesados en CI. Las pruebas reemplazan (mock) las llamadas a `whisper`, `pyannote.audio` y `torchaudio`.
- Asegúrate de establecer `HF_TOKEN` en el entorno o pasar como argumento para ejecutar la pipeline de diarización en producción.

Development notes

- Mantén `requirements.txt` y `docs/README.md` en sincronía cuando añadas dependencias.
- Lee `copilot-instructions.md` para pautas de contribución y pruebas.
# PoC Whisper

Proyecto Proof-of-Concept para transcripción de audio usando OpenAI Whisper y diarización con pyannote.audio.

Estructura propuesta:

- `src/` - Código fuente (`transcribe.py`, `diarize.py`, `gui.py`, ejemplos)
- `scripts/` - Scripts útiles (monitor, helpers)
- `outputs/` - Salidas generadas (transcripciones, logs)
- `docs/` - Documentación y README

Archivos que permanecen en la raíz: `.env`, `.env.example`, `.gitignore`, `requirements.txt`

Para ejecutar la GUI desde la raíz:

```bash
python -m src.gui
```

Para ejecutar transcripción simple:

```bash
python -m src.transcribe <archivo_audio> [modelo] [idioma]
```

Para ejecutar diarización:

```bash
python -m src.diarize <archivo_audio> [hf_token] [modelo] [idioma]
```
