# SvaraMitra web app

A FastAPI server and a single web page. Upload or record a clip once, then run speech recognition, audio tagging and speaker verification on it. Everything runs on the CPU of the machine that starts the server.

```
app/
  run_windows.bat          start the app (the first run installs everything into .venv)
  start_public_link.bat    start the app + a temporary public HTTPS link (Cloudflare quick tunnel)
  backend/                 FastAPI server (app.py), model loading and inference (mt2kd_infer.py), LM decoder (ctc_lm.py)
  frontend/                index.html + assets/ (css, js, manifest)
  deploy/                  the exported model: small files are here, the large ones come from Google Drive
  speakers/                registered speakers for speaker verification (one folder per person)
```

## 1. Model files

The small files are already in `deploy/` (labels, LM settings, SV threshold). Copy the rest from Google Drive, folder `MT2KD/deploy/mt2kd_S20_main/` (written by `notebooks/06_export_and_figures/export_zipformer_s_for_app`):

```
deploy/model.pt                         Zipformer-S, 85 MB
deploy/zipformer_src.zip                Zipformer source used by the model
deploy/final_report.json                official test numbers shown on the page (optional)
deploy/lm/3-gram.pruned.1e-7.arpa.gz    LibriSpeech 3-gram LM, 33 MB (optional, for LM decoding)
```

The older 64M Zipformer-M export (`MT2KD/deploy/mt2kd_E0_main/`) uses the same format and loads with the same code.

## 2. Start

Windows: double-click `run_windows.bat`; the page opens at http://localhost:8000. The first run creates `.venv` and installs FastAPI, PyTorch (CPU) and the audio packages.

Any OS, from this folder:

```
python -m venv .venv
.venv\Scripts\activate          (macOS / Linux: source .venv/bin/activate)
pip install -r backend/requirements.txt
python -m uvicorn backend.app:app --port 8000
```

Start-up log should show `language model ready (alpha 0.7, beta 3.0, beam 64)` and `22.2M parameters`.

## 3. Share a public link (optional)

Download `cloudflared-windows-amd64.exe` from https://github.com/cloudflare/cloudflared/releases, rename it to `cloudflared.exe`, put it in this folder and run `start_public_link.bat`. It prints a `https://….trycloudflare.com` link that works on phones (including the microphone). The link changes on every start and works only while the laptop and both windows stay on.

## 4. Registered speakers (optional)

`speakers/<Name>/` with 3 to 5 clean clips of 5 to 15 s (WAV, FLAC, OGG or MP3). Each speaker is the average embedding of their clips. Restart the app after adding speakers.

## Notes

- ASR: character CTC trained on 30 h of read English audiobooks; upper case, no punctuation. Shows the LM transcript and the plain greedy transcript. Spontaneous speech, accents and laptop or phone microphones give more errors than the test numbers.
- AT: 200 FSD50K classes with independent scores; long audio is scored in 10 s windows (maximum per class).
- SV: cosine similarity of 192-D embeddings. Threshold 0.428 = equal-error-rate threshold on 1,770 LibriSpeech dev-clean pairs (provisional).
- Uploaded audio is kept in memory only (last 40 files) and is gone when the server stops; nothing is written to disk.
- Demo mode (simulated, clearly labelled) is used automatically when the backend is not running.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | /api/health | | model and LM loaded, threshold |
| GET | /api/model-info | | model name, validation / official numbers, LM settings |
| GET | /api/speakers | | registered speakers |
| POST | /api/speakers/reload | | re-reads `speakers/` |
| POST | /api/audio | multipart `file` | `audio_id`, duration, waveform peaks |
| POST | /api/asr | `{audio_id}` | transcript (LM and greedy) |
| POST | /api/at | `{audio_id, top_k}` | labels with scores |
| POST | /api/sv | `{audio_id, reference_audio_id}` or `{audio_id, speaker_id}` | similarity, threshold, decision |

Interactive API docs: http://localhost:8000/docs
