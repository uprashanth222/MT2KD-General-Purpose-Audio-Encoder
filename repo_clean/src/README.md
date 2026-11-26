# Source

- `mt2kd/` – the three task modules (ASR / WavLM, AT / SSLAM, SV / ReDimNet2) with data loading, teacher target caches, the Zipformer student, losses and evaluation. Every notebook embeds these files unchanged as its "Part 1", so the notebooks and this folder always match.
- `app/` – the web application (FastAPI backend + front end). See `app/README.md`.
- `data/make_splits.py` – rebuilds the exact file selections of the training runs and writes `data/splits/*.csv` + the leakage check.
- `data/dataset_stats.py` – split summary, clip-duration histograms, FSD50K class balance, split distribution and estimated SNR plots.
- `app/tests/stress_test.py` – 20+ consecutive inferences and edge cases (short, long, silent, wrong format, empty, oversized) against the running app.
- `tools/chunk_audio.py` – splits long recordings into clips at quiet points (used for the mentor's 10-12 minute recordings).
