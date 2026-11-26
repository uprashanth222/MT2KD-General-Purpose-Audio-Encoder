# SvaraMitra: one compact encoder for speech, sounds and speakers

SvaraMitra is our re-implementation of **MT2KD** (Yang, Li, Zhang and Woodland, IEEE/ACM TASLP 2025) on a single GPU with a reduced data budget. One shared Zipformer encoder does three jobs:

- **ASR**: speech to text (character CTC, optional word language model)
- **AT**: audio tagging, 200 FSD50K sound classes
- **SV**: speaker verification, 192-D speaker embeddings

Team PS-G-1586, Project School (NGIT / KMEC), Acoustics domain. Guide: P. V. N. Balaram Murthy, Department of CSE, KMEC.

## What we changed from the paper

1. **Smaller, stronger teachers.** WavLM-Large (ASR), SSLAM (AT) and ReDimNet2-B6 (SV) instead of Whisper-large-v3, BEATs and ECAPA-TDNN. Each swap was chosen by a controlled comparison on our own data (see `notebooks/01_teacher_selection`). Teacher parameters go from about 750M to about 418M.
2. **Zipformer-S student with a teaching assistant.** A 22M Zipformer-S learns from the three teachers and from our finished 64M Zipformer-M, which acts as a frozen teaching assistant. Three of its six encoder stacks (S1, S2, S6) start from Zipformer-M's weights.

We also tried gradient balancing (GradNorm + PCGrad, multi-layer and relational distillation). Both runs collapsed; they are kept in `notebooks/03_gradient_balancing` as a negative result.

## Results

Both models: 30 h LibriSpeech (train-clean-100 subset), 15,000 FSD50K clips, 30,000 VoxCeleb1 utterances; 30 Stage-1 epochs + 15 Stage-2 epochs. Official test sets unless marked otherwise.

| | Zipformer-M (64M) | Zipformer-S (22M) |
|---|---|---|
| LibriSpeech test-clean WER, greedy CTC | 45.08% | **43.36%** |
| LibriSpeech dev-clean WER, greedy CTC | 46.07% | **44.26%** |
| dev-clean WER with the 3-gram LM (held-out speakers) * | 20.42% | **19.34%** |
| FSD50K mAP (validation split, 200 classes) | 39.36% | **43.07%** |
| VoxCeleb1-O EER (cosine, 37,720 trials) | 8.44% | **4.30%** |
| Parameters (encoder + heads, deployed) | 64.35M | **22.44M** |
| CPU real-time factor (same 144.8 s of audio) | 0.036 | **0.025** |

\* LM decoding with `src/app/backend/ctc_lm.py`, alpha 0.7, beta 3.0, tuned on one half of the dev-clean speakers and reported on the other half (Zipformer-M: 1,446 utterances; Zipformer-S: 300 utterances).

The base paper reports 2.35% WER, 45.9 AudioSet mAP and 1.13% EER with 960 h of LibriSpeech, full AudioSet and VoxCeleb, 4 GPUs and 90 epochs, so these numbers are not directly comparable. Raw numbers for every row are in `results/`.

## Repository layout

```
data/
  raw/         dataset sources (audio is not stored in git)
  processed/   what preprocessing produces and where it is kept
  splits/      one CSV per dataset: filename, label, duration, split, source (+ leakage check)
src/
  mt2kd/       shared training modules (ASR / AT / SV) used by every notebook
  data/        make_splits.py, dataset_stats.py
  app/         web application: FastAPI backend, web front end, stress test
  tools/       audio chunking script for long recordings
notebooks/     the Google Colab notebooks we actually ran, in order (00 data → 06 export)
results/       final reports, run configs and histories, dataset statistics, figures
docs/          architecture diagrams, paper draft, project report, user guide, links
```

Each folder has its own README with details.

## Running the web app

See `src/app/README.md`. Short version on Windows: put the exported model into `src/app/deploy/` and double-click `run_windows.bat`; the app opens at http://localhost:8000. `start_public_link.bat` shares it over HTTPS (works on phones, microphone included) while the laptop is running.

The trained checkpoints are not in this repository (85 MB for Zipformer-S, 246 MB for Zipformer-M). They are on our Google Drive and available on request.

## Links

- Repository: https://github.com/uprashanth222/MT2KD-General-Purpose-Audio-Encoder
- Web app: runs locally and on the department cluster; a public demo link is shared through a Cloudflare tunnel during demos.

## Team

| Member | Role |
|---|---|
| B. Pavithra | Data lead |
| Nithyantha | Data lead |
| U. Prashanth | Model lead |
| P. Mohammad Mansur | Evaluation lead |
| R. Vanusshka | Paper and documentation |

## References

- X. Yang, Q. Li, C. Zhang, P. C. Woodland, "MT2KD: Toward a general-purpose encoder for speech, speaker, and audio events," IEEE Trans. Audio, Speech, Lang. Process., vol. 33, 2025.
- Z. Yao et al., "Zipformer: A faster and better encoder for automatic speech recognition," ICLR 2024.
- S. Chen et al., "WavLM: Large-scale self-supervised pre-training for full stack speech processing," IEEE JSTSP, 2022.
- T. Alex et al., "SSLAM: Enhancing self-supervised models with audio mixtures for polyphonic soundscapes," ICLR 2025.
- I. Yakovlev, A. Okhotnikov, "ReDimNet2," Interspeech 2026.
- S. I. Mirzadeh et al., "Improved knowledge distillation via teacher assistant," AAAI 2020.
