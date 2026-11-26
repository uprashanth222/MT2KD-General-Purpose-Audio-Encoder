# Notebooks

Every experiment ran in Google Colab as one self-contained cell. Each notebook first writes the shared modules (`src/mt2kd/`) to disk, then runs its steps and prints `STEP n/N` lines. Everything is resumable: checkpoints and reports go to Google Drive (`MyDrive/MT2KD/results/<run>/`), so a disconnected runtime continues where it stopped.

| Folder | Notebook | What it does | Runtime |
|---|---|---|---|
| 00_data | make_splits_and_dataset_stats | writes data/splits/*.csv (filename, label, duration, split, source), the leakage check, dataset statistics plots and copies the run configs / histories into results/runs/ | CPU |
| 01_teacher_selection | asr_teacher_wavlm_vs_whisper | 2 h distillation from WavLM-Large (Whisper-large-v3 as the reference teacher), CTC probe WER on dev-clean | GPU |
| | asr_teacher_ctc_probe | CTC probe comparison of the trained students: WavLM-taught vs Whisper-taught vs random init | GPU |
| | at_teacher_sslam_vs_beats | SSLAM vs BEATs on FSD50K validation (192 classes mapped by AudioSet ID) | GPU |
| | (SV teacher test) | ReDimNet2-B6 vs ECAPA-TDNN; notebook to be added from Drive | GPU |
| 02_zipformer_m | stage1_distillation | data, teacher targets, Stage 1 multi-teacher KD (30 epochs), Zipformer-M 64M | L4 |
| | stage2_finetune_and_eval | Stage 2 multi-task fine-tuning (15 epochs) + official evaluation | L4 |
| 03_gradient_balancing | ps_stage1 / ps_stage2 | GradNorm + PCGrad + multi-layer / relational KD (collapsed, negative result) | GPU |
| 04_zipformer_s | stage1_rounds | Zipformer-S 22M, Stage 1 with the Zipformer-M teaching assistant and inherited stacks | L4 |
| | stage1_rounds_stage2_15_epochs | same cell with Stage 2 extended to 15 epochs (equal to Zipformer-M) | L4 |
| | stage2_rounds | Stage 2 + official evaluation + CPU speed comparison | L4 |
| 05_language_model | lm_decoding_s_vs_m | CTC beam search + LibriSpeech 3-gram for both models | CPU |
| | app_lm_test_clean | test-clean WER of the deployed app with its own LM decoder | CPU |
| 06_export_and_figures | export_zipformer_s_for_app | converts the Zipformer-S checkpoint into the app's format and checks it with the app's own code | CPU |
| | report_figures | figures for the project report (EDA, model summary, failure analysis, t-SNE) | T4 |
| tools | chunk_long_recordings | cuts 10-12 minute recordings into short clips at quiet points | CPU |

Order for a full reproduction: 00 → 02 → 04 → 05 → 06.
