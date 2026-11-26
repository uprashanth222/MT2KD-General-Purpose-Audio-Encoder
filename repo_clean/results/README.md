# Results

| Folder | Contents |
|---|---|
| `zipformer_m/` | 64M model (run `MT2KD_E2E_main_E0`): `final_report.json` (official test numbers), Stage 1 / Stage 2 `history` CSVs, export check, LM settings |
| `zipformer_s/` | 22M model (run `MT2KD_E2E_main_S20_TA_rounds_v2`): `final_report.json`, export check, SV threshold, LM settings, per-epoch metrics taken from the training log |
| `figures/` | training curves and final comparison plots (+ the scripts that drew them) |
| `pilot/` | logs of the first small end-to-end pilot run |
| `runs/` | configs, history CSVs and reports copied from each run folder on Drive (added by `notebooks/00_data`) |
| `dataset_stats/` | split summary, duration histograms, class balance, SNR estimates (added by `notebooks/00_data`) |

Key numbers (also in the top-level README):

| | Zipformer-M | Zipformer-S |
|---|---|---|
| test-clean WER (greedy) | 45.08% | 43.36% |
| FSD50K val mAP | 39.36% | 43.07% |
| VoxCeleb1-O EER | 8.44% | 4.30% |
| deployed parameters | 64.35M | 22.44M |

In `final_report.json`, `asr_test_clean.wer`, `at_val_mAP_200` and `sv_official["veri_test2.txt"].eer` are the numbers above (as fractions).

Larger run files (checkpoints, teacher caches, per-run logs) stay on Google Drive under `MyDrive/MT2KD/results/`.
