# Raw data (not stored in git)

| Dataset | Source | Files we used |
|---|---|---|
| LibriSpeech | https://www.openslr.org/12 | train-clean-100.tar.gz, dev-clean.tar.gz, test-clean.tar.gz |
| LibriSpeech LM | https://www.openslr.org/11 | 3-gram.pruned.1e-7.arpa.gz (word LM for decoding) |
| FSD50K | https://zenodo.org/record/4060432 | FSD50K.dev_audio, FSD50K.ground_truth, FSD50K.metadata |
| VoxCeleb1 | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html | vox1_dev_wav, vox1_test_wav, veri_test2.txt |

On our Drive these are under `MyDrive/MT2KD/datasets/` and are read by the notebooks directly (no extraction to Drive).
