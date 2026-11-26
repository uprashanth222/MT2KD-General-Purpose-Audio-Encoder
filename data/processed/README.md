# Processed data (not stored in git)

Produced by the notebooks on the Colab local disk and cached on Drive (`MyDrive/MT2KD/cache16k/` and each run folder):

- 16 kHz mono audio: FSD50K clips as 10 s centre crops, VoxCeleb1 utterances as WAV, LibriSpeech read from the original FLAC.
- Teacher targets: WavLM-Large layer-24 features (2048-D at 25 Hz), SSLAM logits for 527 AudioSet classes, ReDimNet2-B6 192-D speaker embeddings.
- Features are computed on the fly: 80-bin log-Mel filter bank, 25 ms window, 10 ms hop, no augmentation.
