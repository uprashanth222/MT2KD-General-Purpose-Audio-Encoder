"""Writes data/splits/{librispeech,fsd50k,voxceleb1}.csv with exactly the files the training runs used,
and checks that no speaker of a test split appears in a training or validation split.

It rebuilds the selections with the same functions and seeds as the training notebooks:
  - LibriSpeech: 30 h subset of train-clean-100 (MA.prepare_manifests, nested 2 h / 10 h / 30 h subsets), dev-clean, test-clean
  - FSD50K: MB.subset_idx(n_train, 15000, 0) (= default_rng(42) permutation) of the official train split, the whole official validation split
  - VoxCeleb1: 30,000 training utterances (numpy default_rng(42) permutation), held-out validation speakers, official test partition
MA / MB / MC are the ASR / AT / SV modules in src/mt2kd/ (each notebook writes and imports them as Part 1).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SEED_VOX = 42
FSD_TRAIN_CLIPS = 15000
VOX_TRAIN_UTTS = 30000
LS_HOURS = 30.0


def librispeech(MA) -> pd.DataFrame:
    MA.CFG["subsets"] = {"2h": 2.0, "10h": 10.0, "30h": LS_HOURS}
    MA.prepare_manifests()
    parts = [(MA.load_manifest("30h"), "train", "LibriSpeech/train-clean-100"),
             (MA.load_manifest("dev"), "validation", "LibriSpeech/dev-clean"),
             (MA.inventory("test-clean"), "test", "LibriSpeech/test-clean")]
    out = []
    for df, split, src in parts:
        out.append(pd.DataFrame(dict(filename=df.relpath.values, label=df.text.values, duration=df.duration_s.round(3).values,
                                     split=split, source=src, speaker=df.speaker.astype(str).values)))
    return pd.concat(out, ignore_index=True)


def fsd50k(MB, threads: int = 32) -> pd.DataFrame:
    import soundfile as sf
    FSD = MB.load_fsd(); labels = list(FSD["vocab"].label)
    n = len(FSD["splits"]["train"]["fnames"])
    sel = {"train": sorted(MB.subset_idx(n, min(FSD_TRAIN_CLIPS, n), 0)), "val": list(range(len(FSD["splits"]["val"]["fnames"])))}
    out = []
    for key, split in (("train", "train"), ("val", "validation")):
        S = FSD["splits"][key]; idx = sel[key]
        with ThreadPoolExecutor(threads) as ex:                       # original durations, read from the WAV headers
            dur = list(ex.map(lambda i: float(sf.info(S["wav"][i]).duration), idx))
        lab = [";".join(labels[j] for j in np.where(S["y"][i])[0]) for i in idx]
        out.append(pd.DataFrame(dict(filename=[f"{S['fnames'][i]}.wav" for i in idx], label=lab, duration=np.round(dur, 3),
                                     split=split, source="FSD50K/dev_audio", speaker=None)))
    return pd.concat(out, ignore_index=True)


def voxceleb1(MC) -> pd.DataFrame:
    MC.ensure_dirs(MC.paths()); MC.prepare_dataset("full")
    TR = MC.usable(MC.load_manifest("train_manifest")); DV = MC.usable(MC.load_manifest("validation_manifest"))
    TE = MC.usable(MC.load_manifest("test_manifest"))
    if len(TR) > VOX_TRAIN_UTTS:
        TR = TR.iloc[np.sort(np.random.default_rng(SEED_VOX).permutation(len(TR))[:VOX_TRAIN_UTTS])].reset_index(drop=True)
    out = []
    for df, split, src in ((TR, "train", "VoxCeleb1/dev"), (DV, "validation", "VoxCeleb1/dev (held-out speakers)"), (TE, "test", "VoxCeleb1/test")):
        out.append(pd.DataFrame(dict(filename=df.uid.values, label=df.speaker.astype(str).values,
                                     duration=(df.n_frames / df.sr).round(3).values, split=split, source=src,
                                     speaker=df.speaker.astype(str).values)))
    return pd.concat(out, ignore_index=True)


def leakage_report(dfs: dict[str, pd.DataFrame]) -> dict:
    """Speaker overlap between splits (must be 0 against the test split) and duplicate files across splits."""
    rep = {}
    for name, df in dfs.items():
        r = {}
        files = df.groupby("split").filename.apply(set)
        for a in files.index:
            for b in files.index:
                if a < b:
                    r[f"files {a} & {b}"] = len(files[a] & files[b])
        if df.speaker.notna().any():
            spk = df.groupby("split").speaker.apply(set)
            for a in spk.index:
                for b in spk.index:
                    if a < b:
                        r[f"speakers {a} & {b}"] = len(spk[a] & spk[b])
        rep[name] = r
    return rep


def write(dfs: dict[str, pd.DataFrame], out: str | Path) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for name, df in dfs.items():
        df.to_csv(out / f"{name}.csv", index=False)
    rep = leakage_report(dfs)
    json.dump(rep, open(out / "leakage_check.json", "w"), indent=1)
    return rep
