"""Dataset statistics for the three corpora, computed from the split CSVs in data/splits/.

Usage (from the repository root):
    python src/data/dataset_stats.py --splits data/splits --out results/dataset_stats

Split CSV schema (one row per file):
    filename, label, duration, split, source, [speaker]
    - ASR (LibriSpeech): label = transcript, speaker = reader id
    - AT  (FSD50K):      label = class names joined with ';'
    - SV  (VoxCeleb1):   label = speaker id

Writes: split_summary.csv, duration_histograms.png, class_balance_fsd50k.png, split_distribution.png
and, if a SNR table is given (--snr), snr_distribution.png.
SNR has no clean reference in these datasets, so it is estimated per clip from frame energies (see snr_estimate).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = {"librispeech": "LibriSpeech (ASR)", "fsd50k": "FSD50K (AT)", "voxceleb1": "VoxCeleb1 (SV)"}
CROP_S = {"fsd50k": 10.0, "voxceleb1": 20.0}          # training crops used by the models
COLORS = {"train": "#2a78d6", "validation": "#eb6834", "test": "#3a9a5b"}


def load_splits(folder: str | Path) -> dict[str, pd.DataFrame]:
    folder = Path(folder)
    return {k: pd.read_csv(folder / f"{k}.csv") for k in DATASETS if (folder / f"{k}.csv").exists()}


def split_summary(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in dfs.items():
        for split, g in df.groupby("split", sort=False):
            r = dict(dataset=name, split=split, files=len(g), hours=round(g.duration.sum() / 3600, 2),
                     mean_s=round(g.duration.mean(), 2), median_s=round(g.duration.median(), 2),
                     min_s=round(g.duration.min(), 2), max_s=round(g.duration.max(), 2))
            if "speaker" in g and g.speaker.notna().any():
                r["speakers"] = g.speaker.nunique()
            if name == "fsd50k":
                r["classes"] = len({c for l in g.label.dropna() for c in str(l).split(";")})
            rows.append(r)
    return pd.DataFrame(rows)


def snr_estimate(x: np.ndarray, frame: int = 400, hop: int = 160) -> float:
    """Energy-percentile SNR estimate in dB: noise = 10th percentile of frame power, signal + noise = 90th percentile."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    p = (x[idx] ** 2).mean(1) + 1e-12
    lo, hi = np.percentile(p, 10), np.percentile(p, 90)
    return float(10 * np.log10(max(hi - lo, 1e-12) / lo))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "savefig.dpi": 300, "savefig.bbox": "tight"})
    return plt


def plot_durations(dfs, out: Path):
    plt = _plt()
    fig, axes = plt.subplots(1, len(dfs), figsize=(4 * len(dfs), 3.2))
    for ax, (name, df) in zip(np.atleast_1d(axes), dfs.items()):
        d = df[df.split == "train"].duration
        ax.hist(d, bins=50, color=COLORS["train"], edgecolor="white", linewidth=0.4)
        if name in CROP_S:
            ax.axvline(CROP_S[name], color="#333333", ls="--", lw=1)
            ax.text(CROP_S[name], ax.get_ylim()[1] * 0.92, f" crop {CROP_S[name]:.0f} s", fontsize=8)
        ax.set_title(f"{DATASETS[name]}, training split")
        ax.set_xlabel("duration (s)"); ax.set_ylabel("files")
        ax.text(0.98, 0.70, f"n = {len(d):,}\nmean {d.mean():.1f} s\nmedian {d.median():.1f} s\nmax {d.max():.1f} s",
                transform=ax.transAxes, ha="right", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    fig.tight_layout(); fig.savefig(out / "duration_histograms.png"); plt.close(fig)


def plot_class_balance(df: pd.DataFrame, out: Path):
    plt = _plt()
    tr = df[df.split == "train"].label.dropna().str.split(";").explode().value_counts()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.bar(np.arange(len(tr)), tr.values, color=COLORS["train"], width=0.9); ax.set_yscale("log")
    ax.set_xlim(-1, len(tr)); ax.set_xlabel("FSD50K class (sorted by frequency)"); ax.set_ylabel("training clips (log)")
    ax.set_title(f"Class balance of the FSD50K training subset: {tr.iloc[0]:,} clips for the most frequent class, {tr.iloc[-1]} for the rarest")
    for k in list(range(4)) + list(range(len(tr) - 4, len(tr))):
        ax.annotate(tr.index[k].replace("_", " "), (k, tr.iloc[k]), xytext=(0, 4), textcoords="offset points", rotation=60, fontsize=7)
    fig.tight_layout(); fig.savefig(out / "class_balance_fsd50k.png"); plt.close(fig)
    tr.rename("train_clips").to_csv(out / "class_counts_fsd50k_train.csv")


def plot_split_distribution(summary: pd.DataFrame, out: Path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, col, lab in zip(axes, ("files", "hours"), ("files", "hours of audio")):
        ds = list(dict.fromkeys(summary.dataset)); x = np.arange(len(ds)); w = 0.27
        for k, sp in enumerate(("train", "validation", "test")):
            v = [summary[(summary.dataset == d) & (summary.split == sp)][col].sum() for d in ds]
            ax.bar(x + (k - 1) * w, v, w, label=sp, color=COLORS[sp])
        ax.set_xticks(x); ax.set_xticklabels([DATASETS[d] for d in ds]); ax.set_ylabel(lab); ax.set_yscale("log")
    axes[0].legend(frameon=False); axes[0].set_title("Split distribution (files)"); axes[1].set_title("Split distribution (hours)")
    fig.tight_layout(); fig.savefig(out / "split_distribution.png"); plt.close(fig)


def plot_snr(snr: pd.DataFrame, out: Path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.2)); bins = np.linspace(0, 80, 41)
    for (name, g), c in zip(snr.groupby("dataset", sort=False), ("#2a78d6", "#eb6834", "#3a9a5b")):
        ax.hist(g.snr_db, bins=bins, histtype="step", lw=1.6, color=c, label=f"{DATASETS.get(name, name)}: median {g.snr_db.median():.0f} dB")
    ax.set_xlabel("estimated SNR (dB)"); ax.set_ylabel("clips"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Estimated SNR (energy percentiles, no clean reference)")
    fig.tight_layout(); fig.savefig(out / "snr_distribution.png"); plt.close(fig)


def main(splits: str, out: str, snr_csv: str | None = None):
    out_p = Path(out); out_p.mkdir(parents=True, exist_ok=True)
    dfs = load_splits(splits)
    if not dfs:
        raise SystemExit(f"no split CSVs found in {splits}")
    summary = split_summary(dfs); summary.to_csv(out_p / "split_summary.csv", index=False)
    print(summary.to_string(index=False))
    plot_durations(dfs, out_p); plot_split_distribution(summary, out_p)
    if "fsd50k" in dfs:
        plot_class_balance(dfs["fsd50k"], out_p)
    if snr_csv and Path(snr_csv).exists():
        plot_snr(pd.read_csv(snr_csv), out_p)
    print(f"written to {out_p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="results/dataset_stats")
    ap.add_argument("--snr", default=None, help="optional CSV with columns dataset, filename, snr_db")
    a = ap.parse_args()
    main(a.splits, a.out, a.snr)
