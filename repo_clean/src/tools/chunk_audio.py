"""
chunk_audio.py - split long recordings into short clips for the MT2KD web app.

Cuts at quiet points (not in the middle of a word), converts to 16 kHz mono WAV.

Usage (Windows, from the mt2kd_webapp folder, with the backend's .venv active):
    python tools\\chunk_audio.py "D:\\sanada\\voice.wav"
    python tools\\chunk_audio.py "D:\\sanada"                      (all audio files in a folder)
    python tools\\chunk_audio.py "D:\\sanada\\voice.wav" --seconds 10   (shorter clips)
    python tools\\chunk_audio.py "D:\\sanada\\voice.wav" --speaker Sanada
        -> also copies 5 good clips (10 s each) into speakers\\Sanada\\ for SV "Registered speaker"

Output: <input folder>\\chunks\\<file name>\\<file name>_000.wav, _001.wav, ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SR = 16000
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


def load_audio(path):
    """Load any audio file as 16 kHz mono float32."""
    try:
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:
        try:
            import librosa
            x, sr = librosa.load(str(path), sr=None, mono=True)
        except Exception as e:
            raise RuntimeError(
                f"Cannot read {path.name}: {e}\n"
                "  Tip: convert it to WAV first (e.g. in Audacity), or install ffmpeg.")
    if sr != SR:
        try:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        except Exception:
            # simple fallback resampler
            n = int(round(len(x) * SR / sr))
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x.astype(np.float32)


def frame_energy_db(x, frame=400, hop=160):
    """Energy (dB) every 10 ms."""
    n = max(1, 1 + (len(x) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    idx = np.minimum(idx, len(x) - 1)
    e = np.mean(x[idx] ** 2, axis=1)
    return 10 * np.log10(e + 1e-10)


def find_cuts(x, target_s, min_s):
    """Cut points near every target_s seconds, moved to the quietest spot within +-30%."""
    db = frame_energy_db(x)
    hop = 160
    total = len(x)
    target, min_len = int(target_s * SR), int(min_s * SR)
    window = int(0.3 * target)
    cuts, start = [0], 0
    while total - start > target + min_len:
        lo = start + target - window
        hi = min(start + target + window, total - min_len)
        f_lo, f_hi = lo // hop, max(lo // hop + 1, hi // hop)
        seg = db[f_lo:f_hi]
        # smooth so we pick a pause, not a single quiet frame
        if len(seg) > 20:
            seg = np.convolve(seg, np.ones(20) / 20, mode="same")
        # among quiet frames (within 3 dB of the quietest), take the one closest to the target
        quiet = np.where(seg <= seg.min() + 3.0)[0]
        want = (start + target) // hop - f_lo
        cut = (f_lo + int(quiet[np.argmin(np.abs(quiet - want))])) * hop
        cuts.append(cut)
        start = cut
    cuts.append(total)
    return cuts


def is_mostly_silent(clip, thresh_db=-45.0):
    db = frame_energy_db(clip)
    return np.mean(db > thresh_db) < 0.2   # less than 20% of frames have sound


def write_wav(path, x):
    import soundfile as sf
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 1.0:
        x = x / peak
    sf.write(str(path), x, SR, subtype="PCM_16")


def chunk_file(path, out_root, seconds, min_s, keep_silent):
    x = load_audio(path)
    dur = len(x) / SR
    out_dir = out_root / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cuts = find_cuts(x, seconds, min_s)
    written, skipped = [], 0
    for i in range(len(cuts) - 1):
        clip = x[cuts[i]:cuts[i + 1]]
        if len(clip) < min_s * SR:
            skipped += 1
            continue
        if not keep_silent and is_mostly_silent(clip):
            skipped += 1
            continue
        p = out_dir / f"{path.stem}_{len(written):03d}.wav"
        write_wav(p, clip)
        written.append((p, clip))
    print(f"  {path.name}: {dur/60:.1f} min -> {len(written)} clips "
          f"(~{seconds:g} s each), {skipped} silent/short skipped -> {out_dir}")
    return written


def register_speaker(clips, name, speakers_dir, n=5, clip_s=10.0):
    """Pick the n loudest (most speech) clips, trim to clip_s seconds, save for SV."""
    d = speakers_dir / name
    d.mkdir(parents=True, exist_ok=True)
    scored = sorted(clips, key=lambda pc: -np.mean(frame_energy_db(pc[1]) > -40))
    k = 0
    for p, clip in scored:
        if len(clip) < 5 * SR:
            continue
        c = clip[: int(clip_s * SR)]
        write_wav(d / f"{name}_{k+1}.wav", c)
        k += 1
        if k == n:
            break
    print(f"  Registered speaker '{name}': {k} clips -> {d}")
    if k < 3:
        print("  WARNING: fewer than 3 clips; SV registration works best with 3-5.")
    print("  Restart the backend (or POST /api/speakers/reload) to load the new speaker.")


def main():
    ap = argparse.ArgumentParser(description="Split long audio into short clips.")
    ap.add_argument("input", help="audio file or folder")
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="target clip length in seconds (default 30)")
    ap.add_argument("--min", type=float, default=3.0,
                    help="drop clips shorter than this (default 3 s)")
    ap.add_argument("--out", default=None, help="output folder (default <input>/chunks)")
    ap.add_argument("--keep-silent", action="store_true", help="keep silent clips")
    ap.add_argument("--speaker", default=None,
                    help="also register this speaker name for SV (speakers/<name>/)")
    ap.add_argument("--speakers-dir", default=None,
                    help="speakers folder (default: ../speakers next to this script)")
    a = ap.parse_args()

    inp = Path(a.input).expanduser()
    if not inp.exists():
        sys.exit(f"Not found: {inp}")
    files = [inp] if inp.is_file() else sorted(
        p for p in inp.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not files:
        sys.exit(f"No audio files found in {inp}")
    out_root = Path(a.out) if a.out else (inp.parent if inp.is_file() else inp) / "chunks"
    print(f"Chunking {len(files)} file(s), target {a.seconds:g} s -> {out_root}")

    all_clips = []
    for f in files:
        try:
            all_clips += chunk_file(f, out_root, a.seconds, a.min, a.keep_silent)
        except Exception as e:
            print(f"  SKIPPED {f.name}: {e}")

    if a.speaker:
        sd = Path(a.speakers_dir) if a.speakers_dir else Path(__file__).resolve().parent.parent / "speakers"
        register_speaker(all_clips, a.speaker, sd)
    print(f"Done: {len(all_clips)} clips.")


if __name__ == "__main__":
    main()
