"""MT2KD inference: loads the deploy folder written by MT2KD_EXPORT_CELL.py and runs ASR, audio tagging and speaker embeddings.

The model code below is copied unchanged from the training cells (Cell 1: MC.build_sv_student / MSV, MA.fbank, MA.greedy_decode)
and was verified in the export cell (strict weight loading, dev-clean WER equal to training, reload test).
"""
from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

SR = 16000
REQUIRED = ("model.pt", "labels.json", "zipformer_src.zip")


class DeployFolderError(RuntimeError):
    pass


def _check_deploy(d: Path):
    missing = [f for f in REQUIRED if not (d / f).exists()]
    if missing:
        raise DeployFolderError(
            f"deploy folder {d} is missing {missing}. Download MT2KD/deploy/<run>/ from Google Drive into this folder.")


def _prepare_zipformer(d: Path) -> Path:
    """Unpack the exact Zipformer source saved by the export cell (icefall files + Swoosh port)."""
    z = d / "zipformer_src.zip"
    out = d / "_zipformer"
    stamp = out / ".source_stamp"
    sig = f"{z.stat().st_size}-{int(z.stat().st_mtime)}"
    if not (stamp.exists() and stamp.read_text() == sig):
        import shutil
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
        stamp.write_text(sig)
    return out


class MT2KD:
    """One Zipformer encoder (M or S, sizes read from model.pt) + Stage-2 heads. Thread-safe (one inference at a time)."""

    def __init__(self, deploy_dir: str | Path, threads: int | None = None):
        import torch
        import torch.nn as nn

        self.torch = torch
        d = Path(deploy_dir).resolve()
        _check_deploy(d)
        zdir = _prepare_zipformer(d)
        if str(zdir) not in sys.path:
            sys.path.insert(0, str(zdir))
        for m in [k for k in list(sys.modules) if k in ("zipformer", "scaling", "subsampling", "encoder_interface", "k2_swoosh")
                  or k == "icefall" or k.startswith("icefall.")]:
            del sys.modules[m]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            from subsampling import Conv2dSubsampling
            from zipformer import Zipformer2
        if threads:
            torch.set_num_threads(int(threads))

        def lengths_to_mask(lens, T):
            return torch.arange(T, device=lens.device)[None, :] < lens[:, None]

        class _StopAfterBlock(Exception):
            pass

        class ZipformerStudent(nn.Module):
            def __init__(self, zip_args, sv_block=3):
                super().__init__()
                self.encoder_embed = Conv2dSubsampling(in_channels=80, out_channels=192)
                self.encoder = Zipformer2(**zip_args)
                self.sv_index = sv_block - 1

            def forward(self, feats, feat_lens):
                x, xl = self.encoder_embed(feats, feat_lens)
                enc, el = self.encoder(x.permute(1, 0, 2), xl, ~lengths_to_mask(xl, x.size(1)))
                return enc.permute(1, 0, 2), el

            def sv_features(self, feats, feat_lens):
                x, xl = self.encoder_embed(feats, feat_lens)
                cap = {}

                def hook(m, i, o):
                    cap["y"] = o
                    raise _StopAfterBlock()

                h = self.encoder.encoders[self.sv_index].register_forward_hook(hook)
                try:
                    self.encoder(x.permute(1, 0, 2), xl, ~lengths_to_mask(xl, x.size(1)))
                except _StopAfterBlock:
                    pass
                finally:
                    h.remove()
                return cap["y"].permute(1, 0, 2), xl

        class MSV(nn.Module):
            def __init__(self, C, A, D):
                super().__init__()
                self.att = nn.Sequential(nn.Conv1d(C, A, 1), nn.Tanh(), nn.Conv1d(A, C, 1))
                self.proj = nn.Linear(2 * C, D)

            def forward(self, y, lens):
                x = y.float().transpose(1, 2)
                m = lengths_to_mask(lens, x.size(2))[:, None, :]
                e = self.att(x).masked_fill(~m, float("-inf"))
                w = torch.softmax(e, dim=2)
                mu = (w * x).sum(2)
                var = (w * x * x).sum(2) - mu * mu
                return self.proj(torch.cat([mu, var.clamp(min=1e-5).sqrt()], 1))

        t0 = time.time()
        ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        if ck.get("format") != "mt2kd-infer-v1":
            raise DeployFolderError(f"model.pt has format {ck.get('format')!r}; expected 'mt2kd-infer-v1' (re-run the export cell)")
        self.vocab = list(ck["vocab"])
        self.student = ZipformerStudent(ck["zip_args"], ck.get("sv_block", 3))
        self.student.load_state_dict(ck["student"], strict=True)
        D_enc = int(ck["heads"]["ctc.weight"].shape[1])           # encoder output size: 512 (Zipformer-M) or 256 (Zipformer-S)
        self.heads = nn.ModuleDict(dict(ctc=nn.Linear(D_enc, len(self.vocab)), at=nn.Linear(D_enc, ck["at_classes"]), msv=MSV(**ck["msv"])))
        self.model_name = ck.get("model_name", "Zipformer-M")
        self.heads.load_state_dict(ck["heads"], strict=True)
        for name, mod in self.student.named_modules():
            if hasattr(mod, "batch_count"):
                mod.batch_count = ck["batch_count"]
            if hasattr(mod, "name"):
                mod.name = name
        self.student.eval()
        self.heads.eval()
        self._mask = lengths_to_mask
        self.lock = threading.Lock()

        self.labels = json.loads((d / "labels.json").read_text(encoding="utf-8"))
        self.report = json.loads((d / "export_report.json").read_text(encoding="utf-8")) if (d / "export_report.json").exists() else {}
        self.source = ck.get("source", {})
        self.threshold = dict(ck.get("sv_threshold") or {})
        cal = d / "sv_threshold.json"                    # written later by the calibration step (VoxCeleb1 validation trials)
        if cal.exists():
            self.threshold = json.loads(cal.read_text(encoding="utf-8"))
        self.final_report = None
        fr = d / "final_report.json"                     # optional: Cell 2's final_report.json (official test-set numbers)
        if fr.exists():
            self.final_report = json.loads(fr.read_text(encoding="utf-8"))
        self.params = sum(p.numel() for p in self.student.parameters()) + sum(p.numel() for p in self.heads.parameters())
        self.deploy_dir = d
        self.load_seconds = time.time() - t0
        self.lm_decoder, self.lm_config, self.lm_error = None, None, None
        self._load_lm(d / "lm")

    def _load_lm(self, lmdir: Path):
        """Optional word LM for CTC beam search (written by MT2KD_LM_DECODING_CELL.py into deploy/lm/)."""
        cfg = lmdir / "lm_config.json"
        if not cfg.exists():
            return
        try:
            from . import ctc_lm
        except ImportError:
            import ctc_lm
        try:
            c = json.loads(cfg.read_text(encoding="utf-8"))
            t0 = time.time()
            lm = ctc_lm.NgramLM.load_cached(lmdir / c["lm"], log=lambda *a: print("[MT2KD]", *a, flush=True))
            hw = lmdir / "hotwords.txt"                  # optional: one extra word per line (e.g. YOUTUBE)
            if hw.exists():
                words = [l.strip() for l in hw.read_text(encoding="utf-8").splitlines() if l.strip() and not l.strip().startswith("#")]
                lm.add_hotwords(words)
                print(f"[MT2KD] hotwords added: {len(words)}", flush=True)
            self.lm_decoder = ctc_lm.Decoder(self.vocab, lm, alpha=c["alpha"], beta=c["beta"], beam=c.get("beam", 64))
            self.lm_config = c
            print(f"[MT2KD] language model ready in {time.time() - t0:.0f} s (alpha {c['alpha']}, beta {c['beta']}, beam {c.get('beam', 64)})", flush=True)
        except Exception as e:
            self.lm_error = f"{type(e).__name__}: {e}"
            print(f"[MT2KD] language model NOT loaded ({self.lm_error}); greedy decoding only", flush=True)

    # ------------------------------------------------------------------ front end (identical to training)
    def _fbank(self, wav: np.ndarray):
        import torchaudio.compliance.kaldi as kaldi
        x = self.torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None]
        return kaldi.fbank(x, num_mel_bins=80, frame_length=25.0, frame_shift=10.0, dither=0.0, energy_floor=1e-10,
                           sample_frequency=float(SR), snip_edges=False, low_freq=20.0, high_freq=-400.0, remove_dc_offset=True,
                           preemphasis_coefficient=0.97, window_type="povey", use_energy=False)

    def _greedy(self, ids):
        out, prev = [], -1
        for k in ids:
            if k != prev and k != 0:
                out.append(self.vocab[k])
            prev = k
        return " ".join("".join(out).split())

    def _encode(self, wav):
        fb = self._fbank(wav)[None]
        fl = self.torch.tensor([fb.size(1)])
        enc, el = self.student(fb, fl)
        return enc, el, fb, fl

    # ------------------------------------------------------------------ tasks
    def transcribe(self, wav: np.ndarray, max_chunk_s: float = 30.0) -> dict:
        """Greedy CTC and (if available) CTC beam search + word LM. Audio longer than max_chunk_s is cut at quiet points
        (training utterances are <= ~35 s)."""
        segs = split_at_quiet_points(wav, max_chunk_s)
        greedy, logps = [], []
        with self.lock, self.torch.no_grad():
            for a, b in segs:
                enc, el, _, _ = self._encode(wav[a:b])
                lg = self.heads["ctc"](enc.repeat_interleave(2, dim=1))[0, :int(el[0]) * 2]
                greedy.append(self._greedy(lg.argmax(-1).tolist()))
                logps.append(lg.float().log_softmax(-1).numpy())
        out = dict(text_greedy=" ".join(t for t in greedy if t), segments=len(segs), text_lm=None, lm_seconds=None)
        if self.lm_decoder is not None:
            t0 = time.time()
            out["text_lm"] = " ".join(t for t in (self.lm_decoder.decode(lp) for lp in logps) if t)
            out["lm_seconds"] = time.time() - t0
        return out

    def tag(self, wav: np.ndarray, top_k: int = 5, window_s: float = 10.0) -> dict:
        """Clip-level sigmoid scores (200 FSD50K classes). Longer audio: 10 s windows (training crop), max over windows."""
        W = int(window_s * SR)
        if len(wav) <= W:
            starts = [0]
        else:
            hop = W // 2
            starts = list(range(0, len(wav) - W + 1, hop))
            if starts[-1] != len(wav) - W:
                starts.append(len(wav) - W)
        best = None
        with self.lock, self.torch.no_grad():
            for s in starts:
                enc, el, _, _ = self._encode(wav[s:s + W])
                m = self._mask(el, enc.size(1)).unsqueeze(-1).to(enc.dtype)
                pooled = (enc * m).sum(1) / m.sum(1).clamp(min=1)
                p = self.torch.sigmoid(self.heads["at"](pooled))[0].numpy()
                best = p if best is None else np.maximum(best, p)
        order = np.argsort(-best)[:max(1, int(top_k))]
        return dict(labels=[dict(index=int(i), name=self.labels[i]["name"], mid=self.labels[i]["mid"], score=float(best[i])) for i in order],
                    windows=len(starts))

    def embed(self, wav: np.ndarray, max_s: float = 90.0) -> np.ndarray:
        """192-D speaker embedding from encoder stack 3 + M_SV (L2-normalised). Very long audio: centre max_s seconds."""
        L = int(max_s * SR)
        if len(wav) > L:
            s = (len(wav) - L) // 2
            wav = wav[s:s + L]
        with self.lock, self.torch.no_grad():
            fb = self._fbank(wav)[None]
            fl = self.torch.tensor([fb.size(1)])
            y, yl = self.student.sv_features(fb, fl)
            e = self.heads["msv"](y, yl)[0].numpy().astype(np.float64)
        return e / max(np.linalg.norm(e), 1e-12)


# ---------------------------------------------------------------------- audio helpers
def split_at_quiet_points(wav: np.ndarray, max_s: float, search_s: float = 8.0):
    """[(start, end)] segments of <= max_s seconds, each cut at the quietest 25 ms frame in the last search_s seconds."""
    n, M = len(wav), int(max_s * SR)
    if n <= M:
        return [(0, n)]
    out, a, win = [], 0, int(0.025 * SR)
    while n - a > M:
        lo, hi = a + M - int(search_s * SR), a + M
        frames = (hi - lo) // win
        e = np.square(wav[lo:lo + frames * win]).reshape(frames, win).mean(1)
        cut = lo + int(np.argmin(e)) * win + win // 2
        out.append((a, cut))
        a = cut
    out.append((a, n))
    return out


def decode_audio(data: bytes, filename: str = "") -> tuple[np.ndarray, dict]:
    """Bytes of an audio file -> 16 kHz mono float32 + info. WAV/FLAC/OGG/MP3 via soundfile; others via torchaudio if it can."""
    import io
    info = {}
    try:
        import soundfile as sf
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e1:
        try:
            import torchaudio
            t, sr = torchaudio.load(io.BytesIO(data))
            x = t.numpy().T
        except Exception:
            raise ValueError(f"could not read {filename or 'the file'} as audio ({type(e1).__name__}). Use WAV, FLAC, OGG or MP3.")
    info["sample_rate_in"] = int(sr)
    info["channels"] = int(x.shape[1])
    w = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != SR:
        import torch
        import torchaudio
        w = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(w)), int(sr), SR).numpy()
    w = np.ascontiguousarray(w, dtype=np.float32)
    if not np.isfinite(w).all():
        raise ValueError("the audio contains invalid samples (NaN or infinity)")
    info["duration_s"] = len(w) / SR
    return w, info


def peaks(wav: np.ndarray, n: int = 160) -> list[float]:
    """Peak amplitude per block, scaled to 0..1, for drawing the waveform."""
    if len(wav) == 0:
        return []
    k = max(1, len(wav) // n)
    p = np.abs(wav[: k * n].reshape(-1, k)).max(axis=1) if len(wav) >= n else np.abs(wav)
    m = float(p.max()) or 1.0
    return [round(float(v) / m, 4) for v in p]
