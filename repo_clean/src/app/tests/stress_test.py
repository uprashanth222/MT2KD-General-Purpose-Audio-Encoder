"""Stress test and edge cases for the running web app (report section 5.4).

Start the app first (run_windows.bat), then from src/app/:
    .venv\\Scripts\\python tests\\stress_test.py --audio path\\to\\speech.wav [--ref path\\to\\other_speaker.wav] [--n 25]

1. 20+ consecutive inferences: upload the clip each time, then ASR, AT and SV (vs the reference clip).
2. Edge cases: audio shorter than 0.5 s, audio longer than 30 s, a non-audio file renamed to .wav, an empty file,
   pure silence, and a file over the 50 MB limit.
Writes tests/stress_results.csv and tests/edge_cases.csv and prints both tables.
Uses only the packages the app already needs (numpy, soundfile) plus the standard library.
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def _request(url, data=None, headers=None, timeout=300):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}"), time.time() - t0
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read() or b"{}")
        except Exception:
            body = {}
        return e.code, body, time.time() - t0


def upload(base, name, payload: bytes):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode() \
        + payload + f"\r\n--{b}--\r\n".encode()
    return _request(f"{base}/api/audio", body, {"Content-Type": f"multipart/form-data; boundary={b}"})


def post(base, path, obj):
    return _request(f"{base}{path}", json.dumps(obj).encode(), {"Content-Type": "application/json"})


def wav_bytes(x: np.ndarray) -> bytes:
    buf = io.BytesIO(); sf.write(buf, x.astype(np.float32), SR, format="WAV", subtype="PCM_16"); return buf.getvalue()


def load(path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True); x = x.mean(1)
    if sr != SR:                                     # simple resample for test signals only
        x = np.interp(np.arange(int(len(x) * SR / sr)) / SR, np.arange(len(x)) / sr, x).astype(np.float32)
    return x


def table(rows, cols):
    w = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--audio", required=True, help="a speech clip (WAV/FLAC/MP3)")
    ap.add_argument("--ref", default=None, help="a second clip for SV (default: the first half of --audio)")
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()
    out = Path(__file__).resolve().parent

    st, h, _ = _request(f"{a.base}/api/health")
    print(f"health: {st} {h}")
    x = load(a.audio); ref = load(a.ref) if a.ref else x[: len(x) // 2]
    s, r, _ = upload(a.base, "reference.wav", wav_bytes(ref)); ref_id = r.get("audio_id")

    rows = []
    for i in range(a.n):
        row = dict(run=i + 1)
        s, r, t = upload(a.base, Path(a.audio).name, Path(a.audio).read_bytes()); row.update(upload_status=s, upload_s=round(t, 2))
        aid = r.get("audio_id")
        for task, body in (("asr", {"audio_id": aid}), ("at", {"audio_id": aid, "top_k": 5}), ("sv", {"audio_id": aid, "reference_audio_id": ref_id})):
            s, r, t = post(a.base, f"/api/{task}", body)
            row[f"{task}_status"] = s; row[f"{task}_s"] = round(t, 2)
            if task == "asr": row["words"] = r.get("words")
            if task == "sv": row["similarity"] = r.get("similarity")
        rows.append(row); print(row)
    import csv
    with open(out / "stress_results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    ok = sum(all(v == 200 for k, v in r.items() if k.endswith("_status")) for r in rows)
    lat = np.array([[r["asr_s"], r["at_s"], r["sv_s"]] for r in rows])
    print(f"\n{ok}/{len(rows)} runs fully successful | mean latency ASR {lat[:, 0].mean():.2f} s, AT {lat[:, 1].mean():.2f} s, SV {lat[:, 2].mean():.2f} s "
          f"(audio {len(x) / SR:.1f} s; repeated uploads are new files, so nothing is served from the cache)")

    cases = [("audio 0.3 s (< 0.5 s)", "short.wav", wav_bytes(x[: int(0.3 * SR)])),
             ("audio 45 s (> 30 s)", "long.wav", wav_bytes(np.tile(x, int(np.ceil(45 * SR / len(x))))[: 45 * SR])),
             ("silence 5 s", "silence.wav", wav_bytes(np.zeros(5 * SR))),
             ("wrong format (text as .wav)", "fake.wav", b"this is not audio\n" * 100),
             ("empty file", "empty.wav", b""),
             ("file > 50 MB", "huge.wav", b"\0" * (51 * 2**20))]
    edge = []
    for name, fn, payload in cases:
        s, r, t = upload(a.base, fn, payload); e = dict(case=name, upload_status=s, message=(r.get("detail") or "")[:90])
        if s == 200:
            s2, r2, t2 = post(a.base, "/api/asr", {"audio_id": r["audio_id"]})
            e.update(asr_status=s2, asr_s=round(t2, 2), result=(r2.get("text") or r2.get("detail") or "")[:60])
        edge.append(e)
    with open(out / "edge_cases.csv", "w", newline="") as f:
        cols = ["case", "upload_status", "message", "asr_status", "asr_s", "result"]
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(edge)
    print(); table(edge, ["case", "upload_status", "message", "asr_status", "asr_s", "result"])
    print(f"\nwritten: {out / 'stress_results.csv'}, {out / 'edge_cases.csv'}")


if __name__ == "__main__":
    main()
