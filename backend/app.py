"""MT2KD web backend (FastAPI).

Run from the webapp folder:   python -m uvicorn backend.app:app --port 8000
Then open http://localhost:8000

Folders (relative to the webapp folder, can be changed with environment variables):
  deploy/    MT2KD_DEPLOY    model.pt, labels.json, zipformer_src.zip, export_report.json (from Google Drive: MT2KD/deploy/<run>/)
  speakers/  MT2KD_SPEAKERS  optional registered speakers: speakers/<Name>/*.wav (3-5 clean clips of 5-15 s each)
  frontend/  MT2KD_FRONTEND  index.html
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .mt2kd_infer import MT2KD, SR, DeployFolderError, decode_audio, peaks

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = Path(os.environ.get("MT2KD_DEPLOY", ROOT / "deploy"))
SPEAKERS = Path(os.environ.get("MT2KD_SPEAKERS", ROOT / "speakers"))
FRONTEND = Path(os.environ.get("MT2KD_FRONTEND", ROOT / "frontend"))
MAX_UPLOAD_MB = float(os.environ.get("MT2KD_MAX_UPLOAD_MB", 50))
MAX_SECONDS = float(os.environ.get("MT2KD_MAX_SECONDS", 600))
MIN_SECONDS = 0.5
MAX_STORED = 40
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}

STATE = dict(model=None, error=None, speakers={}, speaker_errors=[])
STORE: "OrderedDict[str, dict]" = OrderedDict()          # audio_id -> {wav, name, info, cache}
STORE_LOCK = threading.Lock()


def load_speakers(model: MT2KD):
    """speakers/<Name>/*.wav -> one embedding per speaker (mean of the L2-normalised clip embeddings)."""
    out, errors = {}, []
    if not SPEAKERS.is_dir():
        return out, errors
    for d in sorted(p for p in SPEAKERS.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        clips = sorted(f for f in d.iterdir() if f.suffix.lower() in AUDIO_EXT)
        embs, secs = [], 0.0
        for f in clips:
            try:
                w, info = decode_audio(f.read_bytes(), f.name)
                if info["duration_s"] < 2.0:
                    errors.append(f"{d.name}/{f.name}: shorter than 2 s, skipped")
                    continue
                embs.append(model.embed(w))
                secs += info["duration_s"]
            except Exception as e:
                errors.append(f"{d.name}/{f.name}: {e}")
        if embs:
            e = np.mean(embs, axis=0)
            out[d.name] = dict(id=d.name, name=d.name.replace("_", " "), clips=len(embs), seconds=round(secs, 1), emb=e / np.linalg.norm(e))
    return out, errors


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        t0 = time.time()
        m = MT2KD(DEPLOY, threads=os.environ.get("MT2KD_THREADS"))
        STATE["model"] = m
        STATE["speakers"], STATE["speaker_errors"] = load_speakers(m)
        print(f"[MT2KD] model loaded from {DEPLOY} in {time.time() - t0:.1f} s | {m.params / 1e6:.1f}M parameters | "
              f"{len(STATE['speakers'])} registered speakers", flush=True)
        for e in STATE["speaker_errors"]:
            print(f"[MT2KD] speaker clip skipped: {e}", flush=True)
    except DeployFolderError as e:
        STATE["error"] = str(e)
        print(f"[MT2KD] MODEL NOT LOADED: {e}", flush=True)
    except Exception as e:                                      # keep serving the page (Demo mode still works)
        STATE["error"] = f"{type(e).__name__}: {e}"
        print(f"[MT2KD] MODEL NOT LOADED: {STATE['error']}", flush=True)
    yield


app = FastAPI(title="MT2KD Audio Intelligence API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


def model() -> MT2KD:
    if STATE["model"] is None:
        raise HTTPException(503, f"The model is not loaded: {STATE['error'] or 'still starting'}")
    return STATE["model"]


def get_audio(audio_id: str) -> dict:
    with STORE_LOCK:
        a = STORE.get(audio_id)
    if a is None:
        raise HTTPException(404, "This recording is no longer on the server (it restarted or too many files were uploaded). Upload it again.")
    return a


def cached(a: dict, key: str, fn):
    if key not in a["cache"]:
        t0 = time.time()
        r = fn()
        a["cache"][key] = (r, time.time() - t0)
    return a["cache"][key]


# ---------------------------------------------------------------------------------------------------- status
@app.get("/api/health")
def health():
    m = STATE["model"]
    return dict(status="ok" if m else "model_not_loaded", model_loaded=m is not None, error=STATE["error"],
                device="cpu", sv_threshold=(m.threshold if m else None), registered_speakers=len(STATE["speakers"]),
                lm_loaded=bool(m and m.lm_decoder), lm_error=(m.lm_error if m else None))


@app.get("/api/model-info")
def model_info():
    m = model()
    src = m.source or {}
    val = src.get("validation_at_this_epoch") or {}
    official = None
    if m.final_report:
        f = m.final_report.get("final", {})
        sv = next(iter((f.get("sv_official") or {}).values()), {}) if f.get("sv_official") else {}
        official = dict(asr_test_clean_wer=(f.get("asr_test_clean") or {}).get("wer"), asr_dev_clean_wer=(f.get("asr_dev_clean") or {}).get("wer"),
                        at_val_map=f.get("at_val_mAP_200"), at_eval_map=f.get("at_eval_mAP_200"),
                        sv_vox1o_eer=sv.get("eer"), sv_status=sv.get("status"))
    return dict(
        model="MT2KD student: Zipformer-M + task heads", parameters=m.params, stage2_epoch=src.get("stage2_epoch"),
        stage2_finished=src.get("stage2_finished"),
        teachers=dict(asr="WavLM-Large (layer 24)", at="SSLAM (AudioSet-2M fine-tuned)", sv="ReDimNet2-B6"),
        validation=dict(asr_dev_clean_wer=val.get("asr_dev_wer"), asr_dev_clean_cer=val.get("asr_dev_cer"),
                        at_fsd50k_val_map=val.get("at_val_mAP"), sv_val_eer_nonofficial=val.get("sv_val_eer")),
        official_test=official, sv_threshold=m.threshold,
        lm=(dict(alpha=m.lm_config["alpha"], beta=m.lm_config["beta"], beam=m.lm_config.get("beam"), lm=m.lm_config.get("lm"),
                 dev_clean_half_B=m.lm_config.get("half_B"), tuned_on=m.lm_config.get("tuned_on"), reported_on=m.lm_config.get("reported_on"))
            if m.lm_config else None),
        notes=["ASR: character CTC trained on 30 h of LibriSpeech (upper case, no punctuation); decoded greedily and, if deploy/lm exists, "
               "with beam search + the LibriSpeech 3-gram LM.",
               "AT: 200 FSD50K classes. SV: cosine similarity of 192-D embeddings."])


@app.get("/api/speakers")
def speakers():
    return [dict(id=s["id"], name=s["name"], clips=s["clips"], seconds=s["seconds"]) for s in STATE["speakers"].values()]


@app.post("/api/speakers/reload")
def speakers_reload():
    STATE["speakers"], STATE["speaker_errors"] = load_speakers(model())
    return dict(speakers=speakers(), skipped=STATE["speaker_errors"])


# ---------------------------------------------------------------------------------------------------- upload once
@app.post("/api/audio")
async def upload(file: UploadFile = File(...)):
    model()
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 2**20:
        raise HTTPException(413, f"The file is {len(data) / 2**20:.1f} MB. The limit is {MAX_UPLOAD_MB:.0f} MB.")
    try:
        wav, info = decode_audio(data, file.filename or "")
    except ValueError as e:
        raise HTTPException(415, str(e))
    if info["duration_s"] < MIN_SECONDS:
        raise HTTPException(422, f"The recording is {info['duration_s']:.2f} s long. It needs at least {MIN_SECONDS} s of audio.")
    if info["duration_s"] > MAX_SECONDS:
        raise HTTPException(422, f"The recording is {info['duration_s'] / 60:.1f} min long. The limit is {MAX_SECONDS / 60:.0f} min.")
    aid = uuid.uuid4().hex[:12]
    with STORE_LOCK:
        STORE[aid] = dict(wav=wav, name=file.filename, info=info, cache={})
        while len(STORE) > MAX_STORED:
            STORE.popitem(last=False)
    return dict(audio_id=aid, filename=file.filename, duration_s=round(info["duration_s"], 2), sample_rate_in=info["sample_rate_in"],
                channels=info["channels"], resampled_to=SR, peaks=peaks(wav))


# ---------------------------------------------------------------------------------------------------- tasks
class ASRReq(BaseModel):
    audio_id: str


class ATReq(BaseModel):
    audio_id: str
    top_k: int = Field(5, ge=1, le=50)


class SVReq(BaseModel):
    audio_id: str
    reference_audio_id: str | None = None
    speaker_id: str | None = None


@app.post("/api/asr")
def asr(req: ASRReq):
    m, a = model(), get_audio(req.audio_id)
    r, secs = cached(a, "asr", lambda: m.transcribe(a["wav"]))
    use_lm = r["text_lm"] is not None
    text = r["text_lm"] if use_lm else r["text_greedy"]
    note = ("Character CTC + LibriSpeech word language model (beam search). Upper case, no punctuation. "
            if use_lm else "Greedy CTC on characters, no language model. Upper case, no punctuation. ")
    note += "Research prototype trained on 30 h of read English (LibriSpeech): conversational or very expressive speech is harder."
    return dict(task="asr", text=text, text_greedy=r["text_greedy"], decoder="ctc+lm" if use_lm else "greedy",
                words=len(text.split()), segments=r["segments"], duration_s=round(a["info"]["duration_s"], 2), compute_s=round(secs, 2), note=note)


@app.post("/api/at")
def at(req: ATReq):
    m, a = model(), get_audio(req.audio_id)
    r, secs = cached(a, "at", lambda: m.tag(a["wav"], top_k=50))
    return dict(task="at", labels=r["labels"][:req.top_k], windows=r["windows"], duration_s=round(a["info"]["duration_s"], 2),
                compute_s=round(secs, 2), note="Scores are independent per class (several sounds can be present at once).")


@app.post("/api/sv")
def sv(req: SVReq):
    m, a = model(), get_audio(req.audio_id)
    if bool(req.reference_audio_id) == bool(req.speaker_id):
        raise HTTPException(422, "Give either reference_audio_id (mode 1) or speaker_id (mode 2).")
    t0 = time.time()
    e_test, _ = cached(a, "emb", lambda: m.embed(a["wav"]))
    if req.reference_audio_id:
        if req.reference_audio_id == req.audio_id:
            raise HTTPException(422, "The reference and the test recording are the same file. Upload a different reference recording.")
        ref = get_audio(req.reference_audio_id)
        e_ref, _ = cached(ref, "emb", lambda: m.embed(ref["wav"]))
        against = dict(mode="reference", name=ref["name"])
    else:
        s = STATE["speakers"].get(req.speaker_id)
        if s is None:
            raise HTTPException(404, f"Registered speaker {req.speaker_id!r} not found.")
        e_ref = s["emb"]
        against = dict(mode="registered", name=s["name"], clips=s["clips"])
    sim = float(np.dot(e_test, e_ref))
    thr = m.threshold or {}
    tv = thr.get("value")
    return dict(task="sv", similarity=round(sim, 4), threshold=tv, threshold_status=thr.get("status", "unknown"),
                same_speaker=(sim >= tv) if tv is not None else None, against=against, compute_s=round(time.time() - t0, 2),
                note=("Threshold is provisional (LibriSpeech pairs); it will be calibrated on VoxCeleb1 validation trials."
                      if thr.get("status") == "provisional" else "Threshold calibrated on VoxCeleb1 validation trials."))


# ---------------------------------------------------------------------------------------------------- the web page
@app.get("/")
def index():
    f = FRONTEND / "index.html"
    if not f.exists():
        raise HTTPException(404, f"{f} not found")
    return FileResponse(f, headers={"Cache-Control": "no-store"})
