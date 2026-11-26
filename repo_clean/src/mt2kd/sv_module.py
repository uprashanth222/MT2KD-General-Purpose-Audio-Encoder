# ====================================================================================================================
# MT2KD STAGE 1 -- SPEAKER-VERIFICATION KNOWLEDGE DISTILLATION: ReDimNet2-B6 (frozen teacher) -> Zipformer-M (third encoder block + M_SV)
# SINGLE COLAB CELL. Choose MODE at the top of the RUN part (end of the cell): "full" (all official VoxCeleb1-dev training utterances) or "sanity".
#
# Loss (MT2KD Eq. 8):  L_SV_KD = 1 - CosineSim(v_teacher, M_SV(R_student^(3)))      -- no speaker labels in the loss
#   v_teacher : frozen ReDimNet2-B6 (PalabraAI/redimnet2, VoxCeleb2-dev, large-margin fine-tuned; 192-D speaker embedding), code pinned to one
#               commit, weights sha256-checked. Optional label-free target normalisation (none / mean-centred / standardised with TRAINING-set
#               statistics), chosen by an explicit experiment on held-out DEV speakers (never on the official test set).
#   R_student : output of Zipformer-M's THIRD encoder stack (encoder.encoders[2]: 3 layers, 384-D, run at 12.5 Hz inside, returned at ~50 Hz)
#   M_SV      : attentive pooling with two 1-D convolutions (paper) -> attentive mean + std -> Linear to the teacher dimension (192)
#
# STORAGE: everything large and temporary in /content (extracted + copied audio, ReDimNet2 code and weights, logs, local checkpoints).
#          Google Drive: /content/drive/MyDrive/MT2KD/results/SV_Stage1_ReDimNet2/ only (manifests, reports, the small utterance-level teacher
#          embedding cache, checkpoints). The dataset folder on Drive is only READ.
# RESUME:  re-run the cell after ANY interruption (disconnect, runtime recycled, Colab limit): finished steps are skipped, the teacher cache
#          resumes block by block (restored from Drive), training resumes from the most advanced valid checkpoint (local disk or Drive) at the
#          exact epoch / batch / optimizer step. Drive checkpoints: every drive_ckpt_every_minutes, every drive_ckpt_every_updates, every epoch.
# ====================================================================================================================
import os
os.environ.setdefault("TORCH_HOME", "/content/torch_cache")    # ReDimNet2 weights on LOCAL disk
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys, re, json, math, time, shutil, hashlib, subprocess, importlib, importlib.util, threading, random, itertools, zipfile, struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

CODE_VERSION = "sv1-redimnet2-v1"

CFG = dict(
    # ---- data (READ ONLY on Drive) and outputs
    vox_root="/content/drive/MyDrive/MT2KD/datasets/VoxCeleb1",
    drive_out="/content/drive/MyDrive/MT2KD/results/SV_Stage1_ReDimNet2",
    drive_api=True,                                   # Google sign-in once per runtime: confirms uploads (md5) + deletes this folder's replaced checkpoints from the trash
    write_root="/content/MT2KD_SV_ReDimNet2_work", local_audio="/content/vox1_local", zipformer_dir="/content/MT2KD_zipformer_sv1",
    extract_dir="/content/vox1_extracted",           # tar / tar.gz archives (also split into .part-NNN pieces) are extracted HERE (local disk)
    seed=42, sample_rate=16000,
    # ---- partitions
    n_val_speakers=40,                                # NON-OFFICIAL validation: 40 VoxCeleb1-dev speakers held out (VoxCeleb1 has no official SV validation split)
    sanity_train_utts=2000,
    sanity_dev_utts=8000,                             # sanity run: only this many dev utterances (+ the official test utterances) are taken from the archive
    n_val_speakers_subset=10,                         # held-out validation speakers when only a subset of VoxCeleb1-dev is available
    crop_seconds=20.0,                                # training / validation KD: teacher AND student see the same centred <= 20 s segment
    # ---- teacher: ReDimNet2-B6, VoxCeleb2-dev only (no speaker overlap with VoxCeleb1), large-margin fine-tuned ("lm")
    redimnet_label="ReDimNet2-B6", redimnet_repo="https://github.com/PalabraAI/redimnet2.git",
    redimnet_commit="c5bbe0b76e37df698c403f8844e41304ceab6307",   # source code pinned (2026-08-28)
    redimnet_model="b6", redimnet_dataset="vox2", redimnet_train_type="lm",
    redimnet_weights_sha256="bc45f032099f3c9f0fb6ad756bca23daeeba456d00c515056e388824ef36f41b",   # b6-vox2-lm.pt (release v1.0.0, 50.9 MB)
    redimnet_dir="/content/redimnet2_src", torch_home="/content/torch_cache",
    teacher_amp_dtype="float16",                      # used only if it matches fp32 (cos >= 0.999, checked); same choice on every GPU -> caches stay valid
    teacher_batch_seconds=240.0, teacher_min_speaker_gap=0.2,
    selection_val_utts_per_speaker=30, selection_trials=20000,   # normalisation experiment on the held-out DEV speakers (never the test set)
    # ---- student (Zipformer-M, unchanged) + SV branch
    sv_block=3, msv_attention_channels=128, cos_eps=1e-8,
    # ---- optimisation (MT2KD / icefall: ScaledAdam + Eden, base lr 0.045)
    optimizer="scaled_adam", base_lr=0.045, lr_batches=7500, lr_epochs=3.5, eden_warmup_batches=500, eden_warmup_start=0.1, clipping_scale=2.0,
    msv_lr_scale=1.0,                                 # separate parameter group for M_SV (same lr by default)
    adamw_lr=3e-4, weight_decay=1e-2, warmup_updates=500, min_lr_frac=0.05, grad_clip=1.0,    # used only with optimizer="adamw"
    batch_count_mode="progress",                      # "progress": Zipformer's step-dependent regularisation (layer skipping, dropout) follows training
                                                      # progress, reaching icefall's end values by the end of a short run; "icefall": icefall's own formula
    max_batch_seconds=600.0, max_batch_utts=128, accum_steps=1, ref_duration=600.0, mem_fraction=0.85,
    local_ckpt_every_updates=200, drive_ckpt_every_updates=1000, ckpt_every_minutes=10.0, drive_ckpt_every_minutes=20.0,
    log_every=20, num_workers=min(4, os.cpu_count() or 2), max_consecutive_skips=20, min_delta=1e-4,
    stall_minutes=15,                                 # archive extraction: stop (instead of hanging) if nothing arrives from Drive for this long
    learning_check_min_cos=0.5,                       # quick check: on 64 training utterances the student must reach this cosine with the teacher
)
MODES = {
    "sanity": dict(max_epochs=10, patience=99, eden_warmup_batches=50),    # pipeline check on ~2,000 utterances: NOT a research result
    "full":   dict(max_epochs=10, patience=3),
}
OFFICIAL = dict(dev_speakers=1211, dev_utts=148642, test_speakers=40, test_utts=4874, veri_test=37720, veri_test2=37611)
TEST_SPEAKERS_DOC = [f"id{n}" for n in range(10270, 10310)]                # VoxCeleb1 test speakers (id10270-id10309), used only if no trial list is found
SR = CFG["sample_rate"]

def log(*a): print(*a, flush=True)
DRIVE = "/content/drive"
def assert_local(p):
    """Every LOCAL write goes through here: refuses Google Drive (Drive writes exist only in class DriveStore)."""
    rp = os.path.realpath(str(p))
    if rp.startswith(DRIVE) or str(p).startswith(DRIVE): raise PermissionError(f"refusing to write to Google Drive: {p}")
    return p

def paths(mode=None):
    W = Path(assert_local(CFG["write_root"]))
    P = dict(meta=W / "meta", teacher=W / "teacher", logs=W / "logs", ckpt=W / "checkpoints", results=W / "results")
    if mode:
        for k in ["ckpt", "logs", "results"]: P[k] = P[k] / mode
        P.update(best=P["ckpt"] / "best", latest=P["ckpt"] / "latest")
    return P

def ensure_dirs(P):
    for k, v in P.items():
        if isinstance(v, Path): assert_local(v); v.mkdir(parents=True, exist_ok=True)

def pip_if_missing(*mods):
    for m in mods:
        mod, pkg = (m, m) if isinstance(m, str) else m
        if importlib.util.find_spec(mod) is None: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

def atomic_json(obj, p):
    p = Path(assert_local(p)); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + ".tmp")
    json.dump(obj, open(tmp, "w"), indent=1, default=str); os.replace(tmp, p)

def atomic_csv(df, p):
    p = Path(assert_local(p)); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + ".tmp"); df.to_csv(tmp, index=False); os.replace(tmp, p)

def atomic_torch_save(obj, p):
    import torch
    p = Path(assert_local(p)); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + ".tmp"); torch.save(obj, tmp); os.replace(tmp, p)

def sha_of(items): return hashlib.sha1("\n".join(map(str, items)).encode()).hexdigest()[:12]

def file_sha1(p, chunk=1 << 22):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()

def file_sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()

def set_seed(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def gpu_mem():
    import torch
    if not torch.cuda.is_available(): return {}
    return dict(gpu_alloc_gib=torch.cuda.memory_allocated() / 2**30, gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)

def disk_free_gib(p):
    p = Path(p)
    while not p.exists(): p = p.parent
    return shutil.disk_usage(p).free / 2**30

def require_space(path, need_gib, what, margin_gib=1.0):
    free = disk_free_gib(path)
    if need_gib > free - margin_gib: raise RuntimeError(f"not enough space for {what}: need {need_gib:.1f} GiB (+{margin_gib} margin), {free:.1f} GiB free at {path}")
    return free

def amp_dtype():
    import torch
    return torch.bfloat16 if torch.cuda.get_device_properties(0).major >= 8 else torch.float16

def env_report():
    import torch, platform
    r = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, gpu=None, code_version=CODE_VERSION)
    for m in ["torchaudio", "scipy", "soundfile"]:
        try: r[m] = importlib.import_module(m).__version__
        except Exception as e: r[m] = f"missing ({type(e).__name__})"
    try:
        import k2; r["k2"] = getattr(k2, "__version__", "installed")
    except Exception: r["k2"] = "not installed (pure-PyTorch Swoosh port used; verified)"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0); r.update(gpu=p.name, gpu_gib=round(p.total_memory / 2**30, 1), compute_capability=f"{p.major}.{p.minor}",
                                                           amp="bfloat16" if p.major >= 8 else "float16 + GradScaler")
    r["disk_free_local_gib"] = round(disk_free_gib("/content" if Path("/content").exists() else "/"), 1)
    for k, v in r.items(): log(f"  {k:22s}: {v}")
    if r["gpu"] is None: raise RuntimeError("no GPU: Runtime > Change runtime type > GPU")
    return r
# ####################################################################################################
# ---- Google Drive output store: the ONLY code in this cell that writes to Google Drive
# ####################################################################################################
class SimulatedDisconnect(Exception):
    """Raised by the resume test to imitate a runtime that dies: nothing is saved at that moment."""

class DriveStore:
    """Writes ONLY into one folder and ONLY the file names in `allowed` (plus files under the sub-folder prefixes in `prefixes`, and '<name>.tmp'
    while a write is in progress).
    * small files (config.json, history.csv, final_report.json): written synchronously, atomically (.tmp then rename)
    * checkpoints (latest.pt, best.pt): copied in a background thread from the local checkpoint (training does not wait),
      written to '<name>.tmp' first and renamed only when the copy is complete -> an interrupted upload never replaces a good file
    * with the Drive API (CFG['drive_api']): each upload is confirmed by comparing Drive's md5 with the local md5, and the copies that
      Drive moves to the trash when a checkpoint is replaced are deleted permanently (only files of THIS folder with THESE names),
      because trashed files keep using Drive quota for 30 days.
    fake=True: a local stand-in folder (used by the resume test; nothing goes to Drive)."""
    def __init__(self, folder, allowed, prefixes=(), use_api=True, fake=False, api_from=None):
        self.dir, self.fake, self.allowed, self.prefixes = Path(folder), fake, tuple(allowed), tuple(prefixes)
        if fake: assert_local(folder)
        else:
            if not os.path.isdir(DRIVE + "/MyDrive"): raise RuntimeError("Google Drive is not mounted")
            # the path as written must be inside MyDrive; its resolved location may be elsewhere on the Drive mount, because a folder
            # such as MT2KD can be a Drive SHORTCUT (resolved to /content/drive/.shortcut-targets-by-id/...)
            if not os.path.normpath(os.path.abspath(str(folder))).startswith(DRIVE + "/MyDrive/"): raise PermissionError(f"Drive output folder must be inside MyDrive: {folder}")
            if not os.path.realpath(str(folder)).startswith(os.path.realpath(DRIVE) + "/"): raise PermissionError(f"{folder} resolves outside Google Drive")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.api = self.fid = None; self.api_note = "not used (stand-in folder)" if fake else "disabled (CFG['drive_api'] = False)"
        if api_from is not None: self.api, self.api_note = api_from.api, api_from.api_note       # share one sign-in between sub-folders
        elif use_api and not fake: self._init_api()
        self.cv = threading.Condition(); self.pending = {}; self.busy = None; self.errors = []; self.done = {}; self._ids = itertools.count(); self.confirmers = []
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------------------------------------------------------------- guards / API
    def _ok(self, name):
        base = name[:-4] if name.endswith(".tmp") else name
        if ".." in Path(base).parts or base.startswith("/"): return False
        return base in self.allowed or any(base.startswith(p) and len(base) > len(p) for p in self.prefixes)
    def _p(self, name):
        if not self._ok(name): raise PermissionError(f"'{name}' is not allowed in {self.dir}: names {self.allowed}, sub-folders {self.prefixes}")
        f = self.dir / name; f.parent.mkdir(parents=True, exist_ok=True); return f
    def _init_api(self):
        try:
            from google.colab import auth; auth.authenticate_user()
            from googleapiclient.discovery import build
            self.api = build("drive", "v3", cache_discovery=False)
            self.api_note = "ON: uploads confirmed by md5; trashed copies of this folder's checkpoints are deleted permanently"
        except Exception as e:
            self.api = None; self.api_note = (f"UNAVAILABLE ({type(e).__name__}: {e}): uploads are not confirmed and replaced checkpoints stay in the Drive "
                                              f"trash (empty it to recover the space)")
    def _folder_id(self):
        if self.fid or self.api is None: return self.fid
        rel = os.path.relpath(os.path.normpath(os.path.abspath(str(self.dir))), DRIVE + "/MyDrive").split(os.sep); parent = "root"
        try:
            for part in rel:                                  # walk the path as written; follow Drive shortcuts to their target folder
                q = (f"name = '{part}' and '{parent}' in parents and trashed = false and "
                     f"(mimeType = 'application/vnd.google-apps.folder' or mimeType = 'application/vnd.google-apps.shortcut')")
                r = self.api.files().list(q=q, fields="files(id,mimeType,shortcutDetails)", pageSize=10, supportsAllDrives=True,
                                          includeItemsFromAllDrives=True).execute().get("files", [])
                if not r: return None                         # folder not synced to Drive yet: retried later
                x = r[0]; parent = x["shortcutDetails"]["targetId"] if x["mimeType"].endswith("shortcut") else x["id"]
            self.fid = parent
        except Exception: return None
        return self.fid

    # ---------------------------------------------------------------- storage information
    def quota(self):
        if self.api:
            try:
                q = self.api.about().get(fields="storageQuota").execute()["storageQuota"]; lim = int(q.get("limit", 0)); use = int(q.get("usage", 0))
                return dict(limit=lim or None, usage=use, trash=int(q.get("usageInDriveTrash", 0)), free=(lim - use) if lim else None, source="Drive API")
            except Exception: pass
        du = shutil.disk_usage(self.dir if self.fake else DRIVE + "/MyDrive")
        return dict(limit=du.total, usage=du.used, trash=None, free=du.free, source="mounted file system (approximate)")
    def files(self): return {f.name: f.stat().st_size for f in sorted(self.dir.iterdir()) if f.is_file()}
    def status(self, expected_bytes=None):
        q = self.quota(); fl = self.files(); g = lambda b: "?" if b is None else f"{b / 2**30:.2f} GiB"
        log(f"  Drive output folder : {self.dir}")
        log(f"  folder size now     : {g(sum(fl.values()))} | files: " + (", ".join(f"{k} ({v / 2**20:.1f} MiB)" for k, v in fl.items()) or "none"))
        log(f"  Drive storage       : free {g(q['free'])} of {g(q['limit'])} (used {g(q['usage'])}" + (f", in trash {g(q['trash'])}" if q["trash"] is not None else "")
            + f") [{q['source']}]")
        log(f"  Drive API           : {self.api_note}")
        if expected_bytes: log(f"  this run will keep  : about {g(expected_bytes)} in this folder")
        unexpected = [k for k in fl if not self._ok(k)]
        if unexpected: log(f"  NOTE: files not written by this code are in the folder (left untouched): {unexpected}")
        return dict(quota=q, files=fl)
    def check_space(self, need_bytes, what):
        """Refuses to start writing large files when the free Drive space is below what the next writes need."""
        q = self.quota()
        if q["free"] is not None and q["free"] < need_bytes:
            raise RuntimeError(f"Google Drive has {q['free'] / 2**30:.2f} GiB free; {what} needs about {need_bytes / 2**30:.2f} GiB. Nothing large was written. "
                               f"Free at least {(need_bytes - q['free']) / 2**30:.2f} GiB more on Drive (deleted files only free space once the Drive trash is "
                               f"emptied) and re-run the cell.")
        free = "?" if q["free"] is None else f"{q['free'] / 2**30:.2f}"
        log(f"  Drive space check OK for {what}: need ~{need_bytes / 2**30:.2f} GiB, free {free} GiB")

    # ---------------------------------------------------------------- writes
    def write_text(self, name, text):
        tmp, dst = self._p(name + ".tmp"), self._p(name)
        with open(tmp, "w") as fh: fh.write(text); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, dst)
    def write_json(self, name, obj): self.write_text(name, json.dumps(obj, indent=1, default=str))
    def push(self, local_file, name):
        """Queues an upload of a local checkpoint file. The file is snapshotted NOW (hard link), so later local saves cannot change what is
        uploaded; a newer request for the same name replaces a queued (not yet started) one."""
        self._p(name); snap = f"{local_file}.upload{next(self._ids)}"
        try: os.link(local_file, snap)
        except OSError: shutil.copyfile(local_file, snap)
        with self.cv:
            if self.errors: os.remove(snap); raise RuntimeError(f"a Google Drive upload failed: {self.errors[-1]} (training stopped; local state is saved)")
            old = self.pending.get(name); self.pending[name] = snap; self.cv.notify_all()
        if old and os.path.exists(old): os.remove(old)
    def _worker(self):
        while True:
            with self.cv:
                while not self.pending: self.cv.wait()
                name = next(iter(self.pending)); src = self.pending.pop(name); self.busy = name
            try: self._upload(src, name)
            except Exception as e: self.errors.append(f"{name}: {type(e).__name__}: {e}"); log(f"  GOOGLE DRIVE UPLOAD FAILED ({name}): {type(e).__name__}: {e}")
            finally:
                try: os.remove(src)
                except OSError: pass
                with self.cv: self.busy = None; self.cv.notify_all()
    def _upload(self, src, name):
        t0 = time.time(); tmp, dst = self._p(name + ".tmp"), self._p(name); h = hashlib.md5(); n = 0
        with open(src, "rb") as fi, open(tmp, "wb") as fo:          # the open handle keeps a consistent snapshot even if the local file is replaced meanwhile
            for b in iter(lambda: fi.read(1 << 24), b""): fo.write(b); h.update(b); n += len(b)
            fo.flush(); os.fsync(fo.fileno())
        os.replace(tmp, dst); md5 = h.hexdigest()
        self.done[name] = dict(bytes=n, md5=md5, written=time.strftime("%Y-%m-%d %H:%M:%S"), confirmed_by_drive_api=None)
        log(f"  Drive: {name} written ({n / 2**20:.0f} MiB in {time.time() - t0:.0f} s)" + ("; waiting for Drive to confirm the upload" if self.api else ""))
        if self.api:                                     # confirmation runs on its own thread: the next upload (and training) never waits for it
            def cf():
                self.done[name]["confirmed_by_drive_api"] = self._confirm(name, md5); self.purge_trash()
            t = threading.Thread(target=cf, daemon=True); t.start(); self.confirmers.append(t)
    def _confirm(self, name, md5, timeout=900):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.cv:
                if name in self.pending: return "superseded by a newer upload"
            fid = self._folder_id()
            if fid:
                try:
                    r = self.api.files().list(q=f"name = '{name}' and '{fid}' in parents and trashed = false", fields="files(id,md5Checksum)").execute().get("files", [])
                    if any(x.get("md5Checksum") == md5 for x in r): log(f"  Drive: {name} upload CONFIRMED (md5 {md5[:10]} matches)"); return True
                except Exception: pass
            time.sleep(15)
        log(f"  Drive: {name} upload not confirmed within {timeout // 60} min (it may still be syncing)"); return False
    def purge_trash(self):
        """Deletes permanently the TRASHED copies of this folder's own files (replaced checkpoints); nothing else is touched."""
        fid = self._folder_id()
        if not fid: return 0
        freed = 0
        try:
            for x in self.api.files().list(q=f"'{fid}' in parents and trashed = true", fields="files(id,name,size)", pageSize=200).execute().get("files", []):
                base = x["name"][:-4] if x["name"].endswith(".tmp") else x["name"]
                if self._ok(base): self.api.files().delete(fileId=x["id"]).execute(); freed += int(x.get("size", 0))
        except Exception as e: log(f"  (could not clean this folder's trashed copies: {type(e).__name__}: {e})")
        if freed: log(f"  Drive: {freed / 2**20:.0f} MiB of replaced copies removed from the trash")
        return freed
    def wait(self, timeout=None):
        """Blocks until every queued upload is written (and confirmed, with the API)."""
        t0 = time.time()
        with self.cv:
            while self.pending or self.busy:
                if timeout and time.time() - t0 > timeout: return False
                self.cv.wait(5)
        for t in self.confirmers: t.join(timeout=max(1.0, 180 - (time.time() - t0)))      # at most ~3 min for Drive's confirmation
        if any(t.is_alive() for t in self.confirmers): log("  (Drive is still syncing; the files are written and will finish uploading in the background)")
        self.confirmers = [t for t in self.confirmers if t.is_alive()]
        if self.api: self.purge_trash()
        if self.errors: raise RuntimeError(f"Google Drive upload failed: {self.errors[-1]}")
        return True

    # ---------------------------------------------------------------- reading back
    def candidates(self, name): return [f for f in (self._p(name), self._p(name + ".tmp")) if f.exists()]
    def promote_tmp(self, name):
        """A complete '<name>.tmp' whose rename was interrupted becomes '<name>'."""
        os.replace(self._p(name + ".tmp"), self._p(name)); return self._p(name)
    def clean_stale_tmp(self):
        for f in self.dir.glob("*.tmp"):
            if self._ok(f.name[:-4]): f.unlink(); log(f"  removed an incomplete upload left by an interrupted run: {f.name}")
    def best_info(self, name="checkpoint_best.pt"): return best_meta(self._p(name)) if self._p(name).exists() else None
    def write_file(self, local_file, name):
        """Synchronous atomic copy of a small local file (e.g. a teacher-cache shard)."""
        tmp, dst = self._p(name + ".tmp"), self._p(name)
        with open(local_file, "rb") as fi, open(tmp, "wb") as fo: shutil.copyfileobj(fi, fo, 1 << 22); fo.flush(); os.fsync(fo.fileno())
        os.replace(tmp, dst)
    def write_csv(self, name, df): self.write_text(name, df.to_csv(index=False))
    def exists(self, name): return self._ok(name) and (self.dir / name).exists()
    def path(self, name): return self._p(name) if self._ok(name) else None


def best_meta(f):
    import torch
    try: ck = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        try: ck = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e: return dict(error=f"{type(e).__name__}: {e}")
    r = dict(val_loss=float(ck["val"]["val_loss"]), epoch=int(ck["epoch"]), update=int(ck["update"]), sig=ck.get("sig")); del ck
    return r

# ####################################################################################################
# ---- VoxCeleb1: inspection of the real folder, header-level inventory, OFFICIAL partitions, manifests, trial lists, local copies
# ####################################################################################################
UID_RE = re.compile(r"(id\d{5})/([^/]+)/([^/]+\.wav)$")
TAR_RE = re.compile(r"(\.tar\.gz|\.tgz|\.tar)(\.part-?\d+|\.\d{3})?$")
TRIAL_NAMES = ("veri_test2.txt", "veri_test.txt", "list_test_all2.txt", "list_test_all.txt", "list_test_hard2.txt", "list_test_hard.txt",
               "iden_split.txt", "vox1_meta.csv")
OFFICIAL_TRIALS = ("veri_test2.txt", "veri_test.txt")       # VoxCeleb1-O (cleaned, original): test speakers only. E/H lists use DEV speakers -> not usable here

def uid_of(path):
    m = UID_RE.search(str(path).replace("\\", "/"))
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else None

def parse_wav_header(b, file_size=None):
    """RIFF/WAVE header -> dict(sr, channels, bits, fmt, n_frames). Walks the chunks (LIST etc. are skipped). Raises ValueError if invalid."""
    if len(b) < 12 or b[:4] not in (b"RIFF", b"RF64") or b[8:12] != b"WAVE": raise ValueError("not a RIFF/WAVE file")
    pos, fmt, data_size, data_off = 12, None, None, None
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack("<I", b[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            if pos + 8 + 16 > len(b): raise ValueError("truncated fmt chunk")
            af, ch, sr, _, ba, bits = struct.unpack("<HHIIHH", b[pos + 8:pos + 24]); fmt = dict(fmt=af, channels=ch, sr=sr, block_align=ba, bits=bits)
        elif cid == b"data":
            data_size, data_off = size, pos + 8; break
        pos += 8 + size + (size & 1)
    if fmt is None: raise ValueError("no fmt chunk")
    if data_size is None: raise ValueError("no data chunk in the first bytes")
    if file_size is not None and (data_size in (0, 0xFFFFFFFF) or data_off + data_size > file_size): data_size = max(0, file_size - data_off)
    if fmt["block_align"] <= 0: raise ValueError("block_align 0")
    return dict(**fmt, n_frames=data_size // fmt["block_align"])

def inspect_vox_root():
    """Real directory structure: audio folders (idXXXXX/<video>/<n>.wav), zip archives, split archives, metadata / trial files."""
    root = Path(CFG["vox_root"])
    if not root.is_dir(): raise FileNotFoundError(f"VoxCeleb1 folder not found: {root}")
    dirs, zips, parts, tars, meta, tree = [], [], [], [], {}, []
    for dp, dn, fn in os.walk(root):
        rel = Path(dp).relative_to(root); depth = len(rel.parts)
        ids = [d for d in dn if re.fullmatch(r"id\d{5}", d)]
        if ids: dirs.append(dict(base=dp, speakers=len(ids))); dn[:] = [d for d in dn if not re.fullmatch(r"id\d{5}", d)]
        if depth <= 2: tree.append(f"{'  ' * depth}{rel if depth else root.name}/  ({len(dn)} folders, {len(fn)} files" +
                                   (f", {len(ids)} speaker folders" if ids else "") + (f"; e.g. {sorted(fn)[:4]}" if fn else "") + ")")
        for f in fn:
            if f.lower().endswith(".zip"): zips.append(str(Path(dp) / f))
            elif re.search(r"vox1_(dev|test)_wav_parta[a-z]$", f): parts.append(str(Path(dp) / f))
            elif TAR_RE.search(f): tars.append(str(Path(dp) / f))
            if f in TRIAL_NAMES: meta.setdefault(f, str(Path(dp) / f))
        if depth >= 4: dn[:] = []
        dn[:] = [d for d in dn if not d.startswith(".")]
    log("  VoxCeleb1 folder (top levels):\n" + "\n".join("    " + t for t in tree[:40]))
    log(f"  audio folders: {[(Path(d['base']).relative_to(root).as_posix() or '.', d['speakers']) for d in dirs]} | zip archives: {[Path(z).name for z in zips]} | "
        f"split archives: {[Path(p).name for p in parts]} | tar archives: {len(tars)} file(s) {[Path(t).name for t in tars][:3]}{' ...' if len(tars) > 3 else ''} | "
        f"metadata: {sorted(meta)}")
    return dict(root=str(root), dirs=dirs, zips=zips, parts=parts, tars=tars, meta=meta)

def _id_dirs(base):
    out = []
    for dp, dn, fn in os.walk(base):
        ids = [d for d in dn if re.fullmatch(r"id\d{5}", d)]
        if ids: out.append(dict(base=dp, speakers=len(ids))); dn[:] = [d for d in dn if not re.fullmatch(r"id\d{5}", d)]
    return out

def extract_tars(tars):
    """tar / tar.gz archives, also split into pieces (<name>.tar.gz.part-000, -001, ...): the pieces are concatenated in order and streamed into
    ONE `tar` process that extracts only *.wav and *.txt files to LOCAL disk (never to Drive). A completion marker makes this a one-time step."""
    groups = {}
    for t in tars: groups.setdefault(TAR_RE.sub(lambda m: m.group(1), Path(t).name), []).append(t)
    dst = Path(assert_local(CFG["extract_dir"])) / "full"; dst.mkdir(parents=True, exist_ok=True)
    for g, ps in sorted(groups.items()):
        ps = sorted(ps, key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", Path(p).name)]); mark = dst / f".extracted_{g}.ok"
        size = sum(os.path.getsize(p) for p in ps) / 2**30
        if mark.exists() and mark.read_text() == json.dumps([Path(p).name for p in ps]): log(f"  {g}: already extracted to {dst}"); continue
        require_space(dst, size * 1.15, f"extracting {g}", 5.0)
        gz = g.endswith((".tar.gz", ".tgz")); dec = (["-I", "pigz"] if shutil.which("pigz") else ["-z"]) if gz else []
        log(f"  extracting {g} ({len(ps)} piece(s), {size:.1f} GiB, read from Drive once) -> {dst} (local disk; *.wav and *.txt only) ...")
        t0 = time.time(); stop = threading.Event(); stalled = []
        cat = subprocess.Popen(["cat"] + ps, stdout=subprocess.PIPE)
        tarp = subprocess.Popen(["tar", "-x", *dec, "-f", "-", "-C", str(dst), "--wildcards", "*.wav", "*.txt"], stdin=cat.stdout, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        cat.stdout.close()
        def mon():                                       # progress every minute; no growth for CFG['stall_minutes'] -> stop instead of hanging
            last, t_last = -1.0, time.time()
            while not stop.wait(60):
                try: got = int(subprocess.run(["du", "-sb", str(dst)], capture_output=True, text=True).stdout.split()[0]) / 2**30
                except Exception: got = last
                log(f"    ... {got:.1f} GiB extracted | {(time.time() - t0) / 60:.0f} min")
                if got > last + 1e-3: last, t_last = got, time.time()
                elif time.time() - t_last > 60 * CFG["stall_minutes"]:
                    stalled.append(True); log(f"    no progress for {CFG['stall_minutes']} min (Drive read stalled?): stopping"); tarp.kill(); cat.kill(); return
        th = threading.Thread(target=mon, daemon=True); th.start()
        _, err = tarp.communicate(); cat.wait(); stop.set()
        if stalled: raise RuntimeError(f"extracting {g} stalled (Google Drive stopped delivering data). Re-run the cell: extraction restarts.")
        r = subprocess.CompletedProcess(tarp.args, tarp.returncode, "", err)
        errs = [l for l in (r.stderr or "").splitlines() if l.strip() and "*.txt: Not found in archive" not in l and "Exiting with failure status due to previous errors" not in l]
        if "*.wav: Not found in archive" in (r.stderr or ""): raise RuntimeError(f"{g} contains no .wav files")
        if (r.returncode and errs) or cat.returncode:
            raise RuntimeError(f"extracting {g} failed (tar exit {r.returncode}, cat exit {cat.returncode}): {(r.stderr or '')[-500:]} -- if a piece is missing or damaged, "
                               f"the archive cannot be read past it; re-run to retry")
        mark.write_text(json.dumps([Path(p).name for p in ps])); log(f"  {g}: extracted in {(time.time() - t0) / 60:.1f} min")
    return dst

SUBSET_TAR_SCRIPT = r"""
# streams a (concatenated) tar.gz from stdin and keeps ONLY the requested utterances; stops reading as soon as they are all extracted
import sys, os, re, json, time, tarfile
dst, want_file, quota, prog = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
want = set(json.load(open(want_file))); test_spk = {u.split("/")[0] for u in want}
UID = re.compile(r"(id\d{5})/([^/]+)/([^/]+\.wav)$"); TR = ("veri_test2.txt", "veri_test.txt")
class Count:
    def __init__(s, f): s.f, s.n = f, 0
    def read(s, k=-1):
        b = s.f.read(k); s.n += len(b); return b
src = Count(sys.stdin.buffer); seen = kept_t = kept_d = 0; t_last = 0; stopped = "end of archive"
def report(final=False):
    json.dump(dict(read_bytes=src.n, members=seen, test_kept=kept_t, test_wanted=len(want), dev_kept=kept_d, dev_quota=quota, final=final, stopped=stopped),
              open(prog + ".tmp", "w")); os.replace(prog + ".tmp", prog)
with tarfile.open(fileobj=src, mode="r|gz") as tf:
    for m in tf:
        seen += 1
        if time.time() - t_last > 10: report(); t_last = time.time()
        if not m.isfile(): continue
        base = os.path.basename(m.name); mt = UID.search(m.name); keep = False
        if mt:
            u = f"{mt.group(1)}/{mt.group(2)}/{mt.group(3)}"
            if u in want: keep = True; kept_t += 1
            elif mt.group(1) not in test_spk and kept_d < quota: keep = True; kept_d += 1
        elif base in TR: keep = True; u = base
        if keep:
            p = os.path.join(dst, u); os.makedirs(os.path.dirname(p), exist_ok=True); f = tf.extractfile(m)
            with open(p + ".part", "wb") as o: o.write(f.read())
            os.replace(p + ".part", p)
        if kept_t >= len(want) and kept_d >= quota: stopped = "all requested utterances extracted"; break
report(final=True)
"""

def extract_subset(tars, want, quota):
    """SANITY extraction: streams the archive and keeps only the official test utterances (from the trial list) plus the first `quota`
    dev utterances met; stops reading Drive as soon as both are complete. A .tar.gz must be read in order, so how much of it is read depends on
    where these files sit in the archive (printed)."""
    groups = {}
    for t in tars: groups.setdefault(TAR_RE.sub(lambda m: m.group(1), Path(t).name), []).append(t)
    if len(groups) != 1: raise RuntimeError(f"subset extraction expects one archive, found {sorted(groups)}")
    (g, ps), = groups.items(); ps = sorted(ps, key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", Path(p).name)])
    dst = Path(assert_local(CFG["extract_dir"])) / "subset"; dst.mkdir(parents=True, exist_ok=True)
    rule = dict(archive=[Path(p).name for p in ps], test_utts=len(want), dev_quota=quota); mark = dst / ".subset.ok"
    if mark.exists() and json.load(open(mark)).get("rule") == rule: log(f"  subset already extracted to {dst}"); return dst, json.load(open(mark))
    size = sum(os.path.getsize(p) for p in ps) / 2**30; wf = dst / "want.json"; json.dump(sorted(want), open(wf, "w")); prog = dst / "progress.json"
    script = Path(assert_local(CFG["write_root"])) / "subset_tar.py"; script.write_text(SUBSET_TAR_SCRIPT)
    log(f"  SUBSET extraction from {g} ({len(ps)} pieces, {size:.1f} GiB in total): keeping the {len(want):,} official test utterances + the first "
        f"{quota:,} dev utterances, then stopping (local disk {dst})")
    t0 = time.time(); cat = subprocess.Popen(["cat"] + ps, stdout=subprocess.PIPE)
    ch = subprocess.Popen([sys.executable, "-I", str(script), str(dst), str(wf), str(quota), str(prog)], stdin=cat.stdout, stderr=subprocess.PIPE, text=True)
    cat.stdout.close(); last, t_last, stalled = -1, time.time(), False
    while ch.poll() is None:
        time.sleep(15)
        try: p = json.load(open(prog))
        except Exception: p = dict(read_bytes=0, members=0, test_kept=0, dev_kept=0)
        if p["read_bytes"] > last: last, t_last = p["read_bytes"], time.time()
        elif time.time() - t_last > 60 * CFG["stall_minutes"]: stalled = True; ch.kill(); cat.kill(); break
        if int(time.time() - t0) % 60 < 15:
            log(f"    ... read {p['read_bytes'] / 2**30:.1f} of {size:.1f} GiB | test {p['test_kept']:,}/{len(want):,} | dev {p['dev_kept']:,}/{quota:,} | {(time.time() - t0) / 60:.0f} min")
    err = ch.stderr.read() if ch.stderr else ""; cat.kill(); cat.wait()
    if stalled: raise RuntimeError("subset extraction stalled (Google Drive stopped delivering data). Re-run the cell to restart it.")
    if ch.returncode: raise RuntimeError(f"subset extraction failed (exit {ch.returncode}): {err[-500:]}")
    p = json.load(open(prog)); p.update(rule=rule, minutes=round((time.time() - t0) / 60, 1))
    log(f"  subset done in {p['minutes']} min: read {p['read_bytes'] / 2**30:.1f} of {size:.1f} GiB ({p['stopped']}) | test {p['test_kept']:,}/{len(want):,} | dev {p['dev_kept']:,}")
    if p["test_kept"] < len(want): log(f"  WARNING: only {p['test_kept']} of {len(want)} trial utterances are in the archive: the official EER will be reported as NOT official")
    atomic_json(p, mark); return dst, p

def _trial_utts(I):
    u = set()
    for n in OFFICIAL_TRIALS:
        if n in I["meta"]: t = read_trials(I["meta"][n]); u |= set(t.enrol) | set(t.test)
    return u

def effective_layout(mode="full"):
    """The audio sources actually used.
    full  : tar archives are the complete copy -> extracted locally and used INSTEAD of a (possibly partial) extracted folder on Drive.
    sanity: as little as possible is read from Drive: (1) an existing audio folder on Drive is used directly if it already holds every official
            trial utterance and >= CFG['sanity_dev_utts'] dev utterances; (2) otherwise only a SUBSET of the archive is extracted (the trial
            utterances + CFG['sanity_dev_utts'] dev utterances), and reading stops as soon as it is complete."""
    f = paths()["meta"] / f"vox_layout_{mode}.json"
    if f.exists(): return json.load(open(f))
    I = inspect_vox_root(); I["mode"] = mode; want = _trial_utts(I); quota = CFG["sanity_dev_utts"]
    if mode == "sanity" and I["dirs"]:
        have_t = sum(any(os.path.exists(Path(d["base"]) / u) for d in I["dirs"]) for u in sorted(want)) if want else 0
        n_dev = 0; tspk = {u.split("/")[0] for u in want}
        for d in I["dirs"]:
            for sp in sorted(os.listdir(d["base"])):
                if re.fullmatch(r"id\d{5}", sp) and sp not in tspk:
                    for dp, _, fn in os.walk(Path(d["base"]) / sp): n_dev += sum(x.endswith(".wav") for x in fn)
                if n_dev >= quota: break
        log(f"  Drive audio folder: {have_t}/{len(want)} trial utterances present, >= {n_dev:,} dev utterances")
        if (not want or have_t == len(want)) and n_dev >= quota:
            I["source_rule"] = "existing Drive audio folder (no archive read)"; atomic_json(I, f); return I
    if I["tars"]:
        if mode == "sanity":
            dst, p = extract_subset(I["tars"], want, quota); I["subset"] = p
            I["source_rule"] = f"SUBSET of the archive: {p['test_kept']} test + {p['dev_kept']} dev utterances (sanity run only; NOT the full VoxCeleb1-dev)"
        else:
            dst = extract_tars(I["tars"]); I["source_rule"] = "complete archive, extracted locally"
        loc = _id_dirs(dst)
        if I["dirs"]: log(f"  NOTE: the Drive folder(s) {[d['base'] for d in I['dirs']]} are NOT used ({I['source_rule']})")
        I["drive_dirs_ignored"] = I["dirs"]; I["dirs"] = loc
        for dp, _, fn in os.walk(dst):
            for x in fn:
                if x in TRIAL_NAMES: I["meta"].setdefault(x, str(Path(dp) / x))
    else: I["source_rule"] = "audio folders / zip archives as found"
    atomic_json(I, f); return I

def _assemble_parts(parts):
    """vox1_*_wav_partaa.. (official download pieces) -> one local zip (local disk, not Drive)."""
    out = []
    for group in sorted({re.sub(r"_parta[a-z]$", "", Path(p).name) for p in parts}):
        ps = sorted(p for p in parts if Path(p).name.startswith(group + "_parta")); dst = Path(assert_local(CFG["write_root"])) / "archives" / f"{group}.zip"
        if not dst.exists():
            need = sum(os.path.getsize(p) for p in ps) / 2**30; require_space(dst.parent if dst.parent.exists() else "/content", need, f"assembling {group}.zip", 5.0)
            dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_suffix(".zip.part"); t0 = time.time()
            log(f"  assembling {group}.zip from {len(ps)} pieces ({need:.1f} GiB) on local disk ...")
            with open(tmp, "wb") as fo:
                for p in ps:
                    with open(p, "rb") as fi: shutil.copyfileobj(fi, fo, 1 << 24)
            if not zipfile.is_zipfile(tmp): tmp.unlink(); raise RuntimeError(f"{group}: the assembled pieces are not a valid zip")
            os.replace(tmp, dst); log(f"  {dst} ready ({time.time() - t0:.0f} s)")
        out.append(str(dst))
    return out

def inventory(mode="full", rebuild=False):
    """One row per WAV file: uid, speaker, source, header fields, duration, validity + reason. Header-level only (one small read per file)."""
    P = paths(); ensure_dirs(P); f_inv, f_bad = P["meta"] / f"inventory_{mode}.csv", P["meta"] / f"excluded_inventory_{mode}.csv"
    if f_inv.exists() and not rebuild:
        inv = pd.read_csv(f_inv, dtype={"uid": str, "speaker": str, "source": str, "member": str}, keep_default_na=False)
        log(f"  inventory (cached): {len(inv):,} files, {int(inv.valid.sum()):,} valid"); return inv
    I = effective_layout(mode); zips = list(I["zips"]) + (_assemble_parts(I["parts"]) if I["parts"] and mode == "full" else [])
    tasks = []
    for d in I["dirs"]:
        for dp, _, fn in os.walk(d["base"]):
            for f in fn:
                if f.lower().endswith(".wav"): tasks.append(("dir", str(Path(dp) / f), ""))
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zf:
                for zi in zf.infolist():
                    if zi.filename.lower().endswith(".wav") and uid_of(zi.filename): tasks.append(("zip", z, zi.filename))
        except zipfile.BadZipFile as e: log(f"  WARNING: unreadable zip {z}: {e}")
    if not tasks: raise RuntimeError(f"no VoxCeleb1 WAV files found under {CFG['vox_root']} (looked for idXXXXX/<video>/<n>.wav in folders and zip archives)")
    log(f"  reading {len(tasks):,} WAV headers (one small read per file) ...")
    local = threading.local()
    def zf_of(z):
        if not hasattr(local, "z"): local.z = {}
        if z not in local.z: local.z[z] = zipfile.ZipFile(z)
        return local.z[z]
    def head(t):
        kind, src, mem = t; r = dict(uid=uid_of(mem or src), source=src, member=mem, kind=kind)
        r["speaker"] = r["uid"].split("/")[0] if r["uid"] else ""
        try:
            if kind == "dir":
                size = os.path.getsize(src)
                with open(src, "rb") as f: b = f.read(65536)
            else:
                zi = zf_of(src).getinfo(mem); size = zi.file_size
                with zf_of(src).open(mem) as f: b = f.read(65536)
            if size == 0: raise ValueError("empty file")
            h = parse_wav_header(b, size); r.update(sr=h["sr"], channels=h["channels"], bits=h["bits"], fmt=h["fmt"], n_frames=h["n_frames"], bytes=size)
            r["duration_s"] = h["n_frames"] / h["sr"] if h["sr"] else 0.0
            reason = ("not PCM/float WAV" if h["fmt"] not in (1, 3, 0xFFFE) else "no audio frames" if h["n_frames"] <= 0 else
                      "duration < 0.5 s" if r["duration_s"] < 0.5 else "")
        except Exception as e: reason = f"unreadable header: {type(e).__name__}: {e}"
        r["valid"] = not reason; r["reason"] = reason; return r
    t0 = time.time(); rows = []
    with ThreadPoolExecutor(32) as ex:
        for i, r in enumerate(ex.map(head, tasks, chunksize=64)):
            rows.append(r)
            if (i + 1) % 20000 == 0: log(f"    {i + 1:,}/{len(tasks):,} headers | {time.time() - t0:.0f} s")
    inv = pd.DataFrame(rows)
    for c in ["sr", "channels", "bits", "fmt", "n_frames", "bytes"]: inv[c] = inv.get(c, pd.Series(dtype=float)).fillna(0).astype(int)
    inv["duration_s"] = inv.get("duration_s", pd.Series(dtype=float)).fillna(0.0)
    dup = inv.uid.duplicated(keep="first") & inv.uid.ne("")
    if dup.any(): inv.loc[dup, ["valid", "reason"]] = [False, "duplicate uid (same file in another folder/archive; first copy kept)"]
    inv.loc[inv.uid.eq("") | inv.uid.isna(), ["valid", "reason"]] = [False, "path does not match idXXXXX/<video>/<n>.wav"]
    inv = inv.sort_values("uid", kind="stable").reset_index(drop=True); atomic_csv(inv, f_inv); atomic_csv(inv[~inv.valid], f_bad)
    log(f"  inventory: {len(inv):,} files, {int(inv.valid.sum()):,} valid, {int((~inv.valid).sum())} excluded (reasons recorded) | {time.time() - t0:.0f} s | "
        f"sample rates {inv[inv.valid].sr.value_counts().to_dict()} | channels {inv[inv.valid].channels.value_counts().to_dict()}")
    return inv

def read_trials(f):
    rows = []
    for ln in open(f):
        p = ln.split()
        if len(p) == 3 and p[0] in ("0", "1"): rows.append((int(p[0]), p[1], p[2]))
    return pd.DataFrame(rows, columns=["label", "enrol", "test"])

def prepare_dataset(mode="full"):
    """OFFICIAL partition (dev / test), the NON-OFFICIAL held-out validation speakers, manifests, trial lists, dataset report. Deterministic."""
    P = paths(); ensure_dirs(P); f_rep = P["meta"] / "dataset_validation_report.json"
    inv = inventory(mode); I = effective_layout(mode); ok = inv[inv.valid].copy()
    trials, tsrc = {}, None
    for name in OFFICIAL_TRIALS:
        if name in I["meta"]: trials[name] = read_trials(I["meta"][name])
    unusable = [n for n in I["meta"] if n.startswith("list_test")]
    if trials:
        tsrc = f"speakers of the official trial list(s) {sorted(trials)}"
        test_spk = sorted({u.split("/")[0] for t in trials.values() for u in pd.concat([t.enrol, t.test])})
    else:
        tests = ok[ok.source.str.contains("test", case=False) | ok.member.str.contains("test", case=False)]
        if len(tests) and tests.speaker.nunique() < ok.speaker.nunique():
            test_spk = sorted(tests.speaker.unique()); tsrc = "speakers stored in the 'test' folder/archive"
        else: test_spk = [s for s in TEST_SPEAKERS_DOC if s in set(ok.speaker)]; tsrc = "documented VoxCeleb1 test speaker ids id10270-id10309 (no trial list found)"
    test_set = set(test_spk); ok["partition"] = np.where(ok.speaker.isin(test_set), "test", "dev")
    dev_spk = sorted(ok[ok.partition == "dev"].speaker.unique()); rng = np.random.default_rng(CFG["seed"])
    n_val = CFG["n_val_speakers"] if I.get("subset") is None else min(CFG["n_val_speakers_subset"], max(1, len(dev_spk) // 5))
    val_spk = sorted(np.array(dev_spk)[rng.permutation(len(dev_spk))[:n_val]].tolist())
    ok.loc[ok.speaker.isin(set(val_spk)), "partition"] = "val"
    tr, va, te = (ok[ok.partition == p].sort_values("uid").reset_index(drop=True) for p in ("dev", "val", "test"))
    assert not set(tr.speaker) & set(te.speaker) and not set(va.speaker) & set(te.speaker) and not set(tr.speaker) & set(va.speaker)
    rng2 = np.random.default_rng(CFG["seed"] + 1); sub = tr.iloc[np.sort(rng2.permutation(len(tr))[:CFG["sanity_train_utts"]])].reset_index(drop=True)
    cols = ["uid", "speaker", "kind", "source", "member", "sr", "channels", "bits", "n_frames", "duration_s"]
    for name, df in [("train_manifest", tr), ("sanity_train_manifest", sub), ("validation_manifest", va), ("test_manifest", te)]: atomic_csv(df[cols], P["meta"] / f"{name}.csv")
    for name, t in trials.items(): atomic_csv(t, P["meta"] / f"trials_{name.replace('.txt', '')}.csv")
    have = set(te.uid); tri = {}
    for name, t in trials.items():
        miss = sorted((set(t.enrol) | set(t.test)) - have)
        tri[name] = dict(trials=len(t), target=int(t.label.sum()), utterances=len(set(t.enrol) | set(t.test)), missing_utterances=len(miss),
                         official_count=OFFICIAL.get(name.replace(".txt", "")), complete=not miss, first_missing=miss[:5])
    excl = inv[~inv.valid][["uid", "source", "member", "reason"]]
    rep = dict(code_version=CODE_VERSION, vox_root=CFG["vox_root"], mode=mode, source_rule=I.get("source_rule"), subset=I.get("subset"), layout=dict(audio_folders=I["dirs"], zip_archives=I["zips"], split_archives=I["parts"], tar_archives=I.get("tars", []),
                                                                       drive_folders_ignored=I.get("drive_dirs_ignored", []), metadata=I["meta"]),
               files=len(inv), valid=int(inv.valid.sum()), excluded=len(excl), excluded_reasons=excl.reason.str.split(":").str[0].value_counts().to_dict(),
               test_partition_source=tsrc,
               official=dict(dev=dict(speakers=int(tr.speaker.nunique() + va.speaker.nunique()), utterances=len(tr) + len(va), official_speakers=OFFICIAL["dev_speakers"],
                                      official_utterances=OFFICIAL["dev_utts"]),
                             test=dict(speakers=int(te.speaker.nunique()), utterances=len(te), official_speakers=OFFICIAL["test_speakers"], official_utterances=OFFICIAL["test_utts"])),
               train=dict(speakers=int(tr.speaker.nunique()), utterances=len(tr), hours=round(float(tr.duration_s.sum() / 3600), 2)),
               validation=dict(speakers=len(val_spk), utterances=len(va), hours=round(float(va.duration_s.sum() / 3600), 2), speaker_ids=val_spk,
                               note=f"NON-OFFICIAL: {len(val_spk)} VoxCeleb1-dev speakers held out with numpy default_rng({CFG['seed']}); VoxCeleb1 has no official SV validation split"),
               sanity_train=dict(utterances=len(sub), speakers=int(sub.speaker.nunique()), hours=round(float(sub.duration_s.sum() / 3600), 2),
                                 rule=f"numpy default_rng({CFG['seed'] + 1}) permutation of the training manifest, first {CFG['sanity_train_utts']}"),
               trials=tri, unusable_trial_lists={n: "uses VoxCeleb1-dev speakers, which are training data here" for n in unusable},
               durations=dict(min=round(float(ok.duration_s.min()), 2), median=round(float(ok.duration_s.median()), 2), mean=round(float(ok.duration_s.mean()), 2),
                              max=round(float(ok.duration_s.max()), 2), longer_than_crop=int((ok.duration_s > CFG["crop_seconds"]).sum())),
               sample_rates=ok.sr.value_counts().to_dict(), channels=ok.channels.value_counts().to_dict(), bits=ok.bits.value_counts().to_dict())
    atomic_csv(excl, P["meta"] / "excluded_files.csv"); atomic_json(rep, f_rep)
    log(f"  audio source: {I.get('source_rule')}")
    log(f"  OFFICIAL partition from {tsrc}:")
    log(f"    dev  : {rep['official']['dev']['speakers']} speakers / {rep['official']['dev']['utterances']:,} utterances (official {OFFICIAL['dev_speakers']} / {OFFICIAL['dev_utts']:,})")
    log(f"    test : {te.speaker.nunique()} speakers / {len(te):,} utterances (official {OFFICIAL['test_speakers']} / {OFFICIAL['test_utts']:,}) -> evaluation ONLY")
    log(f"  training   : {tr.speaker.nunique()} speakers / {len(tr):,} utterances / {rep['train']['hours']} h")
    log(f"  validation : {len(val_spk)} held-out dev speakers / {len(va):,} utterances (NON-OFFICIAL, speaker-disjoint from training and test)")
    log(f"  sanity     : {len(sub):,} training utterances ({sub.speaker.nunique()} speakers, {rep['sanity_train']['hours']} h)")
    for n, t in tri.items(): log(f"  trials {n}: {t['trials']:,} ({t['target']:,} target) | utterances {t['utterances']:,} | missing {t['missing_utterances']} -> "
                                 + ("COMPLETE (official EER possible)" if t["complete"] else "INCOMPLETE (no official EER from this list)"))
    if not tri: log("  NO official verification trial list found (veri_test2.txt / veri_test.txt): official EER cannot be computed; KD validation metrics only")
    for n in unusable: log(f"  {n}: not used (VoxCeleb1-E/H trials contain dev speakers = training data)")
    for k, (a, b) in dict(dev_speakers=(rep['official']['dev']['speakers'], OFFICIAL['dev_speakers']), test_utts=(len(te), OFFICIAL['test_utts'])).items():
        if a != b: log(f"  WARNING: {k} = {a}, official {b} (incomplete copy or extra files?)")
    return rep

def load_manifest(name): return pd.read_csv(paths()["meta"] / f"{name}.csv", dtype={"uid": str, "speaker": str, "source": str, "member": str}, keep_default_na=False)
def local_wav(uid): return Path(CFG["local_audio"]) / uid

def materialize(df, what):
    """Copies the manifest's WAV files to LOCAL disk (thread pool; resumable: complete files are skipped)."""
    assert_local(CFG["local_audio"]); need = [r for r in df.itertuples() if not local_wav(r.uid).exists()]      # copies are atomic (.part then rename)
    if not need: log(f"  [{what}] local audio ready ({len(df):,} files)"); return
    gib = sum(int(r.n_frames) * int(r.channels) * int(r.bits) // 8 for r in need if str(r.source).startswith(DRIVE) or r.kind != "dir") / 2**30
    require_space(CFG["local_audio"], gib, f"{what} audio", 3.0)
    log(f"  [{what}] copying {len(need):,} WAV files ({gib:.1f} GiB) to local disk ..."); t0 = time.time(); local = threading.local()
    def cp(r):
        dst = local_wav(r.uid); dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_name(dst.name + ".part")
        if r.kind == "dir":
            if not str(r.source).startswith(DRIVE):
                try: os.link(r.source, dst); return 1                       # locally extracted archive: hard link, no second copy
                except OSError: pass
            shutil.copyfile(r.source, tmp)
        else:
            if not hasattr(local, "z"): local.z = {}
            if r.source not in local.z: local.z[r.source] = zipfile.ZipFile(r.source)
            with local.z[r.source].open(r.member) as fi, open(tmp, "wb") as fo: shutil.copyfileobj(fi, fo, 1 << 20)
        os.replace(tmp, dst); return 1
    done = 0
    with ThreadPoolExecutor(16) as ex:
        for _ in ex.map(cp, need, chunksize=32):
            done += 1
            if done % 5000 == 0: log(f"    {done:,}/{len(need):,} | {time.time() - t0:.0f} s")
    log(f"  [{what}] {len(need):,} files copied in {time.time() - t0:.0f} s")

def decode_check(df, what):
    """Full decode of each file ONCE (finite samples, length = header). Failures are recorded and excluded with their reason."""
    import soundfile as sf
    P = paths(); fv, ff = P["meta"] / "decode_verified.txt", P["meta"] / "decode_failures.csv"
    ver = set(open(fv).read().split()) if fv.exists() else set(); bad = pd.read_csv(ff, dtype=str).set_index("uid").reason.to_dict() if ff.exists() else {}
    todo = [r for r in df.itertuples() if r.uid not in ver and r.uid not in bad]
    def chk(r):
        try:
            x, sr = sf.read(str(local_wav(r.uid)), dtype="float32", always_2d=True)
            if sr != int(r.sr) or len(x) != int(r.n_frames): return r.uid, f"decoded {len(x)} frames @ {sr} Hz, header {r.n_frames} @ {r.sr}"
            if not np.isfinite(x).all(): return r.uid, "non-finite samples"
            if float(np.abs(x).max()) == 0.0: return r.uid, "all-zero audio"
            return r.uid, ""
        except Exception as e: return r.uid, f"decode error: {type(e).__name__}: {e}"
    t0 = time.time(); new_ok = []
    with ThreadPoolExecutor(16) as ex:
        for i, (u, why) in enumerate(ex.map(chk, todo, chunksize=32)):
            if why: bad[u] = why
            else: new_ok.append(u)
            if (i + 1) % 20000 == 0: log(f"    {i + 1:,}/{len(todo):,} decoded | {time.time() - t0:.0f} s")
    if new_ok:
        with open(fv, "a") as f: f.write("\n".join(new_ok) + "\n")
    atomic_csv(pd.DataFrame(sorted(bad.items()), columns=["uid", "reason"]), ff)
    fails = df.uid[df.uid.isin(set(bad))].tolist()
    log(f"  [{what}] full decode check: {len(todo):,} decoded now ({len(df) - len(todo):,} earlier) in {time.time() - t0:.0f} s | failures {len(fails)}"
        + (f" -> excluded with reason, e.g. {fails[:2]}" if fails else ""))
    return df[~df.uid.isin(set(bad))].reset_index(drop=True)

def load_wave(uid, crop_seconds=None):
    """16 kHz mono float32. crop_seconds: centred crop (the SAME segment for teacher and student)."""
    import soundfile as sf
    x, sr = sf.read(str(local_wav(uid)), dtype="float32", always_2d=True); x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != SR:
        import torch, torchaudio; x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    if crop_seconds:
        L = int(round(crop_seconds * SR))
        if len(x) > L: s = (len(x) - L) // 2; x = x[s:s + L]
    return np.ascontiguousarray(x, dtype=np.float32)

def crop_len(n_frames, sr, crop_seconds):
    n = int(round(int(n_frames) * SR / int(sr))) if int(sr) != SR else int(n_frames)
    return min(n, int(round(crop_seconds * SR))) if crop_seconds else n
# ####################################################################################################
# ---- scoring: cosine trials and EER (pure numpy)
# ####################################################################################################
def compute_eer(scores, labels):
    """EER (fraction) and threshold. FRR(t) = targets with score <= t, FAR(t) = non-targets with score > t; linear interpolation at the crossing."""
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(labels).astype(bool); nt, nn_ = int(y.sum()), int((~y).sum())
    if nt == 0 or nn_ == 0: raise ValueError("EER needs target and non-target trials")
    o = np.argsort(s, kind="mergesort"); s, y = s[o], y[o]
    frr = np.concatenate([[0.0], np.cumsum(y) / nt]); far = np.concatenate([[1.0], 1.0 - np.cumsum(~y) / nn_]); d = frr - far
    k = int(np.argmax(d >= 0))
    if k == 0: return float(frr[0]), float(s[0])
    a = d[k - 1] / (d[k - 1] - d[k]); eer = frr[k - 1] + a * (frr[k] - frr[k - 1])
    return float(eer), float(s[k - 1])

def l2n(x, eps=1e-12): x = np.asarray(x, dtype=np.float64); return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)

def score_trials(emb, row_of, trials):
    """Cosine scores for the trials whose two utterances both have an embedding; returns (scores, labels, n_missing)."""
    keep = trials.enrol.map(lambda u: u in row_of) & trials.test.map(lambda u: u in row_of); t = trials[keep]
    e = l2n(emb); a = e[t.enrol.map(row_of).values]; b = e[t.test.map(row_of).values]
    return (a * b).sum(-1), t.label.values, int((~keep).sum())

def make_selection_trials(df, n_trials, seed):
    """NON-OFFICIAL trials among held-out DEV speakers (for choosing the target normalisation and for monitoring only): half target, half
    non-target pairs, sampled deterministically, no duplicates, no self-pairs."""
    rng = np.random.default_rng(seed); uids = df.uid.values; spk = df.speaker.values; by = {}
    for i, s in enumerate(spk): by.setdefault(s, []).append(i)
    S = [s for s in sorted(by) if len(by[s]) >= 2]; seen = set(); rows = []
    tries = 0
    while len(rows) < n_trials and tries < 50 * n_trials:
        tries += 1; tgt = len(rows) % 2 == 0
        if tgt: s = S[rng.integers(len(S))]; i, j = rng.choice(by[s], 2, replace=False)
        else:
            i, j = rng.integers(len(uids), size=2)
            if spk[i] == spk[j]: continue
        key = (min(i, j), max(i, j))
        if key in seen or i == j: continue
        seen.add(key); rows.append((int(tgt), uids[i], uids[j]))
    return pd.DataFrame(rows, columns=["label", "enrol", "test"])

def selection_sets(val_df):
    rng = np.random.default_rng(CFG["seed"] + 2); k = CFG["selection_val_utts_per_speaker"]
    parts = [g.iloc[np.sort(rng.permutation(len(g))[:k])] for _, g in val_df.groupby("speaker", sort=True)]
    sel = pd.concat(parts).sort_values("uid").reset_index(drop=True)
    return sel, make_selection_trials(sel, CFG["selection_trials"], CFG["seed"] + 3)

# ####################################################################################################
# ---- ReDimNet2-B6 teacher (frozen, pinned code commit + sha256-checked weights)
# ####################################################################################################
NORMS = ("none", "centre", "standardise")

def spec_name(spec):
    return f"{CFG['redimnet_label']} speaker embedding, " + dict(none="raw (no normalisation)", centre="mean-centred (training-set mean)",
                                                             standardise="standardised (training-set mean / std per dimension)")[spec["norm"]]

def norm_stats(E_tr, norm):
    """Label-free target normalisation from TRAINING-partition embeddings: 'none' -> (0, 1), 'centre' -> (mean, 1), 'standardise' -> (mean, std)."""
    D = E_tr.shape[1]
    if norm == "none": return np.zeros(D, np.float32), np.ones(D, np.float32)
    mu = E_tr.mean(0).astype(np.float32)
    if norm == "centre": return mu, np.ones(D, np.float32)
    sd = E_tr.std(0); return mu, np.maximum(sd, 1e-5 * max(float(sd.mean()), 1e-12)).astype(np.float32)

def _git(*a, cwd=None): return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

class ReDimNetTeacher:
    """Frozen ReDimNet2 (PalabraAI/redimnet2, MIT licence): source code pinned to CFG['redimnet_commit'], weights from the repository's release,
    checked against CFG['redimnet_weights_sha256']. Input: 16 kHz mono waveform (B, samples); output: (B, embed_dim) speaker embedding.
    The model has no padding mask (its signal normalisation, mel mean normalisation and pooling run over the whole input), so a batch only
    ever contains utterances of IDENTICAL length (checked)."""
    use_mask = False

    def __init__(self):
        import torch
        repo = Path(assert_local(CFG["redimnet_dir"])); want = CFG["redimnet_commit"]
        ok = (repo / ".git").exists()
        if ok:
            try: ok = _git("rev-parse", "HEAD", cwd=repo) == want and not _git("status", "--porcelain", cwd=repo)
            except Exception: ok = False
        if not ok:
            shutil.rmtree(repo, ignore_errors=True); log(f"  cloning {CFG['redimnet_repo']} @ {want[:12]} -> {repo}")
            _git("clone", "--quiet", CFG["redimnet_repo"], str(repo)); _git("checkout", "--quiet", want, cwd=repo)
        head = _git("rev-parse", "HEAD", cwd=repo)
        if head != want: raise RuntimeError(f"ReDimNet2 source is at {head}, expected the pinned commit {want}")
        if str(repo) not in sys.path: sys.path.insert(0, str(repo))
        for m in [m for m in sys.modules if m == "redimnet2" or m.startswith("redimnet2.")]: del sys.modules[m]
        import redimnet2 as R2
        if not Path(R2.__file__).resolve().is_relative_to(repo.resolve()): raise RuntimeError(f"imported redimnet2 from {R2.__file__}, not from the pinned clone {repo}")
        wname = f"{CFG['redimnet_model']}-{CFG['redimnet_dataset']}-{CFG['redimnet_train_type']}.pt"; url = R2.URL_TEMPLATE.format(weight_name=wname)
        wdir = Path(assert_local(CFG["torch_home"])) / "redimnet2"; wdir.mkdir(parents=True, exist_ok=True); wf = wdir / wname
        exp = CFG.get("redimnet_weights_sha256")
        if wf.exists() and exp and file_sha256(wf) != exp: log(f"  {wf.name}: sha256 mismatch (damaged download?) -> downloaded again"); wf.unlink()
        if not wf.exists():
            log(f"  downloading {url}"); tmp = wf.with_name(wf.name + ".part"); torch.hub.download_url_to_file(url, str(tmp), progress=False); os.replace(tmp, wf)
        sha = file_sha256(wf)
        if exp and sha != exp: raise RuntimeError(f"{wname}: sha256 {sha} != expected {exp} (the released file changed?). Training blocked.")
        try: ck = torch.load(wf, map_location="cpu", weights_only=True)
        except Exception: ck = torch.load(wf, map_location="cpu", weights_only=False)
        self.model = R2.ReDimNet2Wrap(**ck["model_config"]); res = self.model.load_state_dict(ck["state_dict"], strict=True)
        if res.missing_keys or res.unexpected_keys: raise RuntimeError(f"{wname}: missing {res.missing_keys[:3]}, unexpected {res.unexpected_keys[:3]}")
        self.model.eval().to("cuda")
        for p in self.model.parameters(): p.requires_grad_(False)
        self.amp = None; self._cpu = None; D = int(self.model.embed_dim)
        self.info = dict(model_id=f"PalabraAI/redimnet2:{wname[:-3]}", name=CFG["redimnet_label"], repo=CFG["redimnet_repo"], commit=head, model_class=type(self.model).__name__,
                         model_config=json.loads(json.dumps(ck["model_config"], default=str)), embed_dim=D, hidden_size=D,
                         params=int(sum(p.numel() for p in self.model.parameters())), sampling_rate=SR, attention_mask_used=False,
                         weight_files={wname: dict(bytes=os.path.getsize(wf), sha256=sha, url=url)},
                         training_data=dict(vox2="VoxCeleb2-dev (no speaker overlap with VoxCeleb1)").get(CFG["redimnet_dataset"], CFG["redimnet_dataset"]),
                         train_type=CFG["redimnet_train_type"], licence="MIT",
                         input_format="16 kHz mono float waveform (B, samples), identical lengths within a batch -> the model's own front end "
                                      "(signal normalisation, pre-emphasis, 72 log-mel bands, mean normalisation) -> embedding")
        del ck
        log(f"  {self.info['name']} ({wname}) code @ {head[:12]} | {self.info['params'] / 1e6:.2f}M params | {D}-D embedding | weights sha256 {sha[:16]}... | "
            f"trained on {self.info['training_data']}")

    def embed(self, waves, amp="auto"):
        """-> (B, D) fp32 embeddings. All waves must have the same length."""
        import torch
        if len({len(w) for w in waves}) != 1: raise RuntimeError("ReDimNet2 has no padding mask: one batch = utterances of identical length")
        amp = self.amp if amp == "auto" else amp; x = torch.from_numpy(np.stack(waves).astype(np.float32)).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=amp or torch.float32, enabled=amp is not None): e = self.model(x)
        return e.float()

    def embed_cpu(self, waves):
        """Fallback for a single utterance that does not fit in GPU memory: same weights, fp32 on the CPU."""
        import torch, copy
        if self._cpu is None: self._cpu = copy.deepcopy(self.model).cpu().float().eval()
        with torch.no_grad(): return self._cpu(torch.from_numpy(np.stack(waves).astype(np.float32))).float()

    def smoke_test(self):
        import torch
        rng = np.random.default_rng(0); w = (0.1 * rng.standard_normal(4 * SR)).astype(np.float32); e = self.embed([w, w], amp=None)
        assert e.shape == (2, self.info["embed_dim"]) and bool(torch.isfinite(e).all()) and float((e[0] - e[1]).abs().max()) < 1e-4, tuple(e.shape)
        assert not self.model.training and sum(p.requires_grad for p in self.model.parameters()) == 0
        log(f"  smoke test on 4 s of noise: output {tuple(e.shape)}, finite, batch rows identical, eval mode, frozen -> OK")

def equal_length_batches(lens, max_sec, one=False):
    """Batches of utterances with IDENTICAL sample counts (long utterances cropped to crop_seconds all share one length); <= max_sec of audio."""
    by = {}
    for i, l in enumerate(lens): by.setdefault(int(l), []).append(int(i))
    out = []
    for l in sorted(by):
        per = 1 if one else max(1, int(max_sec * SR // max(l, 1))); idx = by[l]
        out += [idx[s:s + per] for s in range(0, len(idx), per)]
    return out

def run_teacher(T, df, crop, desc, max_sec=None):
    """Teacher embeddings for every utterance of df (float32 array in df order). Audio of the next batch is loaded while the GPU works;
    GPU out-of-memory -> smaller batch, and a single utterance that still does not fit runs on the CPU (same weights, fp32)."""
    import torch
    max_sec = max_sec or CFG["teacher_batch_seconds"]; lens = np.array([crop_len(r.n_frames, r.sr, crop) for r in df.itertuples()]); uids = df.uid.tolist()
    out = [None] * len(df); todo = equal_length_batches(lens, max_sec); t0 = t_log = time.time(); done = 0; n_cpu = 0
    ex = ThreadPoolExecutor(8); load = lambda b: ex.submit(lambda: [load_wave(uids[i], crop) for i in b])
    k = 0; fut = load(todo[0]) if todo else None
    while k < len(todo):
        b = todo[k]; ws = fut.result(); fut = load(todo[k + 1]) if k + 1 < len(todo) else None
        if len({len(w) for w in ws}) > 1:                # decoded length != header (cannot happen after decode_check): one by one
            todo[k:k + 1] = [[i] for i in b]; fut = load(todo[k]); continue
        try: y = T.embed(ws).cpu().numpy()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(b) > 1:
                h = len(b) // 2; todo[k:k + 1] = [b[:h], b[h:]]; fut = load(todo[k]); continue
            y = T.embed_cpu(ws).numpy(); n_cpu += 1; log(f"    [{desc}] {uids[b[0]]} ({lens[b[0]] / SR:.0f} s) computed on the CPU (GPU memory)")
        for j, i in enumerate(b): out[i] = y[j]
        done += len(b); k += 1
        if time.time() - t_log > 60 or k == len(todo):
            el = time.time() - t0; t_log = time.time()
            log(f"    [{desc}] {done:,}/{len(df):,} utterances | {el / 60:.1f} min | ETA {el / max(done, 1) * (len(df) - done) / 60:.1f} min" + (f" | {n_cpu} on CPU" if n_cpu else ""))
    ex.shutdown(wait=False)
    return np.stack(out).astype(np.float32) if out else np.zeros((0, T.info["embed_dim"]), np.float32)

def verify_teacher(T, df, prefer_amp=None):
    """Teacher checks on REAL audio: output shape, finite values, repeatability, distinct utterances, identical-length batch == single,
    speaker separation (same-speaker vs different-speaker cosine), mixed precision vs fp32, frozen state."""
    import torch
    R = {}; D = T.info["embed_dim"]; crop = CFG["crop_seconds"]
    rows = df.sort_values("duration_s").iloc[np.linspace(0, len(df) - 1, 6).astype(int)]; ws = [load_wave(u, crop) for u in rows.uid]
    a = torch.stack([T.embed([w], amp=None)[0] for w in ws]); b = torch.stack([T.embed([w], amp=None)[0] for w in ws])
    R["output_shape"] = [1, int(a.shape[1])]; assert a.shape[1] == D
    R["repeat_max_abs_diff"] = float((a - b).abs().max()); R["finite"] = bool(torch.isfinite(a).all())
    En = torch.nn.functional.normalize(a, dim=-1); Cm = En @ En.T; off = Cm[~torch.eye(len(ws), dtype=torch.bool, device=Cm.device)]
    R["pairwise_cos_raw"] = dict(mean=float(off.mean()), max=float(off.max())); R["distinct_utterances"] = bool(off.max() < 0.99999)
    R["norm_raw"] = dict(mean=float(a.norm(dim=-1).mean()), min=float(a.norm(dim=-1).min()), max=float(a.norm(dim=-1).max()))
    w0 = ws[0]; w1 = ws[0][::-1].copy(); eb = T.embed([w0, w1], amp=None); e1 = T.embed([w1], amp=None)[0]
    R["equal_length_batch_vs_single_min_cos"] = float(min(torch.nn.functional.cosine_similarity(eb[0], a[0], dim=0), torch.nn.functional.cosine_similarity(eb[1], e1, dim=0)))
    # speaker separation on real training utterances (a trained speaker model must separate speakers clearly)
    g = df.groupby("speaker"); spk = [s for s, x in sorted(g, key=lambda t: t[0]) if len(x) >= 3][:8]
    su = pd.concat([g.get_group(s).sort_values("uid").iloc[:3] for s in spk]); se = torch.stack([T.embed([load_wave(u, crop)], amp=None)[0] for u in su.uid])
    sn = torch.nn.functional.normalize(se, dim=-1); sc = (sn @ sn.T).cpu().numpy(); lab = su.speaker.values
    iu = np.triu_indices(len(su), 1); same = lab[iu[0]] == lab[iu[1]]
    R["speaker_check"] = dict(speakers=len(spk), utterances=len(su), same_speaker_cos=float(sc[iu][same].mean()), different_speaker_cos=float(sc[iu][~same].mean()))
    R["speaker_check"]["gap"] = R["speaker_check"]["same_speaker_cos"] - R["speaker_check"]["different_speaker_cos"]
    # mixed precision: float16 (every GPU) unless a previous run of this teacher recorded float32
    cands = [prefer_amp] if prefer_amp else [CFG["teacher_amp_dtype"]]; T.amp = None
    for nm in cands:
        if nm in (None, "float32"): break
        dt = getattr(torch, nm); c = [float(torch.nn.functional.cosine_similarity(T.embed([w], amp=dt)[0], a[i], dim=0)) for i, w in enumerate(ws)]
        R["amp_vs_fp32_min_cos"] = min(c); R["amp_tested"] = nm
        if min(c) >= 0.999: T.amp = dt
    R["amp_used"] = str(T.amp).replace("torch.", "") if T.amp else "float32"
    R["frozen"] = dict(eval_mode=not T.model.training, trainable_params=sum(p.numel() for p in T.model.parameters() if p.requires_grad))
    ok = (R["repeat_max_abs_diff"] < 1e-3 and R["finite"] and R["distinct_utterances"] and R["frozen"]["trainable_params"] == 0 and R["frozen"]["eval_mode"]
          and R["equal_length_batch_vs_single_min_cos"] > 0.9999 and R["speaker_check"]["gap"] >= CFG["teacher_min_speaker_gap"])
    R["passed"] = bool(ok)
    log(f"  output (1, {D}) | repeat max|diff| {R['repeat_max_abs_diff']:.1e} | finite {R['finite']} | raw pairwise cos mean {R['pairwise_cos_raw']['mean']:.3f} "
        f"(distinct {R['distinct_utterances']}) | equal-length batch vs single min cos {R['equal_length_batch_vs_single_min_cos']:.6f}\n"
        f"  speaker check ({len(spk)} training speakers x 3 utterances): same-speaker cos {R['speaker_check']['same_speaker_cos']:.3f} vs different-speaker "
        f"{R['speaker_check']['different_speaker_cos']:.3f} (gap {R['speaker_check']['gap']:.3f}, needs >= {CFG['teacher_min_speaker_gap']})\n"
        f"  mixed precision vs fp32 min cos {R.get('amp_vs_fp32_min_cos', 'n/a')} -> using {R['amp_used']} | frozen {R['frozen']} -> {'PASS' if ok else 'FAIL'}")
    return R

def select_normalisation(E_tr, E_dv, dv):
    """Target normalisation (none / centre / standardise) chosen by EER on NON-OFFICIAL trials among held-out DEV speakers; statistics from the
    TRAINING partition only. The official test set is never used here. Ties (within 0.01 % EER) -> the simpler option."""
    sel, trials = selection_sets(dv); atomic_csv(trials, paths()["meta"] / "selection_trials_NONOFFICIAL.csv")
    rd = {u: i for i, u in enumerate(dv.uid)}; Es = E_dv[[rd[u] for u in sel.uid]]; row = {u: i for i, u in enumerate(sel.uid)}; res = []
    for n in NORMS:
        mu, sd = norm_stats(E_tr, n); s, y, miss = score_trials((Es - mu) / sd, row, trials); assert miss == 0
        res.append(dict(norm=n, eer=compute_eer(s, y)[0]))
    best = sorted(res, key=lambda r: (round(r["eer"], 4), NORMS.index(r["norm"])))[0]; spec = dict(kind="embedding", norm=best["norm"])
    log(f"  selection data: {len(E_tr):,} training utterances (statistics) | {len(sel):,} utterances of {sel.speaker.nunique()} held-out dev speakers | "
        f"{len(trials):,} NON-OFFICIAL trials ({int(trials.label.sum()):,} target)\n  EER per normalisation: " + " | ".join(f"{r['norm']} {r['eer'] * 100:.3f}%" for r in res)
        + f"\n  CHOSEN: {spec_name(spec)} (selection EER {best['eer'] * 100:.3f}%)")
    return dict(teacher=CFG["redimnet_label"], commit=None, chosen=dict(spec=spec, name=spec_name(spec), eer=best["eer"], dim=int(E_tr.shape[1])), candidates=res,
                selection_utterances=len(sel), selection_speakers=int(sel.speaker.nunique()), selection_trials=len(trials), training_utterances_for_statistics=len(E_tr),
                rule="lowest EER (cosine) on non-official trials among held-out dev speakers; ties -> simpler normalisation")

# ------------------------------------------------------------------ utterance-level teacher embedding cache (small: one vector per utterance)
CACHE_VERSION = 2
SHARD = 512

def teacher_identity(T):
    """Everything that determines the teacher's outputs (saved as teacher_info.json; caches are keyed by it)."""
    return json.loads(json.dumps(dict(T.info, amp=str(T.amp).replace("torch.", "") if T.amp else "float32"), default=str))

def cache_identity(tid, spec, crop):
    """The cache stores RAW embeddings, so it does not depend on the normalisation (spec['norm'] is applied when loading)."""
    d = dict(cache_version=CACHE_VERSION, model_id=tid["model_id"], commit=tid["commit"], weights={k: v["sha256"] for k, v in tid["weight_files"].items()},
             model_config=sha_of([json.dumps(tid["model_config"], sort_keys=True)]), crop_seconds=crop, crop_rule="centred crop" if crop else "full utterance",
             sampling_rate=SR, amp=tid["amp"], dim=tid["embed_dim"], dtype="float16", stored="RAW teacher embeddings (normalisation is applied when loading)")
    return sha_of([json.dumps(d, sort_keys=True)]), d

class EmbCache:
    """teacher_embedding_cache/<id>/ : meta.json + shard_XXXXX.npz (uids, emb fp16) + shard_XXXXX.json (sha1, count) written LAST = commit.
    Local mirror in /content; every committed shard is also copied to Drive (small files), so a recycled runtime does not recompute them."""
    def __init__(self, tid, spec, crop, drive_top=None):
        self.tid, self.spec, self.crop, self.drive = tid, spec, crop, drive_top; self.cid, self.meta = cache_identity(tid, spec, crop)
        self.local = paths()["teacher"] / "cache" / self.cid; self.local.mkdir(parents=True, exist_ok=True); self.rel = f"teacher_embedding_cache/{self.cid}"
        mf = self.local / "meta.json"
        if self.drive and self.drive.exists(f"{self.rel}/meta.json"):
            dm = json.load(open(self.drive.path(f"{self.rel}/meta.json")))
            if dm != json.loads(json.dumps(self.meta)): raise RuntimeError(f"Drive cache {self.rel}/meta.json does not match this configuration (stale or damaged): not reused")
        if mf.exists() and json.load(open(mf)) != json.loads(json.dumps(self.meta)): raise RuntimeError(f"local cache {mf} does not match this configuration")
        atomic_json(self.meta, mf)
        if self.drive and not self.drive.exists(f"{self.rel}/meta.json"): self.drive.write_json(f"{self.rel}/meta.json", self.meta)
        self._sync_from_drive()

    def _sync_from_drive(self):
        if not self.drive: return
        d = self.drive.dir / self.rel
        if not d.is_dir(): return
        n = 0
        for j in sorted(d.glob("shard_*.json")):
            if not (self.local / j.name).exists():
                npz = d / j.name.replace(".json", ".npz")
                if npz.exists() and file_sha1(npz) == json.load(open(j))["sha1"]:
                    shutil.copyfile(npz, self.local / npz.name); shutil.copyfile(j, self.local / j.name); n += 1
        if n: log(f"  teacher cache {self.cid}: {n} committed shards restored from Drive")

    def shards(self):
        out = []
        for j in sorted(self.local.glob("shard_*.json")):
            m = json.load(open(j)); npz = self.local / j.name.replace(".json", ".npz")
            if not npz.exists() or file_sha1(npz) != m["sha1"]: log(f"  damaged shard {npz.name}: moved aside"); (self.local / "corrupt").mkdir(exist_ok=True); os.replace(j, self.local / "corrupt" / j.name); continue
            out.append((npz, m))
        return out

    def done_uids(self):
        s = set()
        for npz, m in self.shards(): s.update(np.load(npz, allow_pickle=False)["uids"].tolist())
        return s

    def build(self, df, desc, T):
        if teacher_identity(T) != self.tid: raise RuntimeError("the loaded teacher differs from the identity this cache was opened with")
        have = self.done_uids(); todo = df[~df.uid.isin(have)].reset_index(drop=True)
        log(f"  [{desc}] teacher cache {self.cid} ({'centred crop <= ' + str(self.crop) + ' s' if self.crop else 'full utterances'}): "
            f"{len(df) - len(todo):,} done, {len(todo):,} to compute (committed every {SHARD * 8:,} utterances; an interruption loses at most that block)")
        k0 = len(list(self.local.glob("shard_*.json"))) + len(list((self.local / "corrupt").glob("*.json")) if (self.local / "corrupt").exists() else [])
        for s in range(0, len(todo), SHARD * 8):
            part = todo.iloc[s:s + SHARD * 8].reset_index(drop=True)
            E = run_teacher(T, part, self.crop, f"{desc}, block {s // (SHARD * 8) + 1}/{-(-len(todo) // (SHARD * 8))}")
            if not np.isfinite(E).all() or (np.linalg.norm(E, axis=1) == 0).any(): raise FloatingPointError(f"{desc}: invalid teacher embeddings")
            for q in range(0, len(part), SHARD):
                name = f"shard_{k0:05d}"; k0 += 1; npz = self.local / f"{name}.npz"; tmp = self.local / f"{name}.part.npz"
                np.savez(tmp, uids=np.array(part.uid[q:q + SHARD].tolist()), emb=E[q:q + SHARD].astype(np.float16)); os.replace(tmp, npz)
                m = dict(sha1=file_sha1(npz), count=int(min(SHARD, len(part) - q))); atomic_json(m, self.local / f"{name}.json")
                if self.drive: self.drive.write_file(npz, f"{self.rel}/{name}.npz"); self.drive.write_json(f"{self.rel}/{name}.json", m)
        return self.load(df)

    def complete(self, df): return set(df.uid) <= self.done_uids()
    def load(self, df):
        """-> float32 matrix aligned to df (exactly one valid embedding per utterance is required)."""
        rows, seen = {}, {}
        for npz, m in self.shards():
            z = np.load(npz, allow_pickle=False); u = z["uids"].tolist(); e = z["emb"]
            assert len(u) == m["count"] == len(e), f"{npz.name}: count mismatch"
            for i, x in enumerate(u):
                if x in seen: raise RuntimeError(f"teacher cache: {x} appears twice ({seen[x]}, {npz.name})")
                seen[x] = npz.name; rows[x] = e[i]
        miss = [u for u in df.uid if u not in rows]
        if miss: raise RuntimeError(f"teacher cache {self.cid}: {len(miss)} utterances missing (e.g. {miss[:3]}): run the cache step")
        E = np.stack([rows[u] for u in df.uid]).astype(np.float32) if len(df) else np.zeros((0, self.tid["embed_dim"]), np.float32)
        if not np.isfinite(E).all() or (np.linalg.norm(E, axis=1) == 0).any(): raise FloatingPointError("teacher cache contains invalid embeddings")
        return E
# ####################################################################################################
# ---- student: Zipformer-M source (icefall, unchanged) + pure-PyTorch Swoosh port (k2 replacement) + SV branch
# ####################################################################################################
# ------------------------------------------------------------------ Swoosh: pure-PyTorch port of k2's implementation (k2/python/csrc/torch/v2/autograd/swoosh.cu)
K2_SWOOSH_SRC = r'''# Pure-PyTorch port of k2's Swoosh functions (k2/python/csrc/torch/v2/autograd/swoosh.cu), written by the MT2KD ASR Stage 1 pipeline.
# Same formulas, same fp32 compute, same 8-bit stochastically-rounded derivative in the autograd path. Differences from k2:
# the random numbers for the stochastic rounding come from torch's RNG (k2: its own torch::rand call) -> statistically identical, not bitwise.
import torch
_P = {"l": (4.0, 0.08, 0.035), "r": (1.0, 0.08, 0.313261687)}

def _forward(x, shift, coeff, offset):                    # k2 SwooshForward
    x = x.to(torch.float32); xo = x - shift
    l = torch.log1p(torch.exp(xo)); l = torch.where(torch.isinf(l), xo, l)
    return l - coeff * x - offset

def _forward_and_deriv(x, shift, coeff, offset):          # k2 SwooshForwardAndDeriv
    x = x.to(torch.float32); xo = x - shift
    denom = 1.0 + torch.exp(xo); d = (1.0 - coeff) - 1.0 / denom
    ld = torch.log(denom); ld = torch.where(torch.isinf(ld), xo, ld)
    return ld - coeff * x - offset, d

class _SwooshFunction(torch.autograd.Function):           # k2 SwooshFunction (autograd path)
    @staticmethod
    def forward(ctx, x, shift, coeff, offset, dropout_prob):
        x = x.to(torch.float32).contiguous(); xo = x - shift; e = torch.exp(xo)
        g = 1.0 - 1.0 / (1.0 + e)
        l = torch.log1p(e); l = torch.where(torch.isinf(l), xo, l); y = l - coeff * x - offset
        if dropout_prob != 0.0:
            drop = torch.rand_like(x) < dropout_prob
            y = torch.where(drop, torch.zeros_like(y), y * (1.0 / (1.0 - dropout_prob))); g = torch.where(drop, torch.full_like(g, coeff), g)
        g_int = (g * (255.0 / 1.005) + torch.rand_like(x)).to(torch.uint8)
        ctx.save_for_backward(g_int); ctx.coeff = coeff; ctx.dropout_prob = dropout_prob
        return y
    @staticmethod
    def backward(ctx, y_grad):
        (g,) = ctx.saved_tensors
        f = (g.to(torch.float32) * (1.005 / 255.0) - ctx.coeff) / (1.0 - ctx.dropout_prob)
        return y_grad.to(torch.float32) * f, None, None, None, None

def _swoosh(x, which, dropout_prob):
    shift, coeff, offset = _P[which]
    if not x.requires_grad:                               # k2: logaddexp path when no gradient is needed
        x = x.to(torch.float32); return torch.logaddexp(torch.zeros(1, device=x.device), x - shift) - coeff * x - offset
    return _SwooshFunction.apply(x, shift, coeff, offset, float(dropout_prob))

def swoosh_l(x, dropout_prob=0.0): return _swoosh(x, "l", dropout_prob)
def swoosh_r(x, dropout_prob=0.0): return _swoosh(x, "r", dropout_prob)
def swoosh_l_forward(x): return _forward(x, *_P["l"])
def swoosh_r_forward(x): return _forward(x, *_P["r"])
def swoosh_l_forward_and_deriv(x): return _forward_and_deriv(x, *_P["l"])
def swoosh_r_forward_and_deriv(x): return _forward_and_deriv(x, *_P["r"])
'''
STUB_FUNCS = {"swoosh_l", "swoosh_r", "swoosh_l_forward", "swoosh_r_forward", "swoosh_l_forward_and_deriv", "swoosh_r_forward_and_deriv"}

def setup_zipformer():
    """Private copy of icefall's Zipformer source (unchanged except `import k2` -> the Swoosh port). Returns source information."""
    ZDIR = CFG["zipformer_dir"]; SRC_Z = "/content/icefall/egs/librispeech/ASR/zipformer"
    if not os.path.isdir(SRC_Z):
        subprocess.run("rm -rf /content/icefall; git clone --depth 1 https://github.com/k2-fsa/icefall.git /content/icefall", shell=True, check=True)
    os.makedirs(ZDIR + "/icefall", exist_ok=True)
    for f in ["zipformer.py", "scaling.py", "subsampling.py", "encoder_interface.py", "optim.py"]:
        if not os.path.exists(f"{ZDIR}/{f}"): shutil.copy(f"{SRC_Z}/{f}", f"{ZDIR}/{f}")
    o_ = open(f"{ZDIR}/optim.py").read()                 # lhotse is only used by optim.py's self-tests: replaced by an equivalent seed function
    if "from lhotse.utils import fix_random_seed" in o_:
        o_ = o_.replace("from lhotse.utils import fix_random_seed",
                        "def fix_random_seed(s):\n    import random as _r, numpy as _n\n    _r.seed(s); _n.random.seed(s); torch.manual_seed(s)", 1)
        open(f"{ZDIR}/optim.py", "w").write(o_)
    open(ZDIR + "/icefall/__init__.py", "a").close()
    open(ZDIR + "/icefall/utils.py", "w").write(
        'import torch\n\ndef torch_autocast(device_type="cuda", **kwargs):\n    try:\n        return torch.amp.autocast(device_type, **kwargs)\n'
        '    except (AttributeError, TypeError):\n        return torch.cuda.amp.autocast(**kwargs)\n')
    if not os.path.exists(f"{ZDIR}/k2_swoosh.py") or open(f"{ZDIR}/k2_swoosh.py").read() != K2_SWOOSH_SRC: open(f"{ZDIR}/k2_swoosh.py", "w").write(K2_SWOOSH_SRC)
    t_ = open(f"{ZDIR}/scaling.py").read()
    if "\nimport k2\n" in t_: t_ = t_.replace("\nimport k2\n", "\nimport k2_swoosh as k2\n", 1); open(f"{ZDIR}/scaling.py", "w").write(t_)
    used = set(re.findall(r"\bk2\.(\w+)", t_ + open(f"{ZDIR}/zipformer.py").read() + open(f"{ZDIR}/subsampling.py").read()))
    if used - STUB_FUNCS: raise RuntimeError(f"the Zipformer source uses k2 functions the port does not provide: {used - STUB_FUNCS}")
    sys.path = [p for p in sys.path if p != ZDIR]; sys.path.insert(0, ZDIR); importlib.invalidate_caches()
    for m_ in [k for k in list(sys.modules) if k in ("zipformer", "scaling", "subsampling", "encoder_interface", "k2_swoosh", "optim") or k == "icefall" or k.startswith("icefall.")]:
        if not (getattr(sys.modules[m_], "__file__", "") or "").startswith(ZDIR): del sys.modules[m_]          # e.g. imported earlier from another folder
    try: commit = subprocess.run(["git", "-C", "/content/icefall", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception: commit = "unknown"
    return dict(dir=ZDIR, icefall_commit=commit, k2_functions_used=sorted(used),
                files={f: file_sha1(f"{ZDIR}/{f}")[:12] for f in ["zipformer.py", "scaling.py", "subsampling.py", "k2_swoosh.py", "optim.py"]})

def verify_swoosh(device="cuda"):
    """Swoosh port vs reference: forward exactness (fp32/fp16 inputs, extremes), derivative, 8-bit stochastic backward (bounded, unbiased),
    the icefall modules that call it, and the real k2 if it happens to be installed."""
    import torch, torch.nn.functional as F
    setup_zipformer(); import k2_swoosh as P, scaling as S
    dev = torch.device(device if torch.cuda.is_available() else "cpu"); R = dict(source="k2/python/csrc/torch/v2/autograd/swoosh.cu (ported)")
    xs = torch.cat([torch.linspace(-60, 60, 200001), torch.tensor([-65504., -1e4, -100., 100., 1e4, 65504.])]).to(dev)
    for w, (sh, co, of) in {"l": (4.0, 0.08, 0.035), "r": (1.0, 0.08, 0.313261687)}.items():
        x64 = xs.double(); ref = torch.logaddexp(torch.zeros(1, device=dev, dtype=torch.float64), x64 - sh) - co * x64 - of; dref = torch.sigmoid(x64 - sh) - co
        tol = 2e-6 * (1 + ref.abs())
        y = getattr(P, f"swoosh_{w}_forward")(xs); yd, d = getattr(P, f"swoosh_{w}_forward_and_deriv")(xs)
        e_fwd = float(((y.double() - ref).abs() / tol).max()); e_fd = float(((yd.double() - ref).abs() / tol).max()); e_d = float((d.double() - dref).abs().max())
        y16 = getattr(P, f"swoosh_{w}_forward")(xs.half()); y16_ok = bool(torch.isfinite(y16).all()) and y16.dtype == torch.float32
        xg = xs.clone().requires_grad_(True); yg = getattr(P, f"swoosh_{w}")(xg); yg.backward(torch.ones_like(yg)); g = xg.grad.double()
        q = 1.005 / 255.0; e_q = float((g - dref).abs().max()); bias = float((g - dref).mean())
        xr = torch.randn(2_000_000, device=dev) * 6; xr.requires_grad_(True); yr = getattr(P, f"swoosh_{w}")(xr); yr.backward(torch.ones_like(yr))
        bias_r = float((xr.grad.double() - (torch.sigmoid(xr.detach().double() - sh) - co)).mean())
        e_nog = float(((getattr(P, f"swoosh_{w}")(xs).double() - ref).abs() / tol).max())
        mod = (S.SwooshL if w == "l" else S.SwooshR)(); e_mod = float(((mod(xs).double() - ref).abs() / tol).max())
        own = (S.SwooshLForward if w == "l" else S.SwooshRForward)(xs); e_own = float(((own.double() - ref).abs() / tol)[torch.isfinite(own)].max())
        R[w] = dict(forward_err_rel=e_fwd, forward_and_deriv_err_rel=e_fd, deriv_maxerr=e_d, fp16_input_finite_fp32_output=y16_ok, no_grad_path_err_rel=e_nog,
                    backward_maxerr=e_q, backward_quant_step=q, backward_bias_grid=bias, backward_bias_random=bias_r, icefall_module_err_rel=e_mod,
                    icefall_reference_SwooshForward_err_rel=e_own)
        ok = e_fwd <= 1 and e_fd <= 1 and e_d < 1e-6 and y16_ok and e_nog <= 1 and e_q <= q + 1e-6 and abs(bias_r) < 2e-4 and e_mod <= 1 and e_own <= 1
        R[w]["passed"] = ok
        log(f"  swoosh_{w}: forward rel.err {e_fwd:.2f} (<=1 = within 2e-6 relative) | deriv {e_d:.1e} | fp16 inputs finite {y16_ok} | backward |err| {e_q:.4f} "
            f"<= 8-bit step {q:.4f}, mean bias {bias_r:+.1e} | icefall SwooshL/R module {e_mod:.2f}, icefall reference {e_own:.2f} -> {'OK' if ok else 'FAIL'}")
    # the fused ActivationDropoutAndLinear path (uses forward_and_deriv in backward) vs autograd through the reference, dropout 0
    torch.manual_seed(0); lin = S.ActivationDropoutAndLinear(64, 32, bias=True, activation="SwooshL", dropout_p=0.0).to(dev)
    x = (torch.randn(50, 64, device=dev) * 4).requires_grad_(True); y = lin(x); y.square().sum().backward(); gx = x.grad.clone(); gw = lin.weight.grad.clone()
    x2 = x.detach().double().requires_grad_(True); W = lin.weight.detach().double().requires_grad_(True); b = lin.bias.detach().double()
    y2 = F.linear(torch.logaddexp(torch.zeros(1, device=dev, dtype=torch.float64), x2 - 4) - 0.08 * x2 - 0.035, W, b); y2.square().sum().backward()
    R["fused_linear"] = dict(out_rel=float((y.double() - y2).abs().max() / y2.abs().max()), dx_rel=float((gx.double() - x2.grad).abs().max() / x2.grad.abs().max()),
                             dw_rel=float((gw.double() - W.grad).abs().max() / W.grad.abs().max()))
    R["fused_linear"]["passed"] = max(R["fused_linear"].values()) < 1e-4
    log(f"  ActivationDropoutAndLinear(SwooshL) vs float64 autograd: output {R['fused_linear']['out_rel']:.1e}, dx {R['fused_linear']['dx_rel']:.1e}, "
        f"dW {R['fused_linear']['dw_rel']:.1e} (relative)")
    try:
        import k2 as real
        if not hasattr(real, "swoosh_l_forward"): raise ImportError
        xr = torch.randn(100000, device=dev) * 8
        R["real_k2"] = dict(forward_l_maxdiff=float((real.swoosh_l_forward(xr) - P.swoosh_l_forward(xr)).abs().max()),
                            forward_r_maxdiff=float((real.swoosh_r_forward(xr) - P.swoosh_r_forward(xr)).abs().max()),
                            deriv_l_maxdiff=float((real.swoosh_l_forward_and_deriv(xr)[1] - P.swoosh_l_forward_and_deriv(xr)[1]).abs().max()))
        log(f"  compared with the installed k2: {R['real_k2']}")
    except Exception: R["real_k2"] = "k2 not installed for this PyTorch (not downgraded); compared against k2's published formulas and algorithm instead"
    R["passed"] = R["l"]["passed"] and R["r"]["passed"] and R["fused_linear"]["passed"]
    R["claim"] = ("forward and derivative: same formulas as k2 in fp32 (verified to <= 2e-6 relative against float64); autograd backward: same 8-bit "
                  "stochastic quantisation of the derivative as k2 (bounded by one step, unbiased); random numbers differ, so results are not bit-identical to a k2 run")
    return R

def lengths_to_mask(lens, T):
    import torch
    return torch.arange(T, device=lens.device)[None, :] < lens[:, None]

def set_batch_count(model, bc):
    for name, mod in model.named_modules():
        if hasattr(mod, "batch_count"): mod.batch_count = bc
        if hasattr(mod, "name"): mod.name = name

def fbank(wav):
    """Student frontend = your v1 ASR fbank (80 bins, 25 ms / 10 ms, povey window, snip_edges=False)."""
    import torch, torchaudio.compliance.kaldi as kaldi
    return kaldi.fbank(torch.from_numpy(wav)[None], num_mel_bins=80, frame_length=25.0, frame_shift=10.0, dither=0.0, energy_floor=1e-10,
                       sample_frequency=float(SR), snip_edges=False, low_freq=20.0, high_freq=-400.0, remove_dc_offset=True,
                       preemphasis_coefficient=0.97, window_type="povey", use_energy=False)

def make_batches(lens, epoch, seed, shuffle, max_sec, max_utts, idx=None):
    """Duration batches over `idx` (default: all): padded seconds <= max_sec and <= max_utts items; sorted inside shuffled pools of 1000;
    deterministic for (seed, epoch, idx)."""
    lens = np.asarray(lens); idx = np.arange(len(lens)) if idx is None else np.asarray(sorted(idx))
    rng = np.random.default_rng(seed + 1000 * epoch + (0 if len(idx) == len(lens) else 7919 * len(idx)))
    order = idx[rng.permutation(len(idx))] if shuffle else idx; out = []
    for s in range(0, len(order), 1000):
        ids = order[s:s + 1000]; ids = ids[np.argsort(lens[ids], kind="stable")]; cur, mx = [], 0
        for i in ids:
            nm = max(mx, lens[i])
            if cur and (nm * (len(cur) + 1) / SR > max_sec or len(cur) >= max_utts): out.append(cur); cur, nm = [], lens[i]
            cur.append(int(i)); mx = nm
        if cur: out.append(cur)
    if shuffle: out = [out[j] for j in rng.permutation(len(out))]
    return out

# ------------------------------------------------------------------ Zipformer-M student (unchanged architecture) + the MT2KD SV branch
ZIP_ARGS = dict(output_downsampling_factor=2, downsampling_factor=(1, 2, 4, 8, 4, 2), num_encoder_layers=(2, 2, 3, 4, 3, 2),
                encoder_dim=(192, 256, 384, 512, 384, 256), encoder_unmasked_dim=(192, 192, 256, 256, 256, 192), query_head_dim=(32,) * 6,
                pos_head_dim=(4,) * 6, value_head_dim=(12,) * 6, num_heads=(4, 4, 4, 8, 4, 4), feedforward_dim=(512, 768, 1024, 1536, 1024, 768),
                cnn_module_kernel=(31, 31, 15, 15, 15, 31), pos_dim=48, causal=False)        # identical to the project's ASR student
UNUSED_ICEFALL = re.compile(r"(^|\.)layers\.\d+\.bypass_scale$")     # Zipformer2EncoderLayer.bypass_scale: defined, never used by icefall ('# TODO: remove it')
def fbank_frames(n): return (n + 80) // 160
def embed_frames(n): return (fbank_frames(n) - 7) // 2              # Conv2dSubsampling output (~50 Hz) = the rate at which every encoder stack returns its output

def build_sv_student(teacher_dim):
    """-> (student, msv) on CPU. student = the project's Zipformer-M (same parameter names: encoder_embed.*, encoder.*);
    student.sv_features() returns the output of encoder stack `sv_block` (1-based), msv = M_SV."""
    import torch, torch.nn as nn
    info = setup_zipformer()
    from subsampling import Conv2dSubsampling
    from zipformer import Zipformer2
    import zipformer as zmod
    assert zmod.__file__.startswith(CFG["zipformer_dir"]), "Zipformer imported from the wrong folder"
    class _StopAfterBlock(Exception): pass
    class ZipformerStudent(nn.Module):
        def __init__(self):
            super().__init__(); self.encoder_embed = Conv2dSubsampling(in_channels=80, out_channels=192); self.encoder = Zipformer2(**ZIP_ARGS)
            self.sv_index = CFG["sv_block"] - 1; self.sv_dim = int(ZIP_ARGS["encoder_dim"][self.sv_index])
        def forward(self, feats, feat_lens):                       # full encoder (as in the ASR project): (B, T/4, 512) at 25 Hz
            x, xl = self.encoder_embed(feats, feat_lens)
            enc, el = self.encoder(x.permute(1, 0, 2), xl, ~lengths_to_mask(xl, x.size(1)))
            return enc.permute(1, 0, 2), el
        def sv_features(self, feats, feat_lens, early_exit=True):
            """Output of encoder.encoders[sv_index] (the THIRD stack for sv_block=3): (B, T50, 384), valid lengths T50 = embed frames.
            Taken with a forward hook inside icefall's own Zipformer2.forward (no re-implementation); with early_exit the later stacks are not run."""
            x, xl = self.encoder_embed(feats, feat_lens); cap = {}
            def hook(m, i, o):
                cap["y"] = o
                if early_exit: raise _StopAfterBlock()
            h = self.encoder.encoders[self.sv_index].register_forward_hook(hook)
            try: self.encoder(x.permute(1, 0, 2), xl, ~lengths_to_mask(xl, x.size(1)))
            except _StopAfterBlock: pass
            finally: h.remove()
            y = cap["y"].permute(1, 0, 2)
            assert y.size(1) == x.size(1) and y.size(2) == self.sv_dim, (tuple(y.shape), tuple(x.shape))
            return y, xl
        def sv_params(self):
            """Parameters that the SV branch actually uses: the frontend and stacks 1..sv_block."""
            return list(self.encoder_embed.named_parameters(prefix="encoder_embed")) + \
                   [(f"encoder.encoders.{i}.{n}", p) for i in range(self.sv_index + 1) for n, p in self.encoder.encoders[i].named_parameters()]
    class MSV(nn.Module):
        """MT2KD M_SV: 'an attentive pooling module consisting of two 1D convolution layers' + 'a linear layer projecting the speaker embeddings'.
        Here: Conv1d(C->A,1) -> tanh -> Conv1d(A->C,1) -> softmax over VALID frames (channel-wise weights) -> attentive mean and std (2C)
        -> Linear(2C -> teacher dim). Computed in fp32."""
        def __init__(self, C, A, D):
            super().__init__(); self.att = nn.Sequential(nn.Conv1d(C, A, 1), nn.Tanh(), nn.Conv1d(A, C, 1)); self.proj = nn.Linear(2 * C, D)
            self.C, self.A, self.D = C, A, D
        def forward(self, y, lens):
            with torch.autocast("cuda", enabled=False):
                x = y.float().transpose(1, 2); m = lengths_to_mask(lens, x.size(2))[:, None, :]
                e = self.att(x).masked_fill(~m, float("-inf")); w = torch.softmax(e, dim=2)
                mu = (w * x).sum(2); var = (w * x * x).sum(2) - mu * mu
                return self.proj(torch.cat([mu, var.clamp(min=1e-5).sqrt()], 1))
    s = ZipformerStudent(); s.source_info = info
    return s, MSV(s.sv_dim, CFG["msv_attention_channels"], teacher_dim)

def inspect_student(student, msv):
    enc = student.encoder; stacks = []
    for i, m in enumerate(enc.encoders):
        inner = getattr(m, "encoder", m); stacks.append(dict(index=i + 1, type=type(m).__name__, downsample=int(enc.downsampling_factor[i]), layers=len(inner.layers),
                                                             dim=int(enc.encoder_dim[i]), params=sum(p.numel() for p in m.parameters())))
    assert [s["layers"] for s in stacks] == list(ZIP_ARGS["num_encoder_layers"]) and [s["dim"] for s in stacks] == list(ZIP_ARGS["encoder_dim"])
    k = student.sv_index; sv_enc = sum(p.numel() for _, p in student.sv_params()); msv_p = sum(p.numel() for p in msv.parameters())
    R = dict(source=student.source_info, stacks=stacks, total_params=sum(p.numel() for p in student.parameters()), sv_block=k + 1, sv_block_dim=stacks[k]["dim"],
             sv_block_internal_rate_hz=50.0 / stacks[k]["downsample"], sv_block_output_rate_hz=50.0, sv_encoder_params=sv_enc, msv_params=msv_p,
             paper_table_vii_sv_params_block3=18.9e6,
             note=(f"SV branch = Conv2dSubsampling + stacks 1..{k + 1} ({sv_enc / 1e6:.2f}M) + M_SV ({msv_p / 1e6:.2f}M); MT2KD Table VII lists 18.9M SV parameters "
                   f"for the third block with a 192-D teacher (its M_SV projection is smaller)"))
    log(f"  Zipformer-M (icefall {student.source_info['icefall_commit'][:10]}): stacks " + ", ".join(f"#{s['index']} {s['layers']}x{s['dim']}/{s['downsample']}" for s in stacks)
        + f" | {R['total_params'] / 1e6:.2f}M params")
    log(f"  SV branch: stack #{k + 1} = encoder.encoders[{k}] ({stacks[k]['layers']} layers, {stacks[k]['dim']}-D, runs at {R['sv_block_internal_rate_hz']} Hz, "
        f"returned at 50 Hz) | {R['note']}")
    return R

class SVData:
    def __init__(self, df, targets, crop): self.u = df.uid.tolist(); self.t = targets; self.crop = crop
    def __len__(self): return len(self.u)
    def __getitem__(self, i):
        import torch
        w = load_wave(self.u[i], self.crop)
        return dict(i=i, n=len(w), fb=fbank(w), t=None if self.t is None else torch.from_numpy(self.t[i]))

def sv_collate(b):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    return dict(i=torch.tensor([x["i"] for x in b]), n=torch.tensor([x["n"] for x in b]),
                fb=pad_sequence([x["fb"] for x in b], batch_first=True, padding_value=math.log(1e-10)), fb_lens=torch.tensor([x["fb"].size(0) for x in b]),
                t=None if b[0]["t"] is None else torch.stack([x["t"] for x in b]))

def sv_loader(df, targets, batches, crop, num_workers, seed=0):
    import torch
    def wi(_): torch.set_num_threads(1)
    return torch.utils.data.DataLoader(SVData(df, targets, crop), batch_sampler=batches, num_workers=num_workers, collate_fn=sv_collate, pin_memory=True,
                                       worker_init_fn=wi, prefetch_factor=4 if num_workers else None, generator=torch.Generator().manual_seed(int(seed)),
                                       timeout=600 if num_workers else 0)            # a stuck/dead worker raises after 10 min instead of hanging
# ####################################################################################################
# ---- KD objective, evaluation, optimiser, resumable training
# ####################################################################################################
RESUME_KEYS = {"sig", "student", "msv", "opt", "scaler", "scheduler", "state", "rng_cpu", "rng_cuda", "rng_np", "rng_py"}
class SimulatedDisconnect(Exception):
    """Raised by the resume test to imitate a runtime that dies: nothing is saved at that moment."""

def kd_loss(v, t, eps):
    """MT2KD Eq. 8: 1 - CosineSim(teacher, student) between UN-normalised vectors; fp32; eps guards the denominator."""
    v = v.float(); t = t.float(); cos = (v * t).sum(-1) / (v.norm(dim=-1) * t.norm(dim=-1)).clamp(min=eps)
    return (1.0 - cos).mean(), cos

def sv_forward(student, msv, b, dev, amp):
    import torch
    fb, fl = b["fb"].to(dev, non_blocking=True), b["fb_lens"].to(dev)
    with torch.autocast("cuda", dtype=amp): y, yl = student.sv_features(fb, fl)
    return msv(y, yl)

def student_embeddings(student, msv, df, crop, dev, amp, max_sec=600.0):
    """Student SV embeddings (eval mode, no gradient) in df order."""
    import torch
    lens = np.array([crop_len(r.n_frames, r.sr, crop) for r in df.itertuples()]); out = np.zeros((len(df), msv.D), np.float32)
    was = student.training; student.eval(); msv.eval()
    with torch.no_grad():
        for b in sv_loader(df, None, make_batches(lens, 0, 0, False, max_sec, CFG["max_batch_utts"]), crop, CFG["num_workers"]):
            out[b["i"].numpy()] = sv_forward(student, msv, b, dev, amp).float().cpu().numpy()
    student.train(was); msv.train(was); return out

def sv_evaluate(student, msv, C, dev, amp, df=None, targets=None, trials=None):
    """KD metrics on a manifest (no parameter updates) + EER on NON-OFFICIAL validation trials (monitoring only)."""
    import torch
    df = C["dv"] if df is None else df; targets = C["T_dv"] if targets is None else targets; trials = C.get("sel_trials") if trials is None else trials
    E = student_embeddings(student, msv, df, CFG["crop_seconds"], dev, amp); inv = int((~np.isfinite(E).all(1)).sum())
    loss, cos = kd_loss(torch.from_numpy(np.nan_to_num(E)), torch.from_numpy(targets), CFG["cos_eps"])
    tn = l2n(targets); base = float(np.linalg.norm(tn.mean(0)))           # best possible cosine of ONE constant vector = |mean of unit targets|
    r = dict(val_loss=float(loss), val_cos=float(cos.mean()), val_cos_constant_baseline=base, teacher_norm=float(np.linalg.norm(targets, axis=1).mean()),
             student_norm=float(np.linalg.norm(np.nan_to_num(E), axis=1).mean()), invalid_embeddings=inv, val_utts=len(df))
    if trials is not None and len(trials):
        s, y, miss = score_trials(np.nan_to_num(E), {u: i for i, u in enumerate(df.uid)}, trials)
        if len(s) and y.any() and (~y.astype(bool)).any(): r["val_eer_nonofficial"] = compute_eer(s, y)[0]
    return r, E

def make_optimizer(student, msv, cfg):
    """scaled_adam: icefall ScaledAdam (two named-parameter groups: Zipformer frontend+stacks 1..3, M_SV) + Eden; adamw: AdamW (scheduler None)."""
    import torch
    enc = [(n, p) for n, p in student.sv_params() if p.requires_grad]; head = [("msv." + n, p) for n, p in msv.named_parameters()]
    if cfg["optimizer"] == "scaled_adam":
        from optim import ScaledAdam, Eden
        opt = ScaledAdam([dict(named_params=enc, lr=cfg["base_lr"]), dict(named_params=head, lr=cfg["base_lr"] * cfg["msv_lr_scale"])],
                         lr=cfg["base_lr"], clipping_scale=cfg["clipping_scale"])
        return opt, Eden(opt, cfg["lr_batches"], cfg["lr_epochs"], warmup_batches=cfg["eden_warmup_batches"], warmup_start=cfg["eden_warmup_start"])
    if cfg["optimizer"] != "adamw": raise ValueError(cfg["optimizer"])
    ps = [p for _, p in enc]; hp = [p for _, p in head]
    return torch.optim.AdamW([dict(params=[p for p in ps if p.ndim >= 2], weight_decay=cfg["weight_decay"]), dict(params=[p for p in ps if p.ndim < 2], weight_decay=0.0),
                              dict(params=hp, weight_decay=0.0, lr=cfg["adamw_lr"] * cfg["msv_lr_scale"])], lr=cfg["adamw_lr"]), None

def freeze_unused(student):
    """Only the frontend and stacks 1..sv_block are on the SV path; the later stacks keep their initial values and get no optimiser state."""
    used = {n for n, _ in student.sv_params()}
    for n, p in student.named_parameters(): p.requires_grad_(n in used)
    return sum(p.numel() for p in student.parameters() if p.requires_grad)

def new_models(C, dev):
    set_seed(CFG["seed"]); s, m = build_sv_student(C["dim"]); freeze_unused(s); return s.to(dev), m.to(dev)

def _load_resumable(f):
    import torch
    try: ck = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as e: return None, f"unreadable ({type(e).__name__}: {str(e)[:120]})"
    if not isinstance(ck, dict) or not RESUME_KEYS <= set(ck): return None, "not a complete resumable checkpoint"
    return ck, ""

def _position(S): return (S["update"], S["epoch"], len(S["seen"]))

def memory_probe(student, msv, C, dev, amp, max_sec):
    import torch
    L = int(round(CFG["crop_seconds"] * SR)); total = torch.cuda.get_device_properties(0).total_memory
    while max_sec >= 20:
        nb = max(1, min(CFG["max_batch_utts"], int(max_sec * SR // L))); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); peak = float("nan")
        try:
            T_ = fbank_frames(L); b = dict(fb=torch.randn(nb, T_, 80), fb_lens=torch.full((nb,), T_)); v = sv_forward(student, msv, b, dev, amp)
            loss, _ = kd_loss(v, torch.randn(nb, msv.D, device=dev), 1e-8); loss.backward(); peak = torch.cuda.max_memory_allocated(); ok = peak < CFG["mem_fraction"] * total
        except torch.OutOfMemoryError: ok = False
        for p in list(student.parameters()) + list(msv.parameters()): p.grad = None
        torch.cuda.empty_cache()
        log(f"  memory probe: {nb} x {L / SR:.0f} s = {nb * L / SR:.0f} s padded -> peak {peak / 2**30:.2f} GiB of {total / 2**30:.1f} GiB {'OK' if ok else '-> reducing'}")
        if ok: return max_sec
        max_sec *= 0.8
    raise RuntimeError("even a 20 s batch does not fit in GPU memory")

def batch_count_of(update, MBS, cfg, progress):
    ic = update * MBS * cfg["accum_steps"] / cfg["ref_duration"]          # icefall: batches of the reference duration seen so far
    return max(ic, 20000.0 * progress) if cfg["batch_count_mode"] == "progress" else ic

def sv_train(mode, C, P=None, drive=None, overrides=None, interrupt_after_update=None, probe=True):
    """Resumes automatically from the most advanced VALID checkpoint among: local latest, Drive checkpoint_latest.pt (+ an upload whose rename was interrupted)."""
    import torch
    P = P or paths(mode); ensure_dirs(P); dev = torch.device("cuda"); amp = amp_dtype(); cfg = dict(CFG, **MODES[mode]); cfg.update(overrides or {})
    LATEST, BEST = P["latest"] / "checkpoint_latest.pt", P["best"] / "checkpoint_best.pt"; N = len(C["tr"])
    lens = np.array([crop_len(r.n_frames, r.sr, CFG["crop_seconds"]) for r in C["tr"].itertuples()])
    student, msv = new_models(C, dev); params = [p for _, p in student.sv_params()] + list(msv.parameters()); opt, sch = make_optimizer(student, msv, cfg)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 12, enabled=amp == torch.float16)
    sig_keys = ["seed", "optimizer", "accum_steps", "max_epochs", "max_batch_utts", "crop_seconds", "sv_block", "msv_attention_channels", "cos_eps", "batch_count_mode", "msv_lr_scale"] + \
               (["base_lr", "lr_batches", "lr_epochs", "eden_warmup_batches", "eden_warmup_start", "clipping_scale"] if cfg["optimizer"] == "scaled_adam"
                else ["adamw_lr", "weight_decay", "warmup_updates", "min_lr_frac", "grad_clip"])
    SIG = sha_of([json.dumps({k: cfg[k] for k in sig_keys}, sort_keys=True, default=str), C["manifest_sha"], C["dev_sha"], C["cache_ids"], C["norm_sha"], CODE_VERSION])
    S = dict(epoch=0, batch=0, update=0, best=None, es_counter=0, train_hist=[], val_hist=[], elapsed=0.0, skips=0, finished=None, seen=[], plan_remaining=None,
             ep_sum=0.0, ep_n=0, max_batch_seconds=None, sessions=0, resumes=[], peak_gpu_gib=0.0)
    cands = [("local disk", LATEST)] + ([("Google Drive", f) for f in drive.candidates("checkpoint_latest.pt")] if drive else []); chosen = None
    for src, f in cands:
        if not Path(f).exists(): continue
        ck, why = _load_resumable(f)
        if ck is None: log(f"  ignored {src} file {f}: {why} (e.g. an interrupted save or upload)"); continue
        if ck["sig"] != SIG and src == "local disk":
            dst = P["ckpt"].parent / f"{P['ckpt'].name}_other_config_{time.strftime('%Y%m%d_%H%M%S')}"; del ck; os.replace(P["ckpt"], dst); ensure_dirs(P)
            log(f"  local checkpoints of a DIFFERENT configuration moved aside: {dst}"); continue
        if ck["sig"] != SIG:
            raise RuntimeError(f"{f} ({src}) is a checkpoint of a DIFFERENT configuration / data / teacher cache (sig {ck['sig']}, this run {SIG}). Left untouched: "
                               f"move it out of {drive.dir} to start a new run, or restore the settings it was made with.")
        k = _position(ck["state"]); log(f"  valid checkpoint on {src}: {Path(f).name}, epoch {ck['state']['epoch'] + 1}, batch {ck['state']['batch']}, global step {k[0]}")
        if chosen is None or k > chosen[2]: chosen = (src, f, k, ck)
        else: del ck
    if chosen:
        src, f, k, ck = chosen
        if src == "Google Drive" and str(f).endswith(".tmp"): f = drive.promote_tmp("checkpoint_latest.pt")
        student.load_state_dict(ck["student"], strict=True); msv.load_state_dict(ck["msv"], strict=True); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        if sch is not None and ck["scheduler"].get("eden"): sch.load_state_dict(ck["scheduler"]["eden"])
        S = ck["state"]; torch.set_rng_state(ck["rng_cpu"].cpu()); torch.cuda.set_rng_state(ck["rng_cuda"].cpu()); np.random.set_state(ck["rng_np"]); random.setstate(ck["rng_py"])
        log("\n" + "#" * 70 + "\n#  RESUMING FROM CHECKPOINT\n" + "#" * 70)
        log(f"  source {src}: {f}\n  epoch {S['epoch'] + 1}/{cfg['max_epochs']} | batch position {S['batch']} ({len(S['seen'])}/{N} utterances of this epoch done) | "
            f"global optimizer step {S['update']} | best val KD loss {S['best']['val_loss'] if S['best'] else None} | early-stop counter {S['es_counter']}/{cfg['patience']}")
        S.setdefault("resumes", []).append(dict(source=src, update=S["update"], epoch=S["epoch"], batch=S["batch"], seen=len(S["seen"]), time=time.strftime("%Y-%m-%d %H:%M:%S")))
        if src == "Google Drive": shutil.copyfile(f, str(LATEST) + ".part"); os.replace(str(LATEST) + ".part", LATEST)
        del ck
        if cfg["max_batch_seconds"] < S["max_batch_seconds"]:
            log(f"  max_batch_seconds lowered {S['max_batch_seconds']:.0f} -> {cfg['max_batch_seconds']:.0f}: the remaining utterances of this epoch are re-batched")
            S["max_batch_seconds"] = cfg["max_batch_seconds"]; S["plan_remaining"] = sorted(set(range(N)) - set(S["seen"])); S["batch"] = 0
    else: log("\n" + "#" * 70 + "\n#  STARTING NEW TRAINING RUN\n" + "#" * 70)
    if drive: drive.clean_stale_tmp()
    # ---- the best checkpoint is never replaced by a worse model (a newer best.pt from work lost in an interruption is a floor)
    lb = best_meta(BEST) if BEST.exists() else None
    if lb and lb.get("sig") != SIG: dst = BEST.with_name(f"best_other_config_{time.strftime('%Y%m%d_%H%M%S')}.pt"); os.replace(BEST, dst); lb = None
    db = drive.best_info() if drive else None
    if db and "error" in db: log(f"  Drive checkpoint_best.pt unreadable ({db['error']}): will be replaced by the next best model"); db = None
    if db and db.get("sig") != SIG: raise RuntimeError(f"{drive.dir / 'checkpoint_best.pt'} belongs to a different configuration; left untouched. Move it away to start a new run.")
    floor = None
    for src_, m in (("Google Drive", db), ("local disk", lb)):
        if m and (S["best"] is None or m["val_loss"] < S["best"]["val_loss"]) and (floor is None or m["val_loss"] < floor["val_loss"]): floor = dict(m, source=src_)
    if floor: log(f"  existing best checkpoint on {floor['source']} (val KD loss {floor['val_loss']:.5f}) is only replaced by a better model")
    if drive:
        if db and (lb is None or lb["val_loss"] > db["val_loss"]): shutil.copyfile(drive.dir / "checkpoint_best.pt", str(BEST) + ".part"); os.replace(str(BEST) + ".part", BEST)
        elif lb and (db is None or lb["val_loss"] < db["val_loss"]): drive.push(BEST, "checkpoint_best.pt")
    S["sessions"] += 1
    if drive and not drive.fake:
        n_all = sum(p.numel() for p in student.parameters()) + sum(p.numel() for p in msv.parameters()); n_tr = sum(p.numel() for p in params)
        drive.check_space(2 * (4.05 * n_all + 8.2 * n_tr + 4.05 * n_all) + 4.05 * n_all + 50e6, "checkpoints (latest + best, each via a temporary copy, + exported weights)")
    if S["max_batch_seconds"] is None: S["max_batch_seconds"] = memory_probe(student, msv, C, dev, amp, cfg["max_batch_seconds"]) if probe else cfg["max_batch_seconds"]
    MBS = S["max_batch_seconds"]
    def lr_at(update, prog):
        w = cfg["adamw_lr"] * min(1.0, (update + 1) / cfg["warmup_updates"]); cosv = cfg["min_lr_frac"] + (1 - cfg["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return min(w, cfg["adamw_lr"] * cosv) if update < cfg["warmup_updates"] else cfg["adamw_lr"] * cosv
    progress = lambda: (S["epoch"] + len(S["seen"]) / N) / cfg["max_epochs"]
    def cur_lr():
        if sch is None:
            lr = lr_at(S["update"], progress()); opt.param_groups[0]["lr"] = opt.param_groups[1]["lr"] = lr; opt.param_groups[2]["lr"] = lr * cfg["msv_lr_scale"]; return lr
        sch.step_epoch(S["epoch"]); sch.step_batch(S["update"] + 1); return sch.get_last_lr()[0]
    run_cfg = dict(code_version=CODE_VERSION, sig=SIG, mode=mode, cfg={k: cfg[k] for k in sorted(cfg) if not callable(cfg[k])}, max_batch_seconds_effective=MBS,
                   teacher=dict(model_id=C["tid"]["model_id"], commit=C["tid"]["commit"], representation=spec_name(C["spec"]), spec=C["spec"], dim=C["dim"],
                                normalisation=f"{C['spec']['norm']} ({mode} training-set statistics, label-free), sha {C['norm_sha']}"),
                   caches=C["cache_ids"], data=dict(train_utts=N, train_speakers=int(C["tr"].speaker.nunique()), train_hours=round(float(C["tr"].duration_s.sum() / 3600), 3),
                                                     train_manifest_sha=C["manifest_sha"], val_utts=len(C["dv"]), val_manifest_sha=C["dev_sha"]),
                   preprocessing=dict(student="80-D kaldi fbank (25 ms / 10 ms, povey, snip_edges=False) of the same centred <= crop_seconds segment the teacher saw",
                                      teacher="ReDimNet2 front end (signal normalisation, pre-emphasis, 72 log-mel, mean normalisation); 16 kHz mono"),
                   student=dict(model="Zipformer-M (icefall)", zip_args=ZIP_ARGS, sv_block=CFG["sv_block"], sv_block_dim=student.sv_dim,
                                msv="attentive pooling (Conv1d-tanh-Conv1d, masked softmax) -> attentive mean+std -> Linear", trainable_params=int(sum(p.numel() for p in params))),
                   loss="1 - CosineSim(v_teacher, M_SV(R_student^(3))) (MT2KD Eq. 8); no speaker labels")
    atomic_json(run_cfg, P["logs"] / "run_config.json")
    if drive:
        try:
            import yaml; drive.write_text("config.yaml", yaml.safe_dump(json.loads(json.dumps(run_cfg, default=str)), sort_keys=False))
        except Exception as e: log(f"  (config.yaml not written: {e})")
    log(f"  train {N:,} utts ({C['tr'].speaker.nunique()} speakers) | val {len(C['dv']):,} | max {cfg['max_epochs']} epochs | "
        + (f"ScaledAdam + Eden (base lr {cfg['base_lr']}, lr_batches {cfg['lr_batches']}, lr_epochs {cfg['lr_epochs']}, warm-up {cfg['eden_warmup_batches']})" if sch else f"AdamW {cfg['adamw_lr']}")
        + f" | {str(amp).split('.')[-1]} | batches <= {MBS:.0f} s | teacher {C['dim']}-D | batch_count {cfg['batch_count_mode']}")
    t_ref = [time.time()]; last_local = [time.time()]; last_drive = [time.time()]; t_session = time.time(); budget = cfg.get("max_session_hours")
    def sched_state(): return dict(update=S["update"], progress=progress(), lr=float(opt.param_groups[0]["lr"]), eden=sch.state_dict() if sch is not None else None)
    def save_latest(to_drive=False):
        S["elapsed"] += time.time() - t_ref[0]; t_ref[0] = time.time()
        atomic_torch_save(dict(sig=SIG, student=student.state_dict(), msv=msv.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), scheduler=sched_state(), state=S,
                               rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(), rng_np=np.random.get_state(), rng_py=random.getstate(), cfg=run_cfg["cfg"],
                               teacher=run_cfg["teacher"], caches=C["cache_ids"], manifests=dict(train=C["manifest_sha"], val=C["dev_sha"]), code_version=CODE_VERSION), LATEST)
        atomic_csv(pd.DataFrame(S["train_hist"]), P["logs"] / "training_log.csv"); last_local[0] = time.time()
        if to_drive and drive: drive.push(LATEST, "checkpoint_latest.pt"); drive.write_csv("training_log.csv", pd.DataFrame(S["train_hist"])); last_drive[0] = time.time()
    def write_val():
        V = pd.DataFrame(S["val_hist"]); atomic_csv(V, P["logs"] / "validation_log.csv")
        if drive: drive.write_csv("validation_log.csv", V)
    student.train(); msv.train(); win = []; pending = []; stop = None
    try:
        while S["epoch"] < cfg["max_epochs"] and not S["finished"]:
            ep = S["epoch"]; batches = make_batches(lens, ep, cfg["seed"], True, MBS, cfg["max_batch_utts"], idx=S["plan_remaining"])
            k0 = S["batch"]; t_ep = time.time(); torch.cuda.reset_peak_memory_stats(); nb = [len(x) for x in batches]
            log(f"epoch {ep + 1}: {len(batches)} batches, {min(nb)}-{max(nb)} utterances per batch" + (f" | continuing at batch {k0}" if k0 or S["plan_remaining"] is not None else ""))
            tb = time.time()
            for k, b in enumerate(sv_loader(C["tr"], C["T_tr"], batches[k0:], CFG["crop_seconds"], cfg["num_workers"], seed=cfg["seed"] + ep), start=k0):
                set_batch_count(student, batch_count_of(S["update"], MBS, cfg, progress()))
                v = sv_forward(student, msv, b, dev, amp); loss, cos = kd_loss(v, b["t"].to(dev), cfg["cos_eps"]); n_inv = int((~torch.isfinite(v).all(-1)).sum())
                if not torch.isfinite(loss) or n_inv:
                    S["skips"] += 1; log(f"  non-finite loss / {n_inv} invalid student embeddings at step {S['update']} (batch skipped, {S['skips']} in a row)")
                    opt.zero_grad(set_to_none=True)
                    for kk in pending: S["seen"] += batches[kk]
                    pending = []; S["seen"] += batches[k]; S["batch"] = k + 1
                    if S["skips"] > cfg["max_consecutive_skips"]: save_latest(True); raise FloatingPointError("too many non-finite losses; state saved")
                    continue
                scaler.scale(loss / cfg["accum_steps"]).backward(); pending.append(k)
                win.append((float(loss), float(cos.mean()), float(v.detach().norm(dim=-1).mean()), float(b["t"].norm(dim=-1).mean()), len(b["i"]), int(b["n"].sum())))
                if (k + 1) % cfg["accum_steps"] and k + 1 < len(batches): continue
                lr = cur_lr(); scaler.unscale_(opt); gn = float(torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"] if sch is None else float("inf")))
                if math.isfinite(gn): scaler.step(opt); S["skips"] = 0
                else:
                    S["skips"] += 1; log(f"  WARNING: non-finite gradient norm at step {S['update']} (update skipped, {S['skips']} in a row)")
                    if scaler.is_enabled(): scaler.step(opt)
                    if S["skips"] > cfg["max_consecutive_skips"]: save_latest(True); raise FloatingPointError("too many non-finite gradients; state saved")
                if math.isfinite(gn) and gn > 1e3: log(f"  WARNING: large gradient norm {gn:.1f} at step {S['update']}")
                scaler.update(); opt.zero_grad(set_to_none=True); S["update"] += 1
                for kk in pending: S["seen"] += batches[kk]
                for w_ in win[-len(pending):]: S["ep_sum"] += w_[0] * w_[4]; S["ep_n"] += w_[4]
                pending = []; S["batch"] = k + 1; u = S["update"]
                if u % cfg["log_every"] == 0 or u == 1:
                    w = np.array(win); win = []; dt = time.time() - tb; tb = time.time(); pk = torch.cuda.max_memory_allocated() / 2**30; S["peak_gpu_gib"] = max(S["peak_gpu_gib"], pk)
                    rec = dict(update=u, epoch=ep + 1, batch=k + 1, loss=float(np.average(w[:, 0], weights=w[:, 4])), cos=float(np.average(w[:, 1], weights=w[:, 4])),
                               student_norm=float(w[:, 2].mean()), teacher_norm=float(w[:, 3].mean()), lr=lr, grad_norm=gn, batch_count=batch_count_of(u, MBS, cfg, progress()),
                               gpu_peak_gib=round(pk, 3), sec_per_batch=dt / len(w), utts_per_sec=float(w[:, 4].sum() / max(dt, 1e-9)), audio_sec_per_sec=float(w[:, 5].sum() / SR / max(dt, 1e-9)))
                    S["train_hist"].append(rec); el = time.time() - t_ep
                    log(f"[ep {ep + 1}/{cfg['max_epochs']} | b {k + 1}/{len(batches)} | step {u}] KD loss {rec['loss']:.4f} | cos {rec['cos']:.4f} | |v_s| {rec['student_norm']:.2f} "
                        f"|v_t| {rec['teacher_norm']:.2f} | |g| {gn:.2f} | lr {lr:.2e} | {rec['utts_per_sec']:.0f} utt/s | GPU {pk:.1f} GiB | epoch ETA "
                        f"{el / max(k + 1 - k0, 1) * (len(batches) - k - 1) / 60:.1f} min")
                if drive and (u % cfg["drive_ckpt_every_updates"] == 0 or time.time() - last_drive[0] > 60 * cfg["drive_ckpt_every_minutes"]):
                    save_latest(True); log(f"  checkpoint saved at step {u} (epoch {ep + 1}, batch {k + 1}/{len(batches)}): local + Google Drive")
                elif u % cfg["local_ckpt_every_updates"] == 0 or time.time() - last_local[0] > 60 * cfg["ckpt_every_minutes"]: save_latest(); log(f"  checkpoint saved at step {u} (local)")
                if interrupt_after_update is not None and u == interrupt_after_update: raise SimulatedDisconnect(f"simulated runtime loss after step {u}")
                if budget and time.time() - t_session > 3600 * budget: save_latest(True); write_val(); stop = f"session time budget of {budget} h reached"; break
            if stop: break
            assert len(set(S["seen"])) == N == len(S["seen"]), f"epoch {ep + 1} saw {len(set(S['seen']))} distinct of {N} utterances ({len(S['seen'])} total)"
            v_, _ = sv_evaluate(student, msv, C, dev, amp); tr_loss = S["ep_sum"] / max(S["ep_n"], 1)
            v_.update(epoch=ep + 1, update=S["update"], train_loss=tr_loss, lr=float(opt.param_groups[0]["lr"]), epoch_minutes=(time.time() - t_ep) / 60)
            improved = S["best"] is None or v_["val_loss"] < S["best"]["val_loss"] - cfg["min_delta"]
            if improved and floor and v_["val_loss"] >= floor["val_loss"]:
                S["best"] = {k: floor[k] for k in ("val_loss", "epoch", "update")}; S["es_counter"] = 0; improved = False; floor = None
            elif improved:
                S["best"] = dict(val_loss=v_["val_loss"], val_cos=v_["val_cos"], epoch=ep + 1, update=S["update"], val_eer_nonofficial=v_.get("val_eer_nonofficial")); S["es_counter"] = 0; floor = None
                atomic_torch_save(dict(student=student.state_dict(), msv=msv.state_dict(), val=v_, epoch=ep + 1, update=S["update"], sig=SIG, cfg=run_cfg["cfg"], teacher=run_cfg["teacher"],
                                       caches=C["cache_ids"], zip_args=ZIP_ARGS, sv_block=CFG["sv_block"], msv_dims=dict(C=msv.C, A=msv.A, D=msv.D), code_version=CODE_VERSION), BEST)
                if drive: drive.push(BEST, "checkpoint_best.pt")
            else: S["es_counter"] += 1
            v_["best"] = improved; v_["es_counter"] = S["es_counter"]; S["val_hist"].append(v_)
            log(f"=== epoch {ep + 1}: train KD loss {tr_loss:.4f} | VAL KD loss {v_['val_loss']:.4f} | val cos {v_['val_cos']:.4f} (one constant vector: {v_['val_cos_constant_baseline']:.4f}) | "
                f"val EER (non-official trials) {v_.get('val_eer_nonofficial', float('nan')) * 100:.2f}% | invalid {v_['invalid_embeddings']} | step {S['update']} | "
                f"{v_['epoch_minutes']:.1f} min" + (" | NEW BEST" if improved else f" | no improvement ({S['es_counter']}/{cfg['patience']})") + " ===")
            S["epoch"] = ep + 1; S["batch"] = 0; S["seen"] = []; S["plan_remaining"] = None; S["ep_sum"] = 0.0; S["ep_n"] = 0
            if S["es_counter"] >= cfg["patience"]: S["finished"] = f"early stopping after epoch {ep + 1} (best epoch {S['best']['epoch']})"
            elif S["epoch"] >= cfg["max_epochs"]: S["finished"] = f"max_epochs {cfg['max_epochs']} reached"
            save_latest(True); write_val(); plots(P, drive)
    except torch.OutOfMemoryError:
        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); save_latest(True); write_val()
        if drive: drive.wait()
        raise RuntimeError(f"CUDA out of memory. State saved at step {S['update']}. Set CFG['max_batch_seconds'] lower (effective {MBS:.0f}) and re-run.")
    except KeyboardInterrupt:
        save_latest(True); write_val(); log("interrupted: state saved; waiting for the Google Drive upload ...")
        if drive: drive.wait()
        raise
    if stop: log(f"{stop}: state saved at step {S['update']}. Re-run the cell to continue.")
    del student, msv, opt, params; torch.cuda.empty_cache()
    if drive: drive.wait()
    return dict(finished=S["finished"], stop=stop, state=S, P=P)

# ####################################################################################################
# ---- resume test and learning check (short runs on tiny subsets; a LOCAL stand-in folder replaces Drive)
# ####################################################################################################
DRIVE_MODE_FILES = ("checkpoint_latest.pt", "checkpoint_best.pt", "student_encoder.pt", "sv_projection.pt", "training_log.csv", "validation_log.csv",
                    "final_report.json", "training_curves.png", "config.yaml", "train_manifest.csv")

def tiny_context(C, n_train, n_dev, dev_from_train=False):
    trs = C["tr"].sort_values(["duration_s", "uid"]); it = np.linspace(0, len(trs) // 2, n_train).astype(int)
    tr = trs.iloc[it].drop_duplicates("uid"); Ttr = C["T_tr"][[C["row_tr"][u] for u in tr.uid]]
    if dev_from_train: dv, Tdv = tr, Ttr
    else:
        dvs = C["dv"].sort_values(["duration_s", "uid"]); dv = dvs.iloc[np.linspace(0, len(dvs) // 2, n_dev).astype(int)].drop_duplicates("uid"); Tdv = C["T_dv"][[C["row_dv"][u] for u in dv.uid]]
    tr, dv = tr.reset_index(drop=True), dv.reset_index(drop=True)
    return dict(C, tr=tr, dv=dv, T_tr=Ttr, T_dv=Tdv, sel_trials=None, manifest_sha=sha_of(sorted(tr.uid)), dev_sha=sha_of(sorted(dv.uid)))

def _tinyP(root, name):
    P = dict(ckpt=root / name / "ckpt", logs=root / name / "logs", results=root / name / "results"); P.update(best=P["ckpt"] / "best", latest=P["ckpt"] / "latest"); return P

def resume_test(C):
    """Deliberately interrupted short runs (24 / 8 utterances): run A killed without saving; local checkpoints deleted (= runtime recycle) and an
    incomplete upload planted; run B must resume from the Drive copy; killed again; run C must resume from the more advanced local checkpoint and finish.
    Checks: resume source / epoch / batch / global step; each step done once with the reference's (epoch, batch); losses = reference (3 % / 10 %)."""
    import torch, gc
    root = Path(CFG["write_root"]) / "resume_test"
    if root.exists(): shutil.rmtree(root)
    Ct = tiny_context(C, 24, 8); LE, DE = 3, 7
    ov = dict(max_epochs=4, eden_warmup_batches=10, warmup_updates=4, local_ckpt_every_updates=LE, drive_ckpt_every_updates=DE, ckpt_every_minutes=1e9, drive_ckpt_every_minutes=1e9, log_every=1,
              max_batch_seconds=40.0, max_batch_utts=3, patience=99, max_session_hours=None)    # <= 3 utterances per batch -> 8 batches/epoch, 32 steps
    def go(P, ddir, interrupt=None):
        drv = DriveStore(ddir, DRIVE_MODE_FILES, fake=True); out = "finished"
        try: sv_train("sanity", Ct, P=P, drive=drv, overrides=ov, interrupt_after_update=interrupt, probe=False)
        except SimulatedDisconnect as e: out = str(e)
        drv.wait(); gc.collect(); torch.cuda.empty_cache(); return out
    def state(P): ck = torch.load(P["latest"] / "checkpoint_latest.pt", map_location="cpu", weights_only=False); s = ck["state"]; del ck; return s
    log("---- reference run (uninterrupted)"); Pr = _tinyP(root, "reference"); go(Pr, root / "drive_reference"); R = state(Pr)
    ref = {h["update"]: h for h in R["train_hist"]}; total = R["update"]; ends = {v["update"] for v in R["val_hist"]}
    lpts = {u for u in ref if u % LE == 0} | ends; dpts = {u for u in ref if u % DE == 0} | ends; lastle = lambda pts, u: max([p for p in pts if p <= u], default=0)
    c1 = next((u for u in range(DE + 2, total) if u not in lpts | dpts), None)
    c2 = next((u for u in range(c1 + 2, total) if u not in lpts | dpts and lastle(lpts, u) > lastle(dpts, u) > lastle(dpts, c1)), None) if c1 else None
    if c2 is None: raise RuntimeError(f"resume test: the reference run has only {total} steps, too few for two interruptions")
    expect = lambda u: (ref[u]["epoch"], 0) if u in ends else (ref[u]["epoch"] - 1, ref[u]["batch"])
    Pb, Db = _tinyP(root, "interrupted"), root / "drive_interrupted"; checks = []
    log(f"---- run A: killed after step {c1}"); go(Pb, Db, c1)
    shutil.rmtree(Pb["ckpt"]); (Db / "checkpoint_latest.pt.tmp").write_bytes(os.urandom(1 << 20))
    log(f"---- runtime recycle simulated -> run B must resume from Drive at step {lastle(dpts, c1)}; killed after step {c2}"); go(Pb, Db, c2)
    log(f"---- run C: must resume from the local checkpoint at step {lastle(lpts, c2)} and finish"); go(Pb, Db)
    F = state(Pb); res = F["resumes"]; exp = [("Google Drive", lastle(dpts, c1)), ("local disk", lastle(lpts, c2))]
    checks.append(("resume sources and global steps", [(r["source"], r["update"]) for r in res] == exp, f"{[(r['source'], r['update']) for r in res]} expected {exp}"))
    pos = [(r["epoch"], r["batch"]) for r in res]; epos = [expect(u) for _, u in exp]
    checks.append(("resumed epoch and batch position", pos == epos, f"{pos} expected {epos}"))
    upd = [h["update"] for h in F["train_hist"]]
    checks.append(("every optimizer step done exactly once", upd == list(range(1, total + 1)), f"{len(upd)} steps, reference {total}"))
    checks.append(("each step used the reference's epoch/batch", all((h["epoch"], h["batch"]) == (ref[h["update"]]["epoch"], ref[h["update"]]["batch"]) for h in F["train_hist"]), ""))
    rel = {h["update"]: abs(h["loss"] - ref[h["update"]]["loss"]) / max(ref[h["update"]]["loss"], 1e-6) for h in F["train_hist"]}; first = [rel[u + 1] for _, u in exp]
    checks.append(("first loss after each resume == reference", max(first) < 3e-2, f"relative differences {[f'{x:.1e}' for x in first]}"))
    checks.append(("all losses == reference", max(rel.values()) < 0.1, f"max relative difference {max(rel.values()):.1e}"))
    checks.append(("final state == reference", F["epoch"] == R["epoch"] and F["update"] == R["update"] and bool(F["finished"])
                   and abs(F["val_hist"][-1]["val_loss"] - R["val_hist"][-1]["val_loss"]) <= 0.1 * max(R["val_hist"][-1]["val_loss"], 1e-3),
                   f"epochs {F['epoch']}/{R['epoch']}, steps {F['update']}/{R['update']}, final val loss {F['val_hist'][-1]['val_loss']:.4f}/{R['val_hist'][-1]['val_loss']:.4f}"))
    checks.append(("incomplete upload ignored and removed", not (Db / "checkpoint_latest.pt.tmp").exists(), ""))
    bm = best_meta(Db / "checkpoint_best.pt"); vbest = min(v["val_loss"] for v in F["val_hist"])
    checks.append(("best checkpoint holds the lowest validation loss", bm["val_loss"] <= vbest + CFG["min_delta"] + 1e-9, f"{bm['val_loss']:.5f} vs {vbest:.5f}"))
    for n, okk, info in checks: log(f"  [{'PASS' if okk else 'FAIL'}] {n}: {info}")
    rep = dict(passed=all(c[1] for c in checks), code_version=CODE_VERSION, time=time.strftime("%Y-%m-%d %H:%M:%S"), interrupts=[c1, c2], total_steps=total,
               checks=[dict(check=n, passed=bool(o), info=i) for n, o, i in checks])
    atomic_json(rep, paths()["logs"] / "resume_test_report.json")
    if rep["passed"]: shutil.rmtree(root); log("RESUME TEST PASSED")
    else: log(f"RESUME TEST FAILED: training is blocked (test folders kept in {root})")
    return rep

def learning_check(C, n=64, epochs=24):
    """~300 optimizer steps on 64 training utterances (validation = the same utterances, eval mode): the student must reach
    cos >= CFG['learning_check_min_cos'] with the teacher, i.e. clearly more than one constant vector can (printed)."""
    import torch, gc
    root = Path(CFG["write_root"]) / "learning_check"
    if root.exists(): shutil.rmtree(root)
    Ct = tiny_context(C, n, 0, dev_from_train=True); P = _tinyP(root, "run")
    ov = dict(max_epochs=epochs, eden_warmup_batches=50, warmup_updates=50, local_ckpt_every_updates=10**9, drive_ckpt_every_updates=10**9, ckpt_every_minutes=1e9, drive_ckpt_every_minutes=1e9,
              log_every=50, max_batch_seconds=40.0, patience=10**9, max_session_hours=None)
    log(f"---- learning check: {len(Ct['tr'])} training utterances, {epochs} epochs, optimizer {CFG['optimizer']}")
    r = sv_train("sanity", Ct, P=P, drive=None, overrides=ov, probe=False); gc.collect(); torch.cuda.empty_cache(); S = r["state"]
    curve = [round(v["val_cos"], 3) for v in S["val_hist"]]; base = S["val_hist"][-1]["val_cos_constant_baseline"]; ok = curve[-1] >= CFG["learning_check_min_cos"]
    log(f"  cosine with the teacher per epoch (same 64 utterances, eval mode): {curve}\n  final {curve[-1]:.3f} (needs >= {CFG['learning_check_min_cos']}; one constant vector: {base:.3f})")
    rep = dict(passed=bool(ok), code_version=CODE_VERSION, optimizer=CFG["optimizer"], cos_curve=curve, final_cos=curve[-1], constant_baseline=base, steps=S["update"],
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    atomic_json(rep, paths()["logs"] / "learning_check.json"); shutil.rmtree(root)
    log("LEARNING CHECK PASSED" if ok else "LEARNING CHECK FAILED: the student cannot fit even 64 teacher embeddings -> training blocked")
    return rep
# ####################################################################################################
# ---- context, official EER, sanity checks, curves, final report and exports
# ####################################################################################################
TOP_FILES = ("teacher_info.json", "teacher_selection_report.json", "dataset_validation_report.json", "train_manifest.csv", "validation_manifest.csv",
             "test_manifest.csv", "excluded_files.csv", "resume_test_report.json", "selection_trials_NONOFFICIAL.csv")

def usable(df):
    """Drops files that failed the full decode (their reason is in excluded_files.csv)."""
    f = paths()["meta"] / "decode_failures.csv"
    if not f.exists(): return df
    bad = set(pd.read_csv(f, dtype=str).uid); return df[~df.uid.isin(bad)].reset_index(drop=True)

def mode_manifests(mode):
    tr = usable(load_manifest("sanity_train_manifest" if mode == "sanity" else "train_manifest")); dv = usable(load_manifest("validation_manifest"))
    te = usable(load_manifest("test_manifest")); return tr, dv, te

def complete_trials():
    out = {}
    for n in OFFICIAL_TRIALS:
        f = paths()["meta"] / f"trials_{n.replace('.txt', '')}.csv"
        if f.exists(): out[n] = pd.read_csv(f, dtype={"enrol": str, "test": str})
    return out

def build_context(mode, tid, spec, drive_top):
    tr, dv, te = mode_manifests(mode); cache = EmbCache(tid, spec, CFG["crop_seconds"], drive_top)
    E_tr, E_dv = cache.load(tr), cache.load(dv); mu, sd = norm_stats(E_tr, spec["norm"]); norm_sha = hashlib.sha1(mu.tobytes() + sd.tobytes()).hexdigest()[:12]
    f = paths()["teacher"] / f"normalisation_{mode}_{spec['norm']}_{norm_sha}.npz"
    if not f.exists():
        np.savez(str(f) + ".part.npz", mu=mu, sd=sd, train_uids_sha=sha_of(sorted(tr.uid))); os.replace(str(f) + ".part.npz", f)
        if drive_top: drive_top.write_file(f, f"{cache.rel}/normalisation_{mode}_{spec['norm']}.npz")
    _, sel_trials = selection_sets(dv)
    return dict(mode=mode, tr=tr, dv=dv, te=te, T_tr=((E_tr - mu) / sd).astype(np.float32), T_dv=((E_dv - mu) / sd).astype(np.float32), mu=mu, sd=sd, norm_sha=norm_sha,
                spec=spec, dim=int(E_tr.shape[1]), tid=tid, cache_ids=[cache.cid, cache_identity(tid, spec, None)[0]], manifest_sha=sha_of(sorted(tr.uid)),
                dev_sha=sha_of(sorted(dv.uid)), sel_trials=sel_trials, row_tr={u: i for i, u in enumerate(tr.uid)}, row_dv={u: i for i, u in enumerate(dv.uid)})

def official_eer(E, df, label):
    """EER on every COMPLETE official VoxCeleb1-O trial list (cosine, no score normalisation)."""
    row = {u: i for i, u in enumerate(df.uid)}; out = {}
    for n, t in complete_trials().items():
        s, y, miss = score_trials(E, row, t)
        if miss: out[n] = dict(status=f"NOT OFFICIAL: {miss} of {len(t)} trials lack an utterance (missing or excluded files)", trials_scored=len(s),
                               eer_on_available=compute_eer(s, y)[0] if len(s) else None)
        else: out[n] = dict(status="official", trials=len(t), target=int(t.label.sum()), eer=compute_eer(s, y)[0], scoring="cosine, no score normalisation")
        e = out[n].get("eer", out[n].get("eer_on_available")); log(f"  {label} | {n}: " + (f"EER {e * 100:.3f}% on {len(s):,} trials ({out[n]['status']})" if e is not None else out[n]["status"]))
    if not out: log(f"  {label}: no official trial list available -> official EER not computed")
    return out

def teacher_official_eer(C, tid, spec, drive_top):
    if not complete_trials(): return {}
    E = EmbCache(tid, spec, None, drive_top).load(C["te"])
    out = dict(kd_target=official_eer((E - C["mu"]) / C["sd"], C["te"], f"TEACHER ({C['spec']['norm']} = KD target)"))
    if C["spec"]["norm"] != "none": out["raw"] = official_eer(E, C["te"], "TEACHER (raw embedding)")
    return out

def sanity_checks(C, tv, tsel, tid):
    """Numbered PASS/FAIL checks (task section 17). Critical failures block training."""
    import torch, gc
    dev = torch.device("cuda"); amp = amp_dtype(); R = []; st = {}
    def chk(no, name, fn, critical=True):
        t0 = time.time()
        try: info = fn(); R.append(dict(no=no, check=name, critical=critical, status="PASS", info=info)); log(f"[{no:2d}] PASS  {name}: {info}")
        except Exception as e: R.append(dict(no=no, check=name, critical=critical, status="FAIL", info=f"{type(e).__name__}: {e}")); log(f"[{no:2d}] {'FAIL' if critical else 'WARN'}  {name}: {type(e).__name__}: {e}")
    def c1():
        tr, dv, te = C["tr"], C["dv"], C["te"]; a, b, c = set(tr.speaker), set(dv.speaker), set(te.speaker)
        assert not a & b and not a & c and not b & c, "speaker overlap between train / validation / test"
        assert not set(tr.uid) & set(te.uid) and len(tr) and len(dv), "empty manifest or test utterances in training"
        ver = set(open(paths()["meta"] / "decode_verified.txt").read().split()); assert set(tr.uid) | set(dv.uid) <= ver, "manifest files not fully decoded"
        return f"train {len(tr):,} utts / {len(a)} spk | val {len(dv):,} / {len(b)} spk | test {len(te):,} / {len(c)} spk | speaker-disjoint, all decoded"
    def c2():
        w = list(tid["weight_files"].values())[0]; assert tid["commit"] == CFG["redimnet_commit"] and w["sha256"] == CFG["redimnet_weights_sha256"] and tid["params"] > 1e6
        return f"{tid['model_id']} | code @ {tid['commit'][:12]} (pinned) | weights sha256 {w['sha256'][:16]}... (expected) | {tid['params'] / 1e6:.2f}M params | {tid['model_class']}"
    def c3(): assert tv["output_shape"] == [1, tid["embed_dim"]] and tv["finite"]; return f"forward pass on real audio: output {tuple(tv['output_shape'])}, finite, repeat max|diff| {tv['repeat_max_abs_diff']:.1e}"
    def c4():
        sc = tv["speaker_check"]; assert sc["gap"] >= CFG["teacher_min_speaker_gap"]
        return f"speaker embedding separates speakers: same-speaker cos {sc['same_speaker_cos']:.3f} vs different {sc['different_speaker_cos']:.3f} | target: {spec_name(C['spec'])} (selection EER {tsel['chosen']['eer'] * 100:.3f}%)"
    def c5():
        assert tv["distinct_utterances"] and tv["equal_length_batch_vs_single_min_cos"] > 0.9999
        return f"distinct utterances (raw pairwise cos {tv['pairwise_cos_raw']['mean']:.3f}); equal-length batch == single (min cos {tv['equal_length_batch_vs_single_min_cos']:.6f}); no padding ever used"
    def c6():
        T = C["T_tr"]; n = np.linalg.norm(T, axis=1); assert np.isfinite(T).all() and (n > 0).all()
        u = l2n(T[:500]); c = u @ u.T; off = c[~np.eye(len(u), dtype=bool)]
        return f"{T.shape[1]}-D targets ({C['spec']['norm']}) | norm mean {n.mean():.1f} | pairwise cos mean {off.mean():.3f} (one constant vector reaches {np.linalg.norm(l2n(T).mean(0)):.3f})"
    def c7():
        st["S"], st["M"] = new_models(C, dev); st["info"] = inspect_student(st["S"], st["M"])
        return f"{st['info']['total_params'] / 1e6:.2f}M params, stacks {[s['layers'] for s in st['info']['stacks']]}, trainable (SV path + M_SV) {sum(p.numel() for p in st['S'].parameters() if p.requires_grad) / 1e6 + st['info']['msv_params'] / 1e6:.2f}M"
    def c8():
        rows = C["tr"].sort_values("duration_s").iloc[np.linspace(0, len(C["tr"]) - 1, 4).astype(int)].reset_index(drop=True); st["rows"] = rows
        idx = [C["row_tr"][u] for u in rows.uid]; b = sv_collate([SVData(C["tr"], C["T_tr"], CFG["crop_seconds"])[i] for i in idx]); st["b"] = b
        S = st["S"]; S.eval()
        with torch.no_grad():
            y, yl = S.sv_features(b["fb"].to(dev), b["fb_lens"].to(dev)); y2, _ = S.sv_features(b["fb"].to(dev), b["fb_lens"].to(dev), early_exit=False)
        S.train(); assert y.shape[-1] == 384 and yl.tolist() == [embed_frames(int(n)) for n in b["n"]], (tuple(y.shape), yl.tolist())
        d = float((y - y2).abs().max()); assert d < 1e-5, d
        return f"encoder.encoders[{S.sv_index}] output {tuple(y.shape)} (B, T50, 384), lengths {yl.tolist()} = embed frames; early exit == full forward (max diff {d:.1e})"
    def c9():
        S, M = st["S"], st["M"]; b = st["b"]; S.eval(); M.eval(); diffs = []
        with torch.no_grad():
            v = sv_forward(S, M, b, dev, amp)
            for k in range(len(b["n"])):
                L = int(b["fb_lens"][k]); vk = sv_forward(S, M, dict(fb=b["fb"][k:k + 1, :L], fb_lens=b["fb_lens"][k:k + 1]), dev, amp)
                diffs.append(float(torch.nn.functional.cosine_similarity(v[k:k + 1].float(), vk.float()).item()))
        S.train(); M.train(); assert v.shape == (len(b["n"]), C["dim"]) and min(diffs) > 0.999, diffs
        return f"M_SV output {tuple(v.shape)} | padded batch vs each utterance alone: min cosine {min(diffs):.6f}"
    def c10(): assert st["M"].D == C["dim"] == C["T_tr"].shape[1]; return f"student embedding {st['M'].D}-D == teacher {C['dim']}-D ({spec_name(C['spec'])})"
    def c11():
        S, M = st["S"], st["M"]; b = st["b"]; set_batch_count(S, 20000); v = sv_forward(S, M, b, dev, amp); loss, cos = kd_loss(v, b["t"].to(dev), CFG["cos_eps"])
        vn, tn = v.detach().double().cpu().numpy(), b["t"].double().numpy(); man = float(np.mean(1 - (vn * tn).sum(1) / (np.linalg.norm(vn, axis=1) * np.linalg.norm(tn, axis=1))))
        assert torch.isfinite(loss) and abs(float(loss) - man) < 1e-4, (float(loss), man); st["loss"] = loss
        return f"KD loss {float(loss):.5f} == numpy {man:.5f} (1 - cosine of un-normalised vectors) | cos {[round(float(c), 3) for c in cos]}"
    def c12():
        # fp16 (T4): backward through a SCALED loss, as training does with GradScaler; otherwise small gradients underflow to exactly 0
        k = 2.0 ** 12 if amp == torch.float16 else 1.0; (st["loss"] * k).backward(); S, M = st["S"], st["M"]; out = []
        for p in list(S.parameters()) + list(M.parameters()):
            if p.grad is not None: p.grad.div_(k)
        zero = [n for n, p in S.sv_params() if p.grad is not None and not bool(p.grad.any())]
        for g, mods in {"encoder_embed": [S.encoder_embed], **{f"stack{i + 1}": [S.encoder.encoders[i]] for i in range(S.sv_index + 1)}, "M_SV": [M]}.items():
            gs = [p.grad for m in mods for n, p in m.named_parameters() if p.grad is not None and not UNUSED_ICEFALL.search(n)]; assert gs, f"{g}: no gradient"
            nrm = float(torch.sqrt(sum((x.float() ** 2).sum() for x in gs))); assert all(torch.isfinite(x).all() for x in gs) and nrm > 0, g; out.append(f"{g} {nrm:.1e}")
        later = [n for n, p in S.named_parameters() if p.grad is not None and not any(n.startswith(f"encoder.encoders.{i}.") for i in range(S.sv_index + 1)) and not n.startswith("encoder_embed.")]
        assert not later, f"gradients outside the SV path: {later[:4]}"
        assert not zero, f"{len(zero)} tensors on the SV path have an all-zero gradient: {zero[:5]}"
        return " | ".join(out) + f" | every SV-path tensor has a non-zero gradient{' (loss scaled by 4096 for fp16, as GradScaler does)' if k > 1 else ''} | stacks {S.sv_index + 2}-6 receive no gradient (not on the SV path)"
    def c13():
        S, M = st["S"], st["M"]; opt, sch = make_optimizer(S, M, dict(CFG, **MODES["sanity"])); named = dict(S.sv_params() + [("msv." + n, p) for n, p in M.named_parameters()])
        before = {k: v.detach().clone() for k, v in named.items()}; frozen_before = {n: p.detach().clone() for n, p in S.named_parameters() if not p.requires_grad}
        unused = {k for k, v in named.items() if v.grad is None}; assert all(UNUSED_ICEFALL.search(k) for k in unused), sorted(unused)[:5]
        opt.step(); opt.zero_grad(set_to_none=True)
        changed = [k for k, v in named.items() if not torch.equal(v.detach(), before[k]) and k not in unused]; same = [k for k in named if k not in unused and k not in changed]
        assert len(same) <= 0.01 * len(named), f"{len(same)} trainable tensors not updated: {same[:5]}"
        fz = [n for n, p in S.named_parameters() if n in frozen_before and not torch.equal(p.detach(), frozen_before[n])]; assert not fz, fz[:3]
        return f"{type(opt).__name__}: {len(changed)}/{len(named) - len(unused)} trainable tensors updated; {len(frozen_before)} tensors of stacks {S.sv_index + 2}-6 unchanged"
    def c14():
        assert tv["frozen"]["trainable_params"] == 0 and tv["frozen"]["eval_mode"] and not st["b"]["t"].requires_grad
        return f"{tid['name']}: eval mode, 0 trainable parameters, run under no_grad; training uses CACHED teacher vectors (no teacher in the graph)"
    def c15(): assert torch.isfinite(st["loss"]); return "loss, embeddings and gradients finite (checks 11-12)"
    def c16():
        lens = np.array([crop_len(r.n_frames, r.sr, CFG["crop_seconds"]) for r in C["tr"].itertuples()]); B = make_batches(lens, 0, CFG["seed"], True, CFG["max_batch_seconds"], CFG["max_batch_utts"])
        flat = [i for b in B for i in b]; assert sorted(flat) == list(range(len(lens))), "batches do not cover every utterance exactly once"
        assert all(len(b) <= CFG["max_batch_utts"] and len(b) * max(lens[b]) / SR <= CFG["max_batch_seconds"] + 1e-6 for b in B)
        assert st["b"]["fb_lens"].tolist() == [fbank_frames(int(n)) for n in st["b"]["n"]]
        return f"{len(B)} length-sorted batches, every utterance once, <= {CFG['max_batch_seconds']:.0f} s / {CFG['max_batch_utts']} utts, fbank lengths = formula"
    def c17():
        S = st["S"]; b = st["b"]; S.eval(); ds = []
        with torch.no_grad():
            y, yl = S.sv_features(b["fb"].to(dev), b["fb_lens"].to(dev))
            for k in range(len(b["n"])):
                L = int(b["fb_lens"][k]); yk, ykl = S.sv_features(b["fb"][k:k + 1, :L].to(dev), b["fb_lens"][k:k + 1].to(dev)); n_ = int(ykl[0]) - 8
                ds.append(float((y[k, :n_] - yk[0, :n_]).abs().max() / yk[0, :n_].abs().max()))
        S.train(); assert max(ds) < 5e-3, ds
        return f"third-stack features of each utterance: padded batch == alone (max rel. diff {max(ds):.1e}; last 8 frames excluded: stack downsampling pairs the last frame with padding)"
    def c18():
        Ttr = EmbCache(tid, C["spec"], CFG["crop_seconds"]).load(C["tr"]); assert Ttr.shape[0] == len(C["tr"])
        return f"cache {C['cache_ids'][0]}: one valid embedding per utterance (train {len(C['tr']):,}, val {len(C['dv']):,}), shards sha1-verified, metadata = this teacher configuration"
    def c19():
        S, M = st["S"], st["M"]; f = paths()["logs"] / "sanity_ckpt.pt"; atomic_torch_save(dict(student=S.state_dict(), msv=M.state_dict()), f)
        ck = torch.load(f, map_location=dev, weights_only=False); S2, M2 = new_models(C, dev); S2.load_state_dict(ck["student"], strict=True); M2.load_state_dict(ck["msv"], strict=True)
        outs = []
        for a, m in [(S, M), (S2, M2)]:
            a.eval(); m.eval()
            with torch.no_grad(): outs.append(sv_forward(a, m, st["b"], dev, amp))
            a.train(); m.train()
        d = float((outs[0] - outs[1]).abs().max()); os.remove(f); assert d < 1e-5, d; return f"atomic save + strict reload, identical outputs (max diff {d:.1e})"
    def c21():
        r, _ = sv_evaluate(st["S"], st["M"], C, dev, amp, df=C["dv"].iloc[:64], targets=C["T_dv"][:64], trials=pd.DataFrame(columns=["label", "enrol", "test"]))
        assert np.isfinite(r["val_loss"]) and r["invalid_embeddings"] == 0; return f"validation pass on 64 utterances: KD loss {r['val_loss']:.4f}, cos {r['val_cos']:.4f}, no parameter update"
    def cM():
        mb = memory_probe(st["S"], st["M"], C, dev, amp, CFG["max_batch_seconds"]); return f"worst-case batch fits with max_batch_seconds {mb:.0f}"
    plan = [(1, "dataset manifest validation", c1), (2, "ReDimNet2 checkpoint loading", c2), (3, "ReDimNet2 forward pass", c3), (4, "teacher speaker embedding", c4),
            (5, "teacher batching without padding", c5), (6, "teacher embedding validation", c6), (7, "Zipformer-M initialisation", c7), (8, "third-block representation extraction", c8),
            (9, "student pooling / projection (M_SV)", c9), (10, "teacher / student dimensional compatibility", c10), (11, "KD loss calculation", c11),
            (12, "backpropagation", c12), (13, "student parameter update", c13), (14, "teacher remains frozen", c14), (15, "no NaN / Inf", c15),
            (16, "variable-length batching", c16), (17, "padding-mask correctness", c17), (18, "teacher cache integrity", c18), (19, "checkpoint save / reload", c19),
            (21, "validation pass", c21), (23, "GPU memory of the worst-case batch", cM)]
    for no, name, fn in plan:
        chk(no, name, fn)
        if no == 7 and R[-1]["status"] == "FAIL": break
    st.clear(); gc.collect(); torch.cuda.empty_cache()
    crit = [r["no"] for r in R if r["critical"] and r["status"] == "FAIL"]; missing = [n for n, _, _ in plan if n not in {r["no"] for r in R}]
    rep = dict(code_version=CODE_VERSION, checks=R, critical_failed=crit + missing, all_critical_passed=not crit and not missing, cache=C["cache_ids"], time=time.strftime("%Y-%m-%d %H:%M:%S"))
    atomic_json(rep, paths()["logs"] / "sanity_checks.json")
    log(f"\nSANITY CHECKS: {sum(r['status'] == 'PASS' for r in R)}/{len(plan)} passed | critical failures: {rep['critical_failed'] or 'none'} (20 = resume test and 22 = official teacher EER are separate steps)")
    return rep

def plots(P, drive=None):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ft, fv = P["logs"] / "training_log.csv", P["logs"] / "validation_log.csv"
        if not fv.exists() or fv.stat().st_size < 40: return
        V = pd.read_csv(fv); T = pd.read_csv(ft) if ft.exists() and ft.stat().st_size > 10 else None
        fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
        if T is not None and len(T): ax[0].plot(T["update"], T["loss"], alpha=0.5, label="train (logged steps)")
        ax[0].plot(V["update"], V["train_loss"], "o-", label="train (epoch)"); ax[0].plot(V["update"], V["val_loss"], "o-", label="validation"); ax[0].set(title="KD loss = 1 - cos", xlabel="global step"); ax[0].legend()
        ax[1].plot(V["update"], V["val_cos"], "o-", label="validation cos"); ax[1].plot(V["update"], V["val_cos_constant_baseline"], "--", c="gray", label="one constant vector"); ax[1].set(title="cosine with the teacher", xlabel="global step"); ax[1].legend()
        if "val_eer_nonofficial" in V: ax[2].plot(V["update"], V["val_eer_nonofficial"] * 100, "o-"); ax[2].set(title="EER % on NON-OFFICIAL validation trials (monitor)", xlabel="global step")
        if T is not None and len(T): ax[3].semilogy(T["update"], T["grad_norm"].clip(lower=1e-6), label="grad norm"); a3 = ax[3].twinx(); a3.plot(T["update"], T["lr"], c="tab:red"); ax[3].set(title="gradient norm / lr (red)", xlabel="global step")
        fig.tight_layout(); f = P["results"] / "training_curves.png"; fig.savefig(f, dpi=110); plt.close(fig)
        if drive: drive.write_file(f, "training_curves.png")
    except Exception as e: log(f"  (plot not updated: {type(e).__name__}: {e})")

def final_report(mode, C, P, drive, tsel, tid, teer, dsrep, sanity, rt, lc, compare=None, tv=None):
    tv_speaker = (tv or {}).get("speaker_check")
    import torch
    LATEST, BEST = P["latest"] / "checkpoint_latest.pt", P["best"] / "checkpoint_best.pt"
    ck = torch.load(LATEST, map_location="cpu", weights_only=False); S = ck["state"]; del ck
    rep = dict(RESULT_TYPE=("SANITY RUN (~2,000 utterances): a pipeline check, NOT a Stage 1 research result" if mode == "sanity" else "FULL Stage 1"),
               status=("COMPLETED: " + S["finished"]) if S["finished"] else f"NOT FINISHED (epoch {S['epoch'] + 1}, step {S['update']}): re-run the cell to resume",
               teacher=dict(model=tid["name"], model_id=tid["model_id"], repo=tid["repo"], commit=tid["commit"], params=tid["params"], training_data=tid["training_data"],
                            train_type=tid["train_type"], embedding_dim=C["dim"], representation=spec_name(C["spec"]), normalisation=C["spec"]["norm"],
                            normalisation_selection=dict(rule=tsel["rule"], candidates=tsel["candidates"], selection_eer_nonofficial=tsel["chosen"]["eer"]),
                            precision=tid["amp"], weights=tid["weight_files"], speaker_check=tv_speaker),
               student=dict(model="Zipformer-M (icefall, unchanged)", sv_branch=f"Conv2dSubsampling + encoder stacks 1-{CFG['sv_block']} (third stack: 3 layers, 384-D)",
                            msv="attentive pooling (two 1-D convolutions) -> attentive mean+std (768) -> Linear", embedding_dim=C["dim"]),
               objective="L_SV_KD = 1 - CosineSim(v_teacher, M_SV(R_student^(3))) (MT2KD Eq. 8); no speaker labels used in training",
               data=dict(dataset="VoxCeleb1", partition=dsrep["test_partition_source"], official=dsrep["official"], train_utterances=len(C["tr"]), train_speakers=int(C["tr"].speaker.nunique()),
                         validation_utterances=len(C["dv"]), validation_speakers=int(C["dv"].speaker.nunique()), validation_note=dsrep["validation"]["note"],
                         excluded_files=int(len(_ex := pd.read_csv(paths()["meta"] / "excluded_files.csv", dtype=str))),
                         excluded_reasons=_ex.reason.str.split(":").str[0].value_counts().to_dict(), excluded_list="excluded_files.csv"),
               training=dict(epochs=S["epoch"], optimizer_steps=S["update"], best=S["best"], early_stopping=dict(counter=S["es_counter"]), hours=round(S["elapsed"] / 3600, 3),
                             sessions=S["sessions"], resumes=S.get("resumes", []), peak_gpu_gib=S.get("peak_gpu_gib")),
               cache=dict(ids=C["cache_ids"], location=f"{CFG['drive_out']}/teacher_embedding_cache/", dim=C["dim"], dtype="float16"),
               sanity_checks=dict(passed=sanity["all_critical_passed"], failed=sanity["critical_failed"]), resume_test=dict(passed=rt.get("passed")), learning_check=lc,
               teacher_eer_official=teer, references_nonofficial=compare,
               limitations=["the teacher is ReDimNet2-B6 trained on VoxCeleb2-dev, not the paper's ECAPA-TDNN: results are not directly comparable with MT2KD Table VII",
                            "ReDimNet2 is recent (2026) with few independent reproductions; its official EER is measured here on the same trials as the student",
                            "the validation split is NON-OFFICIAL (held-out dev speakers); it was used to choose the target normalisation and for early stopping",
                            "KD targets and student inputs use a centred <= crop_seconds segment of each utterance; official EER uses full utterances",
                            "Zipformer step-dependent regularisation follows training progress (batch_count_mode='progress'), not icefall's absolute schedule",
                            "the paper's M_SV is described only as attentive pooling with two 1-D convolutions + a linear layer; the attentive mean+std form is an interpretation",
                            "VoxCeleb1-E/H trial lists are not usable (their speakers are training data here)",
                            "teacher embeddings are computed on the same centred <= crop_seconds segment as the student (training / validation); test uses full utterances"])
    if BEST.exists():
        dev = torch.device("cuda"); s, m = new_models(C, dev); b = torch.load(BEST, map_location=dev, weights_only=False); s.load_state_dict(b["student"]); m.load_state_dict(b["msv"])
        v, _ = sv_evaluate(s, m, C, dev, amp_dtype()); rep["best_checkpoint_validation"] = dict(epoch=b["epoch"], step=b["update"], val_kd_loss=v["val_loss"], val_cos=v["val_cos"],
                                                                                          val_eer_nonofficial=v.get("val_eer_nonofficial"), reload_matches=abs(v["val_loss"] - b["val"]["val_loss"]) < 2e-3)
        if S["finished"] and complete_trials():
            E = student_embeddings(s, m, C["te"], None, dev, amp_dtype(), max_sec=300.0); rep["student_eer_official"] = official_eer(E, C["te"], "STUDENT (best checkpoint)")
        if S["finished"]:
            le, lp = P["results"] / "student_encoder.pt", P["results"] / "sv_projection.pt"
            atomic_torch_save(dict(student=b["student"], zip_args=ZIP_ARGS, sv_block=CFG["sv_block"], epoch=b["epoch"], note="full Zipformer-M state; stacks after sv_block are untrained in SV-only Stage 1"), le)
            atomic_torch_save(dict(msv=b["msv"], dims=b["msv_dims"], teacher=b["teacher"], normalisation=dict(kind=C["spec"]["norm"], mu=C["mu"], sd=C["sd"], apply="(embedding - mu) / sd"), sv_block=CFG["sv_block"]), lp)
            if drive: drive.push(le, "student_encoder.pt"); drive.push(lp, "sv_projection.pt")
        del s, m; torch.cuda.empty_cache()
    atomic_json(rep, P["results"] / "final_report.json"); plots(P, drive)
    if drive: drive.write_json("final_report.json", rep); drive.wait()
    log("\n" + "=" * 100 + f"\nSV STAGE 1 FINAL REPORT  [{rep['RESULT_TYPE']}]\n" + "=" * 100)
    for k in ["status", "teacher", "student", "data", "training", "best_checkpoint_validation", "teacher_eer_official", "student_eer_official", "resume_test", "learning_check"]:
        if k in rep: log(f"{k:28s}: {rep[k]}")
    return rep
