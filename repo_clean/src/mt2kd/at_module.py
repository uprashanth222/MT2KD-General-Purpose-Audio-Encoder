# ====================================================================================================================
# MT2KD ENHANCEMENT — AUDIO-TAGGING TEACHER TEST: BEATs (current) vs CED-Base / CED-Small / CED-Mini vs SSLAM on FSD50K — SINGLE CELL, NO training
#
# WHY: before replacing BEATs (as we did for Whisper -> WavLM), measure on OUR data whether a newer AudioSet teacher is really better.
#   Every teacher scores the same 16 kHz, 10 s FSD50K crops (the AT2/AT3/AT v2 audio cache: MT2KD/cache16k) of the WHOLE official
#   validation split (4,170 clips; decision) and evaluation split (10,231 clips; confirmation). Metric: mAP over the FSD50K classes whose
#   AudioSet ID is among the 527 outputs: the same mapping and mAP code as the AT v2 cell (BEATs there: 50.29 % on val).
#   Paired bootstrap (100 resamples) gives a 95 % interval for each candidate's difference to BEATs.
# DECISION RULE: CED-Small if within ~1 mAP of the best candidate; otherwise the best candidate; keep BEATs if no candidate is
#   significantly better (95 % interval of the difference above 0).
# TEACHERS: BEATs = your fine-tuned AS2M cpt2 checkpoint on Drive (as in AT v2). CED (Apache-2.0) and SSLAM (MIT) are downloaded from
#   Hugging Face to local disk. A candidate that fails to load is skipped with a message (never fatal); its label order is checked.
# DRIVE: writes only MT2KD/results/AT_teacher_comparison/ (final_report.json, history.csv, config.json; a few KB). Everything else is read-only:
#   the audio crops are restored from MT2KD/cache16k when present, missing ones are built from the WAVs onto LOCAL disk only.
# TIME: ~45-75 min on a T4 (mostly restoring/building the eval-split audio the first time), ~25-40 min on an A100. Resumable.
# ====================================================================================================================
import os, sys, json, math, time, shutil, hashlib, subprocess, importlib, importlib.util, re, random, threading, itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

CODE_VERSION = "at-teacher-test-v2"
os.environ.setdefault("HF_HOME", "/content/hf_home")   # model downloads stay on local disk

CFG = dict(
    # ---- read-only inputs on Drive
    project_root="/content/drive/MyDrive/MT2KD",
    fsd_root="/content/drive/MyDrive/MT2KD/datasets/FSD50K",                 # FSD50K.dev_audio/, FSD50K.ground_truth/, processed/
    friend_at_root="/content/drive/MyDrive/MT2KD/outputs_AT_KD",               # AT1/AT2 outputs (class_mapping.csv, teacher_*.npz): reference only
    beats_search=["/content/drive/MyDrive/MT2KD/datasets/FSD50K", "/content/drive/MyDrive/MT2KD/datasets/FSD50K/checkpoint",
                  "/content/drive/MyDrive/MT2KD/checkpoints/BEATs", "/content/drive/MyDrive/MT2KD/checkpoints", "/content/drive/MyDrive/MT2KD"],
    unilm_repo="https://github.com/microsoft/unilm.git",                       # official BEATs model code (only the beats/ folder is fetched)
    # ---- outputs
    drive_out_root="/content/drive/MyDrive/MT2KD/results",                     # each run: <root>/AT_Stage1_v2_<run>/ (5 files)
    drive_api=True,
    drive_audio_cache="/content/drive/MyDrive/MT2KD/cache16k",                 # <split>_16k_c10.tar (same format as AT3): restored, and updated when clips are added
    save_audio_cache_to_drive=True,
    write_root="/content/MT2KD_at_teacher_test", local_audio="/content/fsd_cache16k", local_beats="/content/beats_local",
    unilm_dir="/content/unilm_beats", zipformer_dir="/content/MT2KD_zipformer_mt2kd",
    # ---- data (identical crop to AT2/AT3, so the cached reference teacher logits refer to the same audio)
    seed=42, max_seconds=10.0, cache_threads=16,
    # ---- teacher
    teacher_batch=32, teacher_match_tol=1e-3, require_teacher_match=True,      # new BEATs probabilities must equal AT2's cached ones within 1e-3
    # ---- student / KD
    n_classes=527, init_bias="teacher_prior",                                  # head bias starts at the mean teacher probability (label-free)
    # ---- optimisation: paper / icefall = ScaledAdam + Eden
    base_lr=0.045, lr_batches=7500, lr_epochs=3.5, eden_warmup_batches=500, eden_warmup_start=0.1, clipping_scale=2.0,
    max_batch_seconds=600.0, max_batch_utts=128, ref_duration=600.0, mem_fraction=0.85, max_consecutive_skips=25,
    max_session_hours=None,
    # ---- validation / early stopping / checkpoints
    patience=5, min_rel_delta=1e-3, eval_batch_seconds=600.0,
    local_ckpt_every_updates=100, drive_ckpt_every_updates=500, ckpt_every_minutes=15.0,
    log_every=50, num_workers=min(8, os.cpu_count() or 2),
)
PRESETS = {
    # quick gate (~1 h on an A100): the 2,000 training clips of AT3's smoke run (their audio and BEATs targets are already cached),
    # small batches -> ~100 steps/epoch, ~3,000 steps. Validation = the whole official FSD50K validation split (4,170 clips).
    "at_sanity": dict(n_train=2000, max_epochs=30, max_batch_seconds=120.0, patience=8, gate_ratio=0.85),
    # all 36,796 FSD50K training clips (~3-4 h on an A100 after ~1 h of one-time audio/teacher preparation)
    "at_full": dict(n_train=None, max_epochs=30, max_batch_seconds=600.0, patience=5),
}
RUN_PRESETS = ["at_sanity"]                          # add "at_full" to continue with all training clips (it starts only if at_sanity passes)
CFG.update(test_splits=["val"], n_bootstrap=100,                  # validation only (decision split); add "eval" for confirmation
           save_audio_cache_to_drive=False)   # test only: audio built here stays on local disk (no large tar written to Drive)   # official FSD50K validation (decision) + evaluation (confirmation) splits
OFFICIAL = {"train": 36796, "val": 4170, "eval": 10231}
SR = 16000
PREV_AT3 = dict(val_mAP_mapped=0.0540, eval_mAP_mapped=0.0433, teacher_val_mAP_mapped=0.5337, note="AT3 smoke run (AdamW, 250 steps, 186-class KD, 500 val clips)")
def drive_dir(preset): return f"{CFG['drive_out_root']}/AT_Stage1_v2_{preset}"

def paths(preset=None):
    W = Path(CFG["write_root"]); P = dict(meta=W / "meta", teacher=W / "teacher", logs=W / "logs", ckpt=W / "checkpoints", results=W / "results")
    if preset:
        for k in ("ckpt", "logs", "results"): P[k] = P[k] / preset
        P.update(best=P["ckpt"] / "best", latest=P["ckpt"] / "latest")
    return P
def ensure_dirs(P):
    for v in P.values():
        if isinstance(v, Path): assert_local(v); v.mkdir(parents=True, exist_ok=True)
def oom_error():
    import torch
    return getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
def code_fingerprint(): return CODE_VERSION

# ####################################################################################################
# ---- reused UNCHANGED from the verified ASR v4 cell: helpers, Zipformer-M + verified Swoosh port, fbank, batching, Drive store
# ####################################################################################################
def log(*a): print(*a, flush=True)

DRIVE = "/content/drive"

def assert_local(p):
    """Every LOCAL write goes through here: refuses anything on Google Drive (Drive writes exist only in class DriveStore)."""
    rp = os.path.realpath(str(p))
    if rp.startswith(DRIVE) or str(p).startswith(DRIVE): raise PermissionError(f"refusing to write to Google Drive: {p}")
    return p

def pip_if_missing(*mods):
    for m in mods:
        mod, pkg = (m, m) if isinstance(m, str) else m
        if importlib.util.find_spec(mod) is None: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

def atomic_json(obj, p):
    p = Path(assert_local(p)); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + ".tmp")
    json.dump(obj, open(tmp, "w"), indent=1, default=str); os.replace(tmp, p)

def atomic_torch_save(obj, p):
    import torch
    p = Path(assert_local(p)); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + ".tmp"); torch.save(obj, tmp); os.replace(tmp, p)

def sha_of(items): return hashlib.sha1("\n".join(map(str, items)).encode()).hexdigest()[:12]

def file_sha1(p, chunk=1 << 22):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()

def set_seed(s):
    import random, numpy as np, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def gpu_mem():
    import torch
    if not torch.cuda.is_available(): return {}
    return dict(gpu_alloc_gib=torch.cuda.memory_allocated() / 2**30, gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                gpu_reserved_gib=torch.cuda.memory_reserved() / 2**30, gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)

def disk_free_gib(p):
    p = Path(p)
    while not p.exists(): p = p.parent
    return shutil.disk_usage(p).free / 2**30

def require_space(path, need_gib, what, margin_gib=1.0):
    free = disk_free_gib(path)
    if need_gib > free - margin_gib: raise RuntimeError(f"not enough space for {what}: need {need_gib:.1f} GiB (+{margin_gib} margin), {free:.1f} GiB free at {path}")
    return free

D_S = 512

ZIP_ARGS = dict(output_downsampling_factor=2, downsampling_factor=(1, 2, 4, 8, 4, 2), num_encoder_layers=(2, 2, 3, 4, 3, 2),
                encoder_dim=(192, 256, 384, 512, 384, 256), encoder_unmasked_dim=(192, 192, 256, 256, 256, 192), query_head_dim=(32,) * 6,
                pos_head_dim=(4,) * 6, value_head_dim=(12,) * 6, num_heads=(4, 4, 4, 8, 4, 4), feedforward_dim=(512, 768, 1024, 1536, 1024, 768),
                cnn_module_kernel=(31, 31, 15, 15, 15, 31), pos_dim=48, causal=False)

def fbank_frames(n): return (n + 80) // 160

def embed_frames(n): return (fbank_frames(n) - 7) // 2

def student_frames(n): return (embed_frames(n) + 1) // 2

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
    o_ = open(f"{ZDIR}/optim.py").read()             # ScaledAdam / Eden, unchanged; lhotse is used only by its self-tests (fix_random_seed)
    if "from lhotse.utils import fix_random_seed" in o_:
        o_ = o_.replace("from lhotse.utils import fix_random_seed",
                        "def fix_random_seed(seed):  # MT2KD: replaces lhotse.utils.fix_random_seed (used only by optim.py's self-tests)\n"
                        "    import random as _r, numpy as _np; _r.seed(seed); _np.random.seed(seed); torch.manual_seed(seed)", 1)
        open(f"{ZDIR}/optim.py", "w").write(o_)
    if re.search(r"^\s*(from|import) lhotse", o_, re.M): raise RuntimeError("icefall optim.py imports lhotse in an unexpected place")
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

def amp_dtype():
    import torch
    return torch.bfloat16 if torch.cuda.get_device_properties(0).major >= 8 else torch.float16

DRIVE_FILES = ("best.pt", "latest.pt", "history.csv", "final_report.json", "config.json")

class SimulatedDisconnect(Exception):
    """Raised by the resume test to imitate a runtime that dies: nothing is saved at that moment."""

class DriveStore:
    """Writes ONLY into one folder (drive_dir(<run>)) and ONLY the five files in DRIVE_FILES (plus '<name>.tmp' while a write is in progress).
    * small files (config.json, history.csv, final_report.json): written synchronously, atomically (.tmp then rename)
    * checkpoints (latest.pt, best.pt): copied in a background thread from the local checkpoint (training does not wait),
      written to '<name>.tmp' first and renamed only when the copy is complete -> an interrupted upload never replaces a good file
    * with the Drive API (CFG['drive_api']): each upload is confirmed by comparing Drive's md5 with the local md5, and the copies that
      Drive moves to the trash when a checkpoint is replaced are deleted permanently (only files of THIS folder with THESE names),
      because trashed files keep using Drive quota for 30 days.
    fake=True: a local stand-in folder (used by the resume test; nothing goes to Drive)."""
    def __init__(self, folder, use_api=True, fake=False):
        self.dir, self.fake = Path(folder), fake
        if fake: assert_local(folder)
        else:
            if not os.path.isdir(DRIVE + "/MyDrive"): raise RuntimeError("Google Drive is not mounted")
            # the path as written must be inside MyDrive; its resolved location may be elsewhere on the Drive mount, because a folder
            # such as MT2KD can be a Drive SHORTCUT (resolved to /content/drive/.shortcut-targets-by-id/...)
            if not os.path.normpath(os.path.abspath(str(folder))).startswith(DRIVE + "/MyDrive/"): raise PermissionError(f"Drive output folder must be inside MyDrive: {folder}")
            if not os.path.realpath(str(folder)).startswith(os.path.realpath(DRIVE) + "/"): raise PermissionError(f"{folder} resolves outside Google Drive")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.api = self.fid = None; self.api_note = "not used (stand-in folder)" if fake else "disabled (CFG['drive_api'] = False)"
        if use_api and not fake: self._init_api()
        self.cv = threading.Condition(); self.pending = {}; self.busy = None; self.errors = []; self.done = {}; self._ids = itertools.count()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------------------------------------------------------------- guards / API
    def _p(self, name):
        base = name[:-4] if name.endswith(".tmp") else name
        if base not in DRIVE_FILES: raise PermissionError(f"'{name}' is not one of the files allowed on Drive: {DRIVE_FILES}")
        return self.dir / name
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
        if expected_bytes: log(f"  this run will keep  : about {g(expected_bytes)} in this folder (best.pt + latest.pt + small files)")
        unexpected = [k for k in fl if (k[:-4] if k.endswith(".tmp") else k) not in DRIVE_FILES]
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
        if self.api:
            self.done[name]["confirmed_by_drive_api"] = self._confirm(name, md5); self.purge_trash()
    def _confirm(self, name, md5, timeout=1800):
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
                if base in DRIVE_FILES: self.api.files().delete(fileId=x["id"]).execute(); freed += int(x.get("size", 0))
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
            if f.name[:-4] in DRIVE_FILES: f.unlink(); log(f"  removed an incomplete upload left by an interrupted run: {f.name}")
    def best_info(self): return best_meta(self._p("best.pt")) if self._p("best.pt").exists() else None

def drive_need_bytes(n_par, drive):
    """Drive space the run needs beyond what its own files already use: best.pt (fp32 weights) + latest.pt (weights + 2 AdamW moments),
    each written as a temporary copy before the old file is replaced (peak = 2 x each)."""
    best_b, latest_b = 4.05 * n_par, 20.0 * n_par          # latest.pt: weights + optimizer state (ScaledAdam keeps more per weight than AdamW)
    have = sum(v for k, v in drive.files().items() if k in ("best.pt", "latest.pt"))
    return max(0.0, 2 * (best_b + latest_b) - have) + 50e6, best_b + latest_b

def _position(S): return (S["update"], S["epoch"], len(S["seen"]))

# ####################################################################################################
# ---- environment
# ####################################################################################################
def env_report():
    import torch, platform
    r = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, gpu=None, code_version=CODE_VERSION)
    try:
        import torchaudio; r["torchaudio"] = torchaudio.__version__
    except Exception as e: r["torchaudio"] = f"missing ({e})"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0); r.update(gpu=p.name, gpu_gib=round(p.total_memory / 2**30, 1), amp="bfloat16" if p.major >= 8 else "float16 + GradScaler")
    r["disk_free_local_gib"] = round(disk_free_gib("/content" if Path("/content").exists() else "/"), 1)
    for k, v in r.items(): log(f"  {k:22s}: {v}")
    if r["gpu"] is None: raise RuntimeError("no GPU: Runtime > Change runtime type > A100 (or T4) GPU")
    return r

# ####################################################################################################
# ---- FSD50K: official splits, labels, vocabulary, 16 kHz 10 s centre-crop audio cache (same as AT2/AT3)
# ####################################################################################################
import numpy as np, pandas as pd

def load_fsd():
    """train / val (from FSD50K.dev) and eval: file ids, labels [N, 200], and the FSD50K vocabulary. Checked against the official sizes."""
    root = Path(CFG["fsd_root"]); proc = root / "processed"; gt = root / "FSD50K.ground_truth"
    for f in [proc / "train_metadata.csv", proc / "val_metadata.csv", proc / "y_train.npy", proc / "y_val.npy", gt / "vocabulary.csv"]:
        if not f.exists(): raise FileNotFoundError(f"{f} not found (expected the FSD50K folder used by AT2/AT3)")
    voc = pd.read_csv(gt / "vocabulary.csv", header=None, names=["idx", "label", "mid"])
    if list(voc.idx) != list(range(200)): raise RuntimeError("vocabulary.csv must list classes 0..199 in order")
    F = dict(vocab=voc, splits={})
    for name, sub in [("train", "FSD50K.dev_audio"), ("val", "FSD50K.dev_audio"), ("eval", "FSD50K.eval_audio")]:
        mf = proc / f"{name}_metadata.csv"
        if not mf.exists(): continue
        df = pd.read_csv(mf, dtype={"fname": str}); y = np.load(proc / f"y_{name}.npy") > 0.5
        if y.shape != (len(df), 200): raise RuntimeError(f"y_{name}.npy has shape {y.shape}, expected ({len(df)}, 200)")
        if len(df) != OFFICIAL[name]: raise RuntimeError(f"{name}: {len(df):,} clips, the official split has {OFFICIAL[name]:,}")
        F["splits"][name] = dict(fnames=df.fname.tolist(), y=y, wav=[str(root / sub / f"{f}.wav") for f in df.fname])
    a, b = set(F["splits"]["train"]["fnames"]), set(F["splits"]["val"]["fnames"])
    if a & b: raise RuntimeError(f"train and val share {len(a & b)} clips")
    return F

def subset_idx(n_total, n, offset):
    """Same rule as AT2/AT3: numpy default_rng(seed + offset).permutation, first n (None = all). offset 0 train, 100 val, 200 eval."""
    perm = np.random.default_rng(CFG["seed"] + offset).permutation(n_total); return [int(i) for i in (perm if n is None else perm[:n])]

def audio_dir(split): return Path(CFG["local_audio"]) / f"{split}_16k_c{int(CFG['max_seconds'])}"
def audio_tar(split): return Path(CFG["drive_audio_cache"]) / f"{split}_16k_c{int(CFG['max_seconds'])}.tar"

def wav_to_npy(src, dst):
    """Identical to AT2/AT3: mono mean -> torchaudio resample to 16 kHz -> centre crop to 10 s -> int16."""
    import soundfile as sf, torch, torchaudio
    x, sr0 = sf.read(src, dtype="float32", always_2d=True); x = x.mean(axis=1); w = torch.from_numpy(np.ascontiguousarray(x))
    if sr0 != SR: w = torchaudio.functional.resample(w, sr0, SR)
    MAXN = int(CFG["max_seconds"] * SR)
    if w.numel() > MAXN: s = (w.numel() - MAXN) // 2; w = w[s:s + MAXN]
    if w.numel() < 1600 or not bool(torch.isfinite(w).all()): raise ValueError(f"bad audio {src}")
    a = (w.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).numpy(); tmp = str(dst) + ".part"
    with open(tmp, "wb") as f: np.save(f, a)
    os.replace(tmp, dst)

def ensure_audio(F, split, idx):
    """Local 10 s crops for exactly these clips: restored from the Drive tar when possible, the rest built from the WAVs (resumable)."""
    S = F["splits"][split]; d = audio_dir(split); assert_local(d); d.mkdir(parents=True, exist_ok=True); fn = S["fnames"]
    for p in d.glob("*.part"): p.unlink()
    def missing():
        have = set(os.listdir(d)); return [i for i in sorted(set(idx)) if f"{fn[i]}.npy" not in have]
    need = missing()
    if need and audio_tar(split).exists():
        t0 = time.time(); log(f"  [{split}] restoring the audio cache from {audio_tar(split)} ...")
        subprocess.run(["tar", "-xf", str(audio_tar(split)), "-C", str(d)], capture_output=True); need = missing()
        log(f"  [{split}] restored in {time.time() - t0:.0f} s | still missing {len(need):,}/{len(set(idx)):,}")
    built, errs = 0, {}
    if need:
        require_space(d, len(need) * 0.33 / 1024, f"{split} audio cache")
        t0 = time.time(); log(f"  [{split}] building {len(need):,} clips from the WAVs on Drive ({CFG['cache_threads']} threads; slow the first time, resumable) ...")
        def work(i):
            for _ in range(3):
                try: wav_to_npy(S["wav"][i], d / f"{fn[i]}.npy"); return True
                except Exception as e: errs[i] = f"{type(e).__name__}: {e}"; time.sleep(1.0)
            return False
        with ThreadPoolExecutor(CFG["cache_threads"]) as ex:
            for b0 in range(0, len(need), 2000):
                built += sum(ex.map(work, need[b0:b0 + 2000])); log(f"  [{split}] {min(b0 + 2000, len(need)):,}/{len(need):,} | {time.time() - t0:.0f} s")
    bad = missing()
    if bad: raise RuntimeError(f"[{split}] {len(bad)} clips could not be read, e.g. {[(fn[i], errs.get(i)) for i in bad[:3]]}; re-run (it resumes)")
    if built and CFG["save_audio_cache_to_drive"]:
        tar = audio_tar(split); tar.parent.mkdir(parents=True, exist_ok=True); tmp = str(tar) + ".part"
        r = subprocess.run(["tar", "-cf", tmp, "-C", str(d), "."], capture_output=True, text=True)
        if r.returncode == 0: os.replace(tmp, tar); log(f"  [{split}] audio cache tar updated on Drive: {tar.name} ({len(os.listdir(d)):,} clips)")
        else: log(f"  [{split}] (could not write the cache tar: {r.stderr[-200:]})")
    log(f"  [{split}] local audio ready: {len(set(idx)):,} clips")

def clip_lengths(F, split, idx):
    """Sample counts of the cached clips (read from the .npy headers; remembered in a small local file)."""
    P = paths(); P["meta"].mkdir(parents=True, exist_ok=True); f = P["meta"] / f"lens_{split}.json"
    known = json.load(open(f)) if f.exists() else {}; fn = F["splits"][split]["fnames"]; d = audio_dir(split); new = 0
    for i in idx:
        if fn[i] not in known: known[fn[i]] = int(np.load(d / f"{fn[i]}.npy", mmap_mode="r").shape[0]); new += 1
    if new: atomic_json(known, f)
    return np.array([known[fn[i]] for i in idx], dtype=np.int64)

def load_clip(split, fname):
    a = np.load(audio_dir(split) / f"{fname}.npy"); return a.astype(np.float32) / 32768.0

# ####################################################################################################
# ---- BEATs teacher (official microsoft/unilm code, fine-tuned AS2M cpt2 checkpoint), frozen
# ####################################################################################################
def setup_beats_code():
    d = Path(CFG["unilm_dir"]); assert_local(d); bd = d / "beats"
    if not (bd / "BEATs.py").exists():
        shutil.rmtree(d, ignore_errors=True)
        cmd = (f"git clone -q --depth 1 --filter=blob:none --sparse {CFG['unilm_repo']} {d} && git -C {d} sparse-checkout set beats")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode or not (bd / "BEATs.py").exists(): raise RuntimeError(f"could not fetch the BEATs code from {CFG['unilm_repo']}: {r.stderr[-300:]}")
    if str(bd) not in sys.path: sys.path.append(str(bd))                      # appended: the Zipformer folder keeps priority
    importlib.invalidate_caches(); import BEATs as beats_mod
    try: commit = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception: commit = "unknown"
    return beats_mod, commit

def _ckpt_kind(path):
    """'finetuned' (has the 527-class predictor and label_dict), 'pretrained' (no predictor), or an error string."""
    import torch
    try: ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e: return f"unreadable ({type(e).__name__}: {str(e)[:120]})", None
    if not isinstance(ck, dict) or "model" not in ck or "cfg" not in ck: return "not a BEATs checkpoint (no 'model'/'cfg')", None
    pw = ck["model"].get("predictor.weight")
    if pw is None or not ck["cfg"].get("finetuned_model", False): return "pretrained", None
    if tuple(pw.shape) != (527, 768) or len(ck.get("label_dict", {})) != 527: return f"unexpected head {tuple(pw.shape)} / {len(ck.get('label_dict', {}))} labels", None
    return "finetuned", ck

def find_beats_checkpoint():
    """Searches the Drive folders in CFG['beats_search'] (one level deep) for BEATs*.pt / *.pt.zip / *.zip and returns the fine-tuned one."""
    cands = []
    for root in CFG["beats_search"]:
        r = Path(root)
        if not r.is_dir(): continue
        top = list(r.glob("*"))                                                # one level deeper only inside folders named like checkpoints/BEATs
        deeper = [q for d_ in top if d_.is_dir() and any(w in d_.name.lower() for w in ("checkpoint", "beats", "ckpt")) for q in d_.glob("*")]
        for p in top + deeper:
            n = p.name.lower()
            if p.is_file() and "beats" in n and (n.endswith(".pt") or n.endswith(".zip")) and "tokenizer" not in n: cands.append(p)
    cands = sorted(set(cands), key=lambda p: (0 if "finetuned" in p.name.lower() else 1, 0 if "cpt2" in p.name.lower() else 1, p.name))
    if not cands: raise FileNotFoundError(f"no BEATs checkpoint file found in {CFG['beats_search']}. Put BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt "
                                          f"(the 'Fine-tuned BEATs_iter3+ (AS2M) (cpt2)' link of github.com/microsoft/unilm/tree/master/beats) in {CFG['fsd_root']}")
    seen = []
    for p in cands:
        loc = Path(CFG["local_beats"]) / (p.name[:-4] if p.name.lower().endswith(".pt.zip") else p.name if p.name.endswith(".pt") else p.name + ".pt")
        loc.parent.mkdir(parents=True, exist_ok=True)
        if not loc.exists() or loc.stat().st_size != p.stat().st_size: shutil.copyfile(p, str(loc) + ".part"); os.replace(str(loc) + ".part", loc)
        kind, ck = _ckpt_kind(loc); seen.append(f"{p.name}: {kind}")
        if kind == "finetuned": return p, loc, ck
    raise RuntimeError("no FINE-TUNED BEATs checkpoint found. Files checked:\n  " + "\n  ".join(seen) +
                       "\n'pretrained' = BEATs_iter3_plus_AS2M.pt (no 527-class head). Download 'Fine-tuned BEATs_iter3+ (AS2M) (cpt2)' "
                       "(last column, last row of the table at github.com/microsoft/unilm/tree/master/beats) and upload it as ONE .pt file.")

class BEATsTeacher:
    """Frozen fine-tuned BEATs. logits(waves) = mean over patches of BEATs' own 527-class predictor (= the logit of BEATs' output probability)."""
    def __init__(self, device="cuda"):
        import torch
        self.mod, self.code_commit = setup_beats_code(); self.src, self.local, ck = find_beats_checkpoint()
        cfg = self.mod.BEATsConfig(ck["cfg"]); self.model = self.mod.BEATs(cfg); self.model.load_state_dict(ck["model"], strict=True)
        self.model = self.model.to(device).eval(); self.device = device
        for p in self.model.parameters(): p.requires_grad_(False)
        self.labels = [ck["label_dict"][i] for i in range(527)]; self.sha1 = file_sha1(self.local)
        self.info = dict(source=str(self.src), sha1=self.sha1, unilm_commit=self.code_commit, labels=527, params_M=round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1),
                         precision="float32", clip_logits="mean over patches of the 527-class predictor (BEATs' own head, eval mode)",
                         preprocessing="16 kHz waveform x 2^15 -> kaldi fbank 128 mel (25/10 ms) -> (x - 15.41663) / (2 x 6.55582)  [BEATs.preprocess]")
        del ck
        log(f"BEATs teacher loaded: {self.src.name} | {self.info['params_M']}M params | 527 classes | fp32, frozen, eval | unilm {self.code_commit[:10]} | sha1 {self.sha1[:10]}")
    def logits(self, waves):
        """waves: list of float32 arrays of EQUAL length (no padding, so a batch equals single-clip extraction) -> [B, 527] numpy."""
        import torch
        assert len({len(w) for w in waves}) == 1, "teacher batches must hold clips of equal length"
        x = torch.from_numpy(np.stack(waves)).to(self.device)
        with torch.inference_mode():
            pred = self.model.predictor; self.model.predictor = None
            try: feats, _ = self.model.extract_features(x, padding_mask=None)
            finally: self.model.predictor = pred
            out = pred(feats).mean(dim=1)
        if not torch.isfinite(out).all(): raise FloatingPointError("BEATs produced non-finite logits")
        return out.float().cpu().numpy()
    def logits_many(self, split, fnames):
        """Clips grouped by identical length (most FSD50K crops are exactly 10 s) and processed in batches."""
        lens = {}; out = {}
        for f in fnames: lens.setdefault(int(np.load(audio_dir(split) / f"{f}.npy", mmap_mode="r").shape[0]), []).append(f)
        for n, group in sorted(lens.items()):
            for s in range(0, len(group), CFG["teacher_batch"]):
                g = group[s:s + CFG["teacher_batch"]]; lg = self.logits([load_clip(split, f) for f in g])
                for f, row in zip(g, lg): out[f] = row
        return out

def teacher_store(split): return paths()["teacher"] / f"beats_{split}.npz"
def build_teacher_logits(T, F, split, idx):
    """Resumable local store of BEATs clip logits for the selected clips (float32 [N, 527]); saved every 2,000 clips."""
    f = teacher_store(split); f.parent.mkdir(parents=True, exist_ok=True); fn = F["splits"][split]["fnames"]
    have = {}
    if f.exists():
        z = np.load(f); meta = json.loads(str(z["meta"]))
        if meta.get("teacher_sha1") == T.sha1: have = dict(zip(z["uids"].tolist(), z["logits"]))
        else: log(f"  [{split}] existing teacher store was made with another checkpoint: recomputed")
    todo = [fn[i] for i in idx if fn[i] not in have]; t0 = time.time()
    if todo: log(f"  [{split}] BEATs logits: {len(have):,} stored, computing {len(todo):,} ...")
    def save():
        u = sorted(have); tmp = str(f) + ".tmp.npz"
        np.savez(tmp, uids=np.array(u), logits=np.stack([have[k] for k in u]).astype(np.float32), meta=json.dumps(dict(teacher_sha1=T.sha1, split=split)))
        os.replace(tmp, f)
    for s in range(0, len(todo), 2000):
        have.update(T.logits_many(split, todo[s:s + 2000])); save()
        log(f"  [{split}] {min(s + 2000, len(todo)):,}/{len(todo):,} | {(min(s + 2000, len(todo))) / max(time.time() - t0, 1e-9):.0f} clips/s")
    if not f.exists(): save()
    return f

def load_teacher_logits(split, fnames):
    z = np.load(teacher_store(split)); pos = {u: i for i, u in enumerate(z["uids"].tolist())}; miss = [u for u in fnames if u not in pos]
    if miss: raise RuntimeError(f"[{split}] {len(miss)} clips have no BEATs logits (e.g. {miss[:3]}): run the teacher step")
    return z["logits"][[pos[u] for u in fnames]].astype(np.float32), json.loads(str(z["meta"]))

def compare_with_at2(T, F, n_per_split=64):
    """New BEATs logits vs the logits cached by your friend's AT2 run on the same clips (same 10 s crop)."""
    rows = {}
    for split in ["train", "val", "eval"]:
        f = Path(CFG["friend_at_root"]) / f"teacher_{split}.npz"
        if not f.exists() or split not in F["splits"]: continue
        z = np.load(f); uids = z["uids"].tolist()[:n_per_split]; ref = z["logits"][:n_per_split].astype(np.float32)
        if not all((audio_dir(split) / f"{u}.npy").exists() for u in uids):
            ensure_audio(F, split, [F["splits"][split]["fnames"].index(u) for u in uids])
        new = T.logits_many(split, uids); mine = np.stack([new[u] for u in uids])
        sig = lambda x: 1 / (1 + np.exp(-x))
        rows[split] = dict(clips=len(uids), max_prob_diff=float(np.abs(sig(mine) - sig(ref)).max()), mean_prob_diff=float(np.abs(sig(mine) - sig(ref)).mean()),
                           max_logit_diff=float(np.abs(mine - ref).max()), top1_agreement=float((mine.argmax(1) == ref.argmax(1)).mean()))
        log(f"  [{split}] {len(uids)} clips: max |prob diff| {rows[split]['max_prob_diff']:.2e} | max |logit diff| {rows[split]['max_logit_diff']:.3f} | "
            f"top-1 class agreement {rows[split]['top1_agreement'] * 100:.0f}%")
    ok = bool(rows) and all(r["max_prob_diff"] < CFG["teacher_match_tol"] for r in rows.values())
    return dict(splits=rows, passed=ok, available=bool(rows))

# ####################################################################################################
# ---- FSD50K class mapping (for EVALUATION only): FSD50K index -> BEATs output index, exact AudioSet ID match
# ####################################################################################################
def build_mapping(F, labels):
    voc = F["vocab"]; a_of = {m: i for i, m in enumerate(labels)}
    rows = [dict(fsd_index=int(r.idx), fsd_label=r.label, fsd_mid=r.mid, beats_index=a_of.get(r.mid, -1)) for r in voc.itertuples()]
    M = pd.DataFrame(rows); M["mapped"] = M.beats_index >= 0
    ref = Path(CFG["friend_at_root"]) / "mapping" / "class_mapping.csv"; cmp = None
    if ref.exists():
        R = pd.read_csv(ref); Ru = R[R.used_for_kd.astype(bool)]
        mine = set(zip(M[M.mapped].fsd_index, M[M.mapped].beats_index)); theirs = set(zip(Ru.fsd_index.astype(int), Ru.audioset_index.astype(int)))
        cmp = dict(identical=mine == theirs, only_new=sorted(mine - theirs), only_at1=sorted(theirs - mine))
    log(f"  mapping: {int(M.mapped.sum())} of 200 FSD50K classes have their AudioSet ID among BEATs' 527 outputs; unmapped: "
        + ", ".join(M[~M.mapped].fsd_label) + (f" | identical to AT1's class_mapping.csv: {cmp['identical']}" if cmp else " | (AT1 mapping file not found)"))
    return M, cmp


# ####################################################################################################
# ---- AT TEACHER COMPARISON: BEATs (current) vs CED-Base / CED-Small / CED-Mini vs SSLAM, on FSD50K, NO training
# ####################################################################################################
# Every teacher scores the SAME 10 s, 16 kHz clips (the AT2/AT3/AT v2 audio cache). Score = mAP over the FSD50K classes whose AudioSet ID
# is one of the 527 AudioSet outputs (the same class mapping and the same mAP code as the AT v2 cell; BEATs' val result there: 50.29 %).
# Each candidate's 527 outputs are assumed to follow the standard AudioSet order (as BEATs' label_dict); this is CHECKED twice:
#   (1) its own label names vs the FSD50K names of the mapped classes, (2) per-class correlation with BEATs' outputs over all clips.
# A candidate whose order cannot be confirmed is reported but NOT ranked. A candidate that cannot be loaded is skipped (logged), never fatal.
SIG = lambda x: 1.0 / (1.0 + np.exp(-x))

def mean_ap(y, p):
    from sklearn.metrics import average_precision_score
    has = y.sum(0) > 0; ap = np.full(y.shape[1], np.nan)
    for j in np.where(has)[0]: ap[j] = average_precision_score(y[:, j], p[:, j])
    return float(np.nanmean(ap)), ap

def _fast_map(y, p):
    """Vectorised average precision per class (non-interpolated; equals sklearn's value when scores have no ties). Classes without positives skipped."""
    o = np.argsort(-p, axis=0, kind="stable"); ys = np.take_along_axis(y.astype(np.float64), o, axis=0)
    tp = np.cumsum(ys, 0); prec = tp / np.arange(1, len(y) + 1)[:, None]; npos = ys.sum(0); keep = npos > 0
    return float(((prec * ys).sum(0)[keep] / npos[keep]).mean())

def paired_bootstrap(y, p_a, p_b, n=100, seed=0):
    """mAP(a) - mAP(b) on resampled clip sets -> (mean, 2.5 %, 97.5 %). Same resamples for both models (paired)."""
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); yy = y[i]
        if not (yy.sum(0) > 0).any(): continue
        d.append(_fast_map(yy, p_a[i]) - _fast_map(yy, p_b[i]))
    d = np.array(d); return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def _norm_name(s):
    s = str(s).lower().replace("_", " ")
    for w in (" and ", ",", "(", ")", "'", "-", ".", "/"): s = s.replace(w, " ")
    return "".join(s.split())

def _hf_compat_shims():
    """Remote model code written for transformers 4.x imports a few helpers that 5.x may have renamed/removed: add no-op/alias versions if missing."""
    import transformers, transformers.utils as tu
    noop = lambda *a, **k: (lambda f: f)
    for n in ("add_code_sample_docstrings", "add_start_docstrings", "add_start_docstrings_to_model_forward", "replace_return_docstrings"):
        if not hasattr(tu, n): setattr(tu, n, noop)
    try:
        from transformers import PreTrainedModel
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"): PreTrainedModel.all_tied_weights_keys = {}
    except Exception: pass
    if not hasattr(transformers, "PretrainedConfig") and hasattr(transformers, "PreTrainedConfig"):
        transformers.PretrainedConfig = transformers.PreTrainedConfig
        import transformers.configuration_utils as cu
        if not hasattr(cu, "PretrainedConfig"): cu.PretrainedConfig = transformers.PreTrainedConfig

BEATS_MIDS = None          # set when BEATs is loaded: the AudioSet IDs of BEATs' 527 outputs (the reference order)

def load_weights_from_file(model, repo, fname="model.safetensors"):
    """Independent of transformers' loader (old remote code + transformers 5 can silently leave weights wrong): reads the checkpoint
    file, matches tensors to the model by name (trying common prefixes), reports how many were ALREADY correct, then copies them all in."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download(repo, fname)); msd = model.state_dict()
    pref = {""} | {k.split(".")[0] + "." for k in sd} | {k.split(".")[0] + "." for k in msd}
    best = ({}, "", "")
    for strip in pref:
        for add in pref:
            m = {}
            for k in sd:
                if strip and not k.startswith(strip): continue
                kk = add + k[len(strip):]
                if kk in msd and tuple(msd[kk].shape) == tuple(sd[k].shape): m[kk] = k
            if len(m) > len(best[0]): best = (m, strip, add)
    m = best[0]; cover = len(m) / max(len(msd), 1)
    already = sum(1 for kk, k in m.items() if torch.equal(msd[kk].detach().cpu().to(sd[k].dtype), sd[k]))
    with torch.no_grad():
        for kk, k in m.items(): msd[kk].copy_(sd[k].to(msd[kk].device, msd[kk].dtype))
    missing = sorted(set(msd) - set(m))
    log(f"    weights: {len(m)}/{len(msd)} model tensors matched in {fname} (prefix '{best[1]}' -> '{best[2]}'); {already} were already correct "
        f"after transformers' loading, {len(m) - already} were WRONG and are now fixed" + (f" | not in the file: {missing[:4]}" if missing else ""))
    if cover < 0.95: raise RuntimeError(f"only {cover:.0%} of the model's tensors found in the checkpoint file: weights cannot be trusted")
    return dict(matched=len(m), total=len(msd), already_correct=already, fixed=len(m) - already, missing=missing[:20])

AUDIOSET_CSV_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"   # official AudioSet class list

def model_class_names(repo):
    """The 527 class names exactly as written in the model's own config.json (read as a file, not through transformers)."""
    from huggingface_hub import hf_hub_download
    c = json.load(open(hf_hub_download(repo, "config.json"))); i2l = c.get("id2label") or {}
    return [i2l.get(str(i), i2l.get(i, "")) for i in range(527)] if len(i2l) >= 527 else None

def audioset_names_to_mids():
    """display_name -> AudioSet ID from the official class list (None if it cannot be downloaded)."""
    try:
        A = pd.read_csv(AUDIOSET_CSV_URL); return dict(zip(A.display_name, A.mid)), list(A.mid)
    except Exception as e:
        log(f"    (official AudioSet class list not downloadable: {type(e).__name__}; the class order is then checked from the outputs only)"); return None, None

class BEATsCandidate:
    """The current teacher, exactly as in the AT v2 cell (fp32, mean over patches of the fine-tuned AS2M cpt2 head)."""
    name = "BEATs iter3+ AS2M cpt2 (current)"; key = "beats"
    def __init__(self):
        global BEATS_MIDS
        self.T = BEATsTeacher(); self.labels_mid = self.T.labels; self.names = None; self.params_M = self.T.info["params_M"]; self.outputs = "logits"
        BEATS_MIDS = list(self.T.labels)
    def scores(self, split, fnames): return self.T.logits_many(split, fnames)
    def release(self): self.T.model.cpu(); del self.T

class CEDCandidate:
    """mispeech/ced-* (Apache-2.0). Front end computed HERE with CED's own recipe (torchaudio MelSpectrogram: 16 kHz, n_fft 512, win 512,
    hop 160, 64 mels, f 0-8000 Hz, center -> AmplitudeToDB top_db 120, per clip), so it does not depend on the remote feature-extractor code.
    Weights are re-checked against model.safetensors; the 527 output columns are put in BEATs' order via AudioSet IDs (class_labels_indices.csv)."""
    def __init__(self, repo, name, key):
        import torch, torchaudio
        pip_if_missing("transformers", "safetensors"); _hf_compat_shims()
        from transformers import AutoModelForAudioClassification, AutoConfig
        self.name, self.key, self.repo = name, key, repo
        self.model = AutoModelForAudioClassification.from_pretrained(repo, trust_remote_code=True).eval()
        self.wcheck = load_weights_from_file(self.model, repo); self.model = self.model.to("cuda").eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        cfg = self.model.config; g = lambda k, d: getattr(cfg, k, d) if getattr(cfg, k, None) is not None else d
        self.params_M = round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=int(g("n_fft", 512)), win_length=int(g("win_size", 512)), hop_length=int(g("hop_size", 160)),
                                                        f_min=float(g("f_min", 0)), f_max=float(g("f_max", 8000)), n_mels=int(g("n_mels", 64)), center=bool(g("center", True))).to("cuda")
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=120).to("cuda")
        tl = g("target_length", None); self.max_samples = (int(tl) - 1) * int(g("hop_size", 160)) if tl else None
        names = model_class_names(repo); self.perm = np.arange(527); self.names = names     # native order = standard AudioSet index order
        log(f"    class order: standard AudioSet index order (first names {names[:3] if names else None}); re-ordered to BEATs' classes by AudioSet ID in STEP 4")
        self.pooling = g("pooling", "?"); self.outputs = "?"; self.fe_check = None
        log(f"  {name}: {repo} | {self.params_M}M params | pooling '{self.pooling}' | target_length {tl} | class list in BEATs' order already: "
            f"{bool((self.perm == np.arange(527)).all())} | own mel front end | fp32, eval")
    def _feats(self, waves):
        import torch
        if self.max_samples and len(waves[0]) > self.max_samples: waves = [w[:self.max_samples] for w in waves]   # stay within ONE CED chunk
        x = torch.from_numpy(np.stack(waves)).to("cuda")
        return self.db(self.mel(x).unsqueeze(1)).squeeze(1)              # [B, 64, T]; dB range clipped per clip (top_db 120)
    def _run(self, waves):
        import torch
        x = self._feats(waves)
        if self.fe_check is None:                                         # one-time comparison with the model's own feature extractor (information only)
            try:
                from transformers import AutoFeatureExtractor
                fe = AutoFeatureExtractor.from_pretrained(self.repo, trust_remote_code=True)
                y = fe([waves[0]], sampling_rate=SR, return_tensors="pt")["input_values"][0].to("cuda")
                self.fe_check = f"shape {tuple(y.shape)} vs {tuple(x[0].shape)}, max |diff| {float((y[..., :x.shape[-1]] - x[0][..., :y.shape[-1]]).abs().max()):.4f}" if y.shape[0] == x.shape[1] else f"shape {tuple(y.shape)} vs {tuple(x[0].shape)}"
            except Exception as e: self.fe_check = f"not available ({type(e).__name__})"
            log(f"    front-end check vs the repository's feature extractor: {self.fe_check}")
        with torch.inference_mode(): out = self.model(input_values=x).logits.float()
        if out.dim() == 3: out = out.mean(1)
        if out.shape[-1] != 527 or not torch.isfinite(out).all(): raise RuntimeError(f"unexpected output {tuple(out.shape)}")
        return out.cpu().numpy()[:, self.perm]
    def scores(self, split, fnames):
        lens = {}; out = {}
        for f in fnames: lens.setdefault(int(np.load(audio_dir(split) / f"{f}.npy", mmap_mode="r").shape[0]), []).append(f)
        for n, group in sorted(lens.items()):
            for s in range(0, len(group), CFG["teacher_batch"]):
                g = group[s:s + CFG["teacher_batch"]]
                for f, row in zip(g, self._run([load_clip(split, f) for f in g])): out[f] = row
        v = np.stack(list(out.values()))
        self.outputs = "probabilities (sigmoid already applied)" if (v.min() >= 0 and v.max() <= 1) else "logits"
        return out
    def release(self): self.model.cpu(); del self.model

class SSLAMCandidate:
    """ta012/SSLAM_AS2M_Finetuned (MIT). Model-card preprocessing: waveform - mean -> kaldi fbank 128 mel (hanning, 10 ms) ->
    pad/cut to 1024 frames -> (x + 4.268) / (4.569 x 2) -> [B, 1, T, F]; outputs 527 logits."""
    name = "SSLAM AS2M fine-tuned"; key = "sslam"
    def __init__(self, repo="ta012/SSLAM_AS2M_Finetuned"):
        import torch
        pip_if_missing("transformers", "timm"); _hf_compat_shims()
        from transformers import AutoModel
        self.repo = repo; self.model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()
        self.wcheck = load_weights_from_file(self.model, repo); self.model = self.model.to("cuda").eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        cfg = self.model.config; i2l = getattr(cfg, "id2label", None) or {}
        self.names = [i2l.get(i, i2l.get(str(i), "")) for i in range(527)] if len(i2l) >= 527 else None
        self.labels_mid = None; self.params_M = round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1); self.outputs = "logits"
        log(f"  {self.name}: {repo} | {self.params_M}M params | {len(i2l)} label names | fp32, eval")
    def _feat(self, w):
        import torch, torchaudio
        x = torch.from_numpy(w).float(); x = x - x.mean()
        m = torchaudio.compliance.kaldi.fbank(x.unsqueeze(0), htk_compat=True, sample_frequency=SR, use_energy=False, window_type="hanning",
                                              num_mel_bins=128, dither=0.0, frame_shift=10)
        T = 1024
        m = torch.nn.functional.pad(m, (0, 0, 0, T - m.shape[0])) if m.shape[0] < T else m[:T]
        return (m + 4.268) / (4.569 * 2)
    def scores(self, split, fnames):
        import torch
        out = {}
        for s in range(0, len(fnames), CFG["teacher_batch"]):
            g = fnames[s:s + CFG["teacher_batch"]]; x = torch.stack([self._feat(load_clip(split, f)) for f in g]).unsqueeze(1).to("cuda")
            with torch.inference_mode(): o = self.model(x)
            lg = getattr(o, "logits", o if torch.is_tensor(o) else o[0]).float()
            if lg.dim() == 3: lg = lg.mean(1)
            if lg.shape[-1] != 527 or not torch.isfinite(lg).all(): raise RuntimeError(f"unexpected output {tuple(lg.shape)}")
            for f, row in zip(g, lg.cpu().numpy()): out[f] = row
        return out
    def release(self): self.model.cpu(); del self.model

CANDIDATES = [  # (key, display name, constructor)
    ("beats", "BEATs iter3+ AS2M cpt2 (current)", lambda: BEATsCandidate()),
    ("ced_base", "CED-Base (86M)", lambda: CEDCandidate("mispeech/ced-base", "CED-Base (86M)", "ced_base")),
    ("ced_small", "CED-Small (22M)", lambda: CEDCandidate("mispeech/ced-small", "CED-Small (22M)", "ced_small")),
    ("ced_mini", "CED-Mini (9.6M)", lambda: CEDCandidate("mispeech/ced-mini", "CED-Mini (9.6M)", "ced_mini")),
    ("sslam", "SSLAM AS2M fine-tuned (88M)", lambda: SSLAMCandidate()),
]

def score_store(key, split): return paths()["teacher"] / (f"scores_{key}_{split}.npz" if key == "beats" else f"scores_v2_{key}_{split}.npz")

def candidate_scores(key, cand, split, fnames):
    """Resumable local store of one candidate's 527 outputs for every clip of the split."""
    import torch
    f = score_store(key, split); f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        z = np.load(f, allow_pickle=False); u = z["uids"].tolist()
        if u == list(fnames): return z["scores"].astype(np.float32), json.loads(str(z["meta"]))
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); S = {}
    for s in range(0, len(fnames), 2000):
        S.update(cand.scores(split, fnames[s:s + 2000])); log(f"    [{split}] {min(s + 2000, len(fnames)):,}/{len(fnames):,} clips | {time.time() - t0:.0f} s")
    sc = np.stack([S[u] for u in fnames]).astype(np.float32); sec = time.time() - t0
    meta = dict(outputs=cand.outputs, seconds=round(sec, 1), clips_per_s=round(len(fnames) / max(sec, 1e-9), 1), gpu_peak_gib=round(gpu_mem().get("gpu_peak_gib", 0), 2))
    tmp = str(f) + ".tmp.npz"; np.savez(tmp, uids=np.array(fnames), scores=sc, meta=json.dumps(meta)); os.replace(tmp, f)
    return sc, meta

def check_label_order(cand, sc, ref_beats, M):
    """(1) names of the mapped classes vs FSD50K names, (2) for each output class, is the most correlated BEATs output the SAME index?"""
    R = {}
    mm = M[M.mapped]
    if cand.names:
        ok = [_norm_name(cand.names[a]) == _norm_name(l) for a, l in zip(mm.beats_index, mm.fsd_label)]
        R["name_match_mapped"] = float(np.mean(ok))
    else: R["name_match_mapped"] = None
    if ref_beats is not None:
        a = sc - sc.mean(0); b = ref_beats - ref_beats.mean(0)
        a /= (np.linalg.norm(a, axis=0) + 1e-12); b /= (np.linalg.norm(b, axis=0) + 1e-12)
        C = a.T @ b                                                    # [527 candidate, 527 BEATs] correlations over all clips
        idx = np.asarray(mm.beats_index)                               # the classes that are scored (mapped FSD50K classes)
        R["corr_argmax_same_index"] = float((C[idx].argmax(1) == idx).mean())
        R["mean_diag_corr"] = float(C[idx, idx].mean())
        R["median_best_corr_any_class"] = float(np.median(C[idx].max(1)))   # high but diag low = shuffled class order; both low = outputs meaningless
        R["top1_agreement_with_beats"] = float((sc.argmax(1) == ref_beats.argmax(1)).mean())
    else: R["corr_argmax_same_index"] = None; R["mean_diag_corr"] = None
    R["order_confirmed"] = bool((R["mean_diag_corr"] or 0) >= 0.3 or (R["corr_argmax_same_index"] or 0) >= 0.5)   # outputs must track the same classes as BEATs
    return R


