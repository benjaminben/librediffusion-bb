#!/usr/bin/env python3
"""
Smoke test for combined UNet+ControlNet engines.

Loads librediffusion via ctypes, builds a pipeline + CLIP, runs one img2img
inference against the committed source/canny fixtures, and writes the
result. Auto-resizes fixtures to match the engine's resolution so a single
fixture set works against any 512^2 / 1024^2 / etc. engine.

Default fixtures live in tools/test_data/. Engines must already be built
(see tools/export_combined_unet_controlnet.py).

Examples:
    # smoke-test a freshly built combined engine
    python tools/test_engine.py \\
        --engine engines/unet_controlnet_canny.engine \\
        --vae-encoder engines/vae_encoder.engine \\
        --vae-decoder engines/vae_decoder.engine \\
        --clip engines/clip.engine

    # verify the engine actually consumes the control image
    python tools/test_engine.py \\
        --engine engines/unet_controlnet_canny.engine \\
        --vae-encoder engines/vae_encoder.engine \\
        --vae-decoder engines/vae_decoder.engine \\
        --clip engines/clip.engine \\
        --control tools/test_data/canny_of_different.png \\
        --output ./test_output_oo_d.png
"""

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURE_DIR = HERE / "test_data"

# ----- librediffusion C API surface used here ----------------------------------
# Error code 0 = success. We let any non-zero return raise RuntimeError.
ERR_OK = 0


def _find_dll() -> Path:
    """Locate librediffusion.{dll,so,dylib} starting from the repo root."""
    candidates = []
    repo_root = HERE.parent
    for ext in ("dll", "so", "dylib"):
        candidates.append(repo_root / "build" / f"librediffusion.{ext}")
        candidates.append(repo_root / "build" / "Release" / f"librediffusion.{ext}")
        candidates.append(repo_root / "build" / "Debug" / f"librediffusion.{ext}")
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find librediffusion shared library under build/. "
        "Build the library first (see CMakeLists.txt)."
    )


def _bind(lib: ctypes.CDLL, name: str, restype, argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


def _add_dll_search_dirs() -> None:
    """Windows + Python 3.8+: ctypes.CDLL no longer falls back to PATH for
    transitive dependencies. We must register each directory containing a
    dep DLL via os.add_dll_directory() before loading librediffusion.dll.

    Reads colon/semicolon-separated dirs from LIBREDIFFUSION_DLL_DIRS, plus
    common CUDA/TensorRT defaults. Missing dirs are silently skipped.
    """
    if not hasattr(os, "add_dll_directory"):
        return  # non-Windows or older Python
    candidates: list[str] = []
    env = os.environ.get("LIBREDIFFUSION_DLL_DIRS", "")
    if env:
        candidates.extend(env.replace(";", os.pathsep).split(os.pathsep))
    # Standard CUDA 13 + TensorRT install locations (best-effort guesses).
    candidates += [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64",
        r"C:\Users\Haunter\src\TensorRT-10.16.1.11\bin",
    ]
    seen = set()
    for d in candidates:
        d = d.strip()
        if not d or d in seen:
            continue
        seen.add(d)
        if Path(d).is_dir():
            os.add_dll_directory(d)


def load_lib(dll_path: Path) -> dict:
    _add_dll_search_dirs()
    lib = ctypes.CDLL(str(dll_path))
    f = {}

    cfg_h = ctypes.c_void_p
    pipe_h = ctypes.c_void_p
    clip_h = ctypes.c_void_p
    half_p = ctypes.POINTER(ctypes.c_uint16)

    # config
    f["config_create"] = _bind(lib, "librediffusion_config_create", ctypes.c_int,
                               [ctypes.POINTER(cfg_h)])
    f["config_destroy"] = _bind(lib, "librediffusion_config_destroy", None, [cfg_h])
    f["config_set_device"] = _bind(lib, "librediffusion_config_set_device",
                                   ctypes.c_int, [cfg_h, ctypes.c_int])
    f["config_set_model_type"] = _bind(lib, "librediffusion_config_set_model_type",
                                       ctypes.c_int, [cfg_h, ctypes.c_int])
    f["config_set_dimensions"] = _bind(lib, "librediffusion_config_set_dimensions",
                                       ctypes.c_int,
                                       [cfg_h, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int])
    f["config_set_batch_size"] = _bind(lib, "librediffusion_config_set_batch_size",
                                       ctypes.c_int, [cfg_h, ctypes.c_int])
    f["config_set_denoising_steps"] = _bind(lib, "librediffusion_config_set_denoising_steps",
                                            ctypes.c_int, [cfg_h, ctypes.c_int])
    f["config_set_frame_buffer_size"] = _bind(lib, "librediffusion_config_set_frame_buffer_size",
                                              ctypes.c_int, [cfg_h, ctypes.c_int])
    f["config_set_guidance_scale"] = _bind(lib, "librediffusion_config_set_guidance_scale",
                                           ctypes.c_int, [cfg_h, ctypes.c_float])
    f["config_set_cfg_type"] = _bind(lib, "librediffusion_config_set_cfg_type",
                                     ctypes.c_int, [cfg_h, ctypes.c_int])
    f["config_set_text_config"] = _bind(lib, "librediffusion_config_set_text_config",
                                        ctypes.c_int,
                                        [cfg_h, ctypes.c_int, ctypes.c_int, ctypes.c_int])
    f["config_set_unet_engine"] = _bind(lib, "librediffusion_config_set_unet_engine",
                                        ctypes.c_int, [cfg_h, ctypes.c_char_p])
    f["config_set_combined"] = _bind(
        lib, "librediffusion_config_set_combined_unet_controlnet_engine",
        ctypes.c_int, [cfg_h, ctypes.c_char_p])
    f["config_set_vae_encoder"] = _bind(lib, "librediffusion_config_set_vae_encoder",
                                        ctypes.c_int, [cfg_h, ctypes.c_char_p])
    f["config_set_vae_decoder"] = _bind(lib, "librediffusion_config_set_vae_decoder",
                                        ctypes.c_int, [cfg_h, ctypes.c_char_p])
    f["config_set_timestep_indices"] = _bind(
        lib, "librediffusion_config_set_timestep_indices",
        ctypes.c_int, [cfg_h, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t])

    # pipeline
    f["pipeline_create"] = _bind(lib, "librediffusion_pipeline_create",
                                 ctypes.c_int, [cfg_h, ctypes.POINTER(pipe_h)])
    f["pipeline_destroy"] = _bind(lib, "librediffusion_pipeline_destroy", None, [pipe_h])
    f["pipeline_init_all"] = _bind(lib, "librediffusion_pipeline_init_all",
                                   ctypes.c_int, [pipe_h])
    f["pipeline_get_stream"] = _bind(lib, "librediffusion_pipeline_get_stream",
                                     ctypes.c_void_p, [pipe_h])

    # CLIP
    f["clip_create"] = _bind(lib, "librediffusion_clip_create",
                             ctypes.c_int,
                             [ctypes.c_char_p, ctypes.POINTER(clip_h)])
    f["clip_destroy"] = _bind(lib, "librediffusion_clip_destroy", None, [clip_h])
    f["clip_compute_embeddings"] = _bind(
        lib, "librediffusion_clip_compute_embeddings",
        ctypes.c_int,
        [clip_h, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p,
         ctypes.POINTER(half_p)])

    # embeds + scheduler
    f["prepare_embeds"] = _bind(lib, "librediffusion_prepare_embeds",
                                ctypes.c_int,
                                [pipe_h, half_p, ctypes.c_int, ctypes.c_int])
    f["prepare_negative_embeds"] = _bind(
        lib, "librediffusion_prepare_negative_embeds",
        ctypes.c_int, [pipe_h, half_p, ctypes.c_int, ctypes.c_int])
    f["prepare_scheduler"] = _bind(
        lib, "librediffusion_prepare_scheduler",
        ctypes.c_int,
        [pipe_h, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
         ctypes.POINTER(ctypes.c_float), ctypes.c_size_t])

    # ControlNet setters
    f["reseed"] = _bind(lib, "librediffusion_reseed",
                        ctypes.c_int, [pipe_h, ctypes.c_int64])

    f["set_control_image"] = _bind(
        lib, "librediffusion_set_control_image",
        ctypes.c_int,
        [pipe_h, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int])
    f["set_controlnet_strength"] = _bind(
        lib, "librediffusion_set_controlnet_strength",
        ctypes.c_int, [pipe_h, ctypes.c_float])

    # img2img
    f["img2img"] = _bind(lib, "librediffusion_img2img",
                         ctypes.c_int,
                         [pipe_h,
                          ctypes.POINTER(ctypes.c_uint8),
                          ctypes.POINTER(ctypes.c_uint8),
                          ctypes.c_int, ctypes.c_int])

    f["error_string"] = _bind(lib, "librediffusion_error_string",
                              ctypes.c_char_p, [ctypes.c_int])

    f["_lib"] = lib
    return f


def _check(api: dict, rc: int, what: str) -> None:
    if rc != ERR_OK:
        msg = api["error_string"](rc).decode()
        raise RuntimeError(f"{what} failed: {msg} (code {rc})")


def _load_rgba(path: Path, w: int, h: int) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    if img.size != (w, h):
        img = img.resize((w, h), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def make_sd_turbo_scheduler_at(t_index: int, num_inference_steps: int = 50,
                               base_model: str = "stabilityai/sd-turbo"):
    """Return scheduler params for a given INDEX into an LCMScheduler's
    precomputed timesteps — matches what the runner does internally and
    what tools/generate_scheduler_tables.py emits for the engine.

    Earlier versions of this function treated `t_index` as a raw timestep
    value (0..999), which made low indices like 7 collapse to "no-noise"
    and the pipeline produced output ≈ source.
    """
    from diffusers import LCMScheduler
    scheduler = LCMScheduler.from_pretrained(base_model, subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps)
    if t_index < 0 or t_index >= num_inference_steps:
        raise ValueError(
            f"--timestep must be in [0, {num_inference_steps - 1}] (LCM index); got {t_index}"
        )
    timestep_int = int(scheduler.timesteps[t_index].item())
    c_skip, c_out = scheduler.get_scalings_for_boundary_condition_discrete(timestep_int)
    alpha_prod_t = scheduler.alphas_cumprod[timestep_int]
    a_sqrt = float(alpha_prod_t.sqrt())
    b_sqrt = float((1 - alpha_prod_t).sqrt())
    return (np.float32(timestep_int),
            np.float32(a_sqrt),
            np.float32(b_sqrt),
            np.float32(float(c_skip)),
            np.float32(float(c_out)))


def make_hyper_sd_scheduler_at(t_index: int, num_inference_steps: int = 1,
                               base_model: str = "runwayml/stable-diffusion-v1-5"):
    """Return scheduler params for SD-1.5 + Hyper-SD-1step using TCDScheduler.

    Hyper-SD's model card recommends TCDScheduler with eta=1.0; the 1-step
    variant collapses denoising to a single TCD step (so `num_inference_steps=1`
    -> only t_index=0 is valid). Higher counts match the 2/4/8-step Hyper-SD
    variants.

    TCDScheduler doesn't expose `get_scalings_for_boundary_condition_discrete`
    (that's LCM-only), so we inline the formula. Both schedulers use
    sigma_data=0.5 and config.timestep_scaling (default 10) -- the math is
    identical to LCM's helper, just spelled out.
    """
    from diffusers import TCDScheduler
    scheduler = TCDScheduler.from_pretrained(base_model, subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps)
    if t_index < 0 or t_index >= num_inference_steps:
        raise ValueError(
            f"--timestep must be in [0, {num_inference_steps - 1}] (TCD index); got {t_index}"
        )
    timestep_int = int(scheduler.timesteps[t_index].item())
    sigma_data = 0.5
    timestep_scaling = float(getattr(scheduler.config, "timestep_scaling", 10.0))
    scaled_t = timestep_int * timestep_scaling
    c_skip = sigma_data ** 2 / (scaled_t ** 2 + sigma_data ** 2)
    c_out = scaled_t / (scaled_t ** 2 + sigma_data ** 2) ** 0.5
    alpha_prod_t = scheduler.alphas_cumprod[timestep_int]
    a_sqrt = float(alpha_prod_t.sqrt())
    b_sqrt = float((1 - alpha_prod_t).sqrt())
    return (np.float32(timestep_int),
            np.float32(a_sqrt),
            np.float32(b_sqrt),
            np.float32(float(c_skip)),
            np.float32(float(c_out)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True,
                   help="UNet engine path. Treated as combined unet+controlnet by "
                        "default; pass --no-controlnet to load it as a plain UNet.")
    p.add_argument("--no-controlnet", action="store_true",
                   help="Bypass ControlNet: load --engine as a plain UNet, skip "
                        "set_control_image / set_controlnet_strength. Useful as a "
                        "baseline to confirm img2img itself works.")
    p.add_argument("--vae-encoder", required=True)
    p.add_argument("--vae-decoder", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--source", default=str(DEFAULT_FIXTURE_DIR / "source.png"))
    p.add_argument("--control", default=str(DEFAULT_FIXTURE_DIR / "canny.png"))
    p.add_argument("--prompt", default="a watercolor painting of a mountain landscape")
    p.add_argument("--negative", default="blurry, low quality, distorted")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--timestep", type=int, default=1)
    p.add_argument("--guidance", type=float, default=1.2)
    p.add_argument("--text-hidden-dim", type=int, default=1024,
                   help="SD-Turbo: 1024; SD1.5: 768")
    p.add_argument("--scheduler", choices=["sd-turbo", "hyper-sd"], default="sd-turbo",
                   help="Scheduler factory: sd-turbo (LCMScheduler from "
                        "stabilityai/sd-turbo, 50 inference steps; default) "
                        "or hyper-sd (TCDScheduler from runwayml/stable-diffusion-v1-5, "
                        "1 inference step -- for SD-1.5 stacks with Hyper-SD-1step "
                        "fused). The C++ runtime config is identical between the "
                        "two; only the (timestep, alpha, beta, c_skip, c_out) tuple "
                        "differs.")
    p.add_argument("--output", default="./test_output.png")
    p.add_argument("--display", action="store_true")
    p.add_argument("--dll", default=None,
                   help="Override librediffusion shared-library path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dll_path = Path(args.dll) if args.dll else _find_dll()
    print(f"[lib] loading {dll_path}")
    api = load_lib(dll_path)

    cfg = ctypes.c_void_p()
    _check(api, api["config_create"](ctypes.byref(cfg)), "config_create")
    try:
        api["config_set_device"](cfg, 0)
        api["config_set_model_type"](cfg, 1)  # MODEL_SD_TURBO
        api["config_set_dimensions"](cfg, args.width, args.height,
                                     args.width // 8, args.height // 8)
        api["config_set_batch_size"](cfg, 1)
        api["config_set_denoising_steps"](cfg, 1)
        api["config_set_frame_buffer_size"](cfg, 1)
        api["config_set_guidance_scale"](cfg, args.guidance)
        api["config_set_cfg_type"](cfg, 2)  # SD_CFG_SELF
        api["config_set_text_config"](cfg, 77, args.text_hidden_dim, 0)
        if args.no_controlnet:
            api["config_set_unet_engine"](cfg, args.engine.encode())
        else:
            api["config_set_combined"](cfg, args.engine.encode())
        api["config_set_vae_encoder"](cfg, args.vae_encoder.encode())
        api["config_set_vae_decoder"](cfg, args.vae_decoder.encode())

        timesteps_arr = (ctypes.c_int * 1)(args.timestep)
        api["config_set_timestep_indices"](cfg, timesteps_arr, 1)

        pipe = ctypes.c_void_p()
        _check(api, api["pipeline_create"](cfg, ctypes.byref(pipe)),
               "pipeline_create")

        try:
            _check(api, api["pipeline_init_all"](pipe), "pipeline_init_all")
            # CRITICAL: without reseed, init_noise_ stays zeroed and the
            # single-step Turbo update degenerates to x_0 ≈ source — no
            # transformation. TD's runner calls reseed for the same reason.
            _check(api, api["reseed"](pipe, args.seed), "reseed")
            stream = api["pipeline_get_stream"](pipe)

            # CLIP embeddings
            clip = ctypes.c_void_p()
            _check(api, api["clip_create"](args.clip.encode(), ctypes.byref(clip)),
                   "clip_create")
            try:
                pos_emb = ctypes.POINTER(ctypes.c_uint16)()
                neg_emb = ctypes.POINTER(ctypes.c_uint16)()
                _check(api, api["clip_compute_embeddings"](
                    clip, args.prompt.encode(), 0, stream, ctypes.byref(pos_emb)),
                    "clip_compute_embeddings(positive)")
                _check(api, api["clip_compute_embeddings"](
                    clip, args.negative.encode(), 0, stream, ctypes.byref(neg_emb)),
                    "clip_compute_embeddings(negative)")
                _check(api, api["prepare_embeds"](pipe, pos_emb, 77, args.text_hidden_dim),
                       "prepare_embeds")
                _check(api, api["prepare_negative_embeds"](
                    pipe, neg_emb, 77, args.text_hidden_dim),
                       "prepare_negative_embeds")
            finally:
                api["clip_destroy"](clip)

            # Scheduler
            if args.scheduler == "hyper-sd":
                t, a, b, cs, co = make_hyper_sd_scheduler_at(args.timestep)
            else:
                t, a, b, cs, co = make_sd_turbo_scheduler_at(args.timestep)
            ts_arr = (ctypes.c_float * 1)(t)
            a_arr = (ctypes.c_float * 1)(a)
            b_arr = (ctypes.c_float * 1)(b)
            cs_arr = (ctypes.c_float * 1)(cs)
            co_arr = (ctypes.c_float * 1)(co)
            _check(api, api["prepare_scheduler"](
                pipe, ts_arr, a_arr, b_arr, cs_arr, co_arr, 1),
                   "prepare_scheduler")

            # Fixtures
            print(f"[fix] source:  {args.source}")
            print(f"[fix] control: {args.control}")
            src = _load_rgba(Path(args.source), args.width, args.height)
            ctl = _load_rgba(Path(args.control), args.width, args.height)

            if not args.no_controlnet:
                ctl_ptr = ctl.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                _check(api, api["set_control_image"](pipe, ctl_ptr, args.width, args.height),
                       "set_control_image")
                _check(api, api["set_controlnet_strength"](pipe, args.strength),
                       "set_controlnet_strength")

            # Inference
            out = np.zeros_like(src)
            src_ptr = src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

            t0 = time.perf_counter()
            _check(api, api["img2img"](pipe, src_ptr, out_ptr, args.width, args.height),
                   "img2img")
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[run] img2img complete in {elapsed:.1f} ms")

            out_path = Path(args.output)
            Image.fromarray(out, mode="RGBA").save(out_path)
            # Per-channel stats: distinguishes "real img2img" (broad spread)
            # from "blue noise" (narrow, blue-dominant, high std-of-stds).
            for ci, cn in enumerate(("R", "G", "B", "A")):
                ch = out[:, :, ci].astype(np.float32)
                print(f"[stat] {cn}: min={ch.min():.0f} max={ch.max():.0f} "
                      f"mean={ch.mean():.1f} std={ch.std():.1f}")
            print(f"[out] wrote {out_path}")
            if args.display:
                Image.fromarray(out, mode="RGBA").show()
        finally:
            api["pipeline_destroy"](pipe)
    finally:
        api["config_destroy"](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
