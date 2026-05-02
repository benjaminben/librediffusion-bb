#!/usr/bin/env python3
"""
Export a combined UNet + ControlNet TensorRT engine for librediffusion v1.

The combined engine extends the standard UNet with two extra inputs:
  - control_image:        [1, 3, H, W]  fp16
  - controlnet_strength:  [1]           fp16

ControlNet residual addition is folded INSIDE the engine, so per-frame C++
changes are minimal (one extra binding for the control image, one for the
strength scalar). The strength is applied as a scalar Mul on each
ControlNet residual; TensorRT is expected to fuse the Mul/Add into the
adjacent UNet skip-connection ops.

This is the orchestration entry point. Implementation lives in:
  - _export_models.py  (CombinedUNetControlNet, model loading, LoRA fusion,
                       dummy inputs)
  - _export_build.py   (ONNX export + simplify, TRT engine build)
  - _export_diagnostics.py  (model + ONNX + engine diagnostics; pre-existing)

Usage:
    python tools/export_combined_unet_controlnet.py \\
        --base-model stabilityai/sd-turbo \\
        --controlnet lllyasviel/sd-controlnet-canny \\
        --width 512 --height 512 \\
        --output engines/unet_controlnet_canny.engine
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

import _export_diagnostics as diag
from _export_build import build_engine, export_onnx, simplify_onnx
from _export_models import CombinedUNetControlNet, load_models, make_dummy_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="stabilityai/sd-turbo",
                        help="HuggingFace base UNet model id "
                             "(SD-Turbo: stabilityai/sd-turbo; "
                             "SD-1.5: runwayml/stable-diffusion-v1-5)")
    parser.add_argument("--controlnet", default="lllyasviel/sd-controlnet-canny",
                        help="HuggingFace ControlNet model id")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="v1 fixes batch_size=1; values != 1 rejected")
    parser.add_argument("--text-seq-len", type=int, default=77)
    parser.add_argument("--text-hidden-dim", type=int, default=1024,
                        help="SD-Turbo: 1024; SD1.5: 768")
    parser.add_argument("--distillation-lora", default=None,
                        help="Optional HF repo id or local file path of a "
                             "distillation LoRA to fuse into the UNet before "
                             "ONNX trace (e.g. ByteDance/Hyper-SD for SD-1.5 "
                             "Hyper-SD-1step). When omitted, the UNet is "
                             "exported as-is -- preserves vanilla SD-Turbo "
                             "exports. Pair with --distillation-lora-weight-name "
                             "when the value is a repo id with multiple LoRA "
                             "files in it.")
    parser.add_argument("--distillation-lora-weight-name", default=None,
                        help="Filename within --distillation-lora repo "
                             "(e.g. Hyper-SD15-1step-lora.safetensors). "
                             "Ignored when --distillation-lora is a local path.")
    parser.add_argument("--output", required=True, help="Output engine path")
    parser.add_argument("--onnx-path", default=None,
                        help="Where to save the intermediate .onnx (default: <output>.onnx)")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--workspace-mb", type=int, default=8192)
    parser.add_argument("--skip-build", action="store_true",
                        help="Export ONNX only; skip TRT engine build")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_sidecar(engine_path: Path, args: argparse.Namespace, diagnostics: dict) -> None:
    meta = {
        "engine_kind": "combined_unet_controlnet",
        "base_model": args.base_model,
        "controlnet": args.controlnet,
        "distillation_lora": args.distillation_lora,
        "distillation_lora_weight_name": args.distillation_lora_weight_name,
        "width": args.width,
        "height": args.height,
        "batch_size": args.batch_size,
        "text_seq_len": args.text_seq_len,
        "text_hidden_dim": args.text_hidden_dim,
        "precision": "fp16" if args.fp16 else "fp32",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "diagnostics": diagnostics,
    }
    sidecar = engine_path.with_suffix(engine_path.suffix + ".meta.json")
    with open(sidecar, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[meta] wrote {sidecar}")
    log_path = engine_path.with_suffix(engine_path.suffix + ".diag.log")
    diag.write_diagnostics_log(diagnostics, log_path)
    print(f"[meta] wrote {log_path}")


def main() -> int:
    args = parse_args()
    if args.batch_size != 1:
        print("error: v1 fixes batch_size=1 (combined-engine + single-step config lock)")
        return 2

    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 else torch.float32
    diagnostics: dict = {"args": vars(args), "torch_version": torch.__version__}

    unet, controlnet = load_models(args, device, dtype)
    model = CombinedUNetControlNet(unet, controlnet).to(device).eval()
    dummies = make_dummy_inputs(args, device, dtype)
    diagnostics.update(diag.run_model_diagnostics(unet, controlnet, model, dummies, dtype))

    output_path = Path(args.output)
    # Default scratch (.onnx + external-data sidecars) goes into a sibling
    # `_build/` subdir of the engine output, so the engine folder stays
    # uncluttered. Override with --onnx-path to write elsewhere.
    if args.onnx_path:
        onnx_path = Path(args.onnx_path)
    else:
        onnx_path = output_path.parent / "_build" / (output_path.stem + ".onnx")

    with torch.inference_mode():
        export_onnx(model, dummies, onnx_path)
    diagnostics.update(diag.run_onnx_diagnostics(onnx_path))
    simplify_onnx(onnx_path)

    if args.skip_build:
        print("[done] --skip-build set; ONNX written but engine not built")
        write_sidecar(output_path, args, diagnostics)
        return 0

    diagnostics["build_info"] = build_engine(onnx_path, output_path,
                                             args.fp16, args.workspace_mb)
    write_sidecar(output_path, args, diagnostics)
    print(f"[done] {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
