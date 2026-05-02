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

Usage:
    python tools/export_combined_unet_controlnet.py \\
        --base-model stabilityai/sd-turbo \\
        --controlnet lllyasviel/sd-controlnet-canny \\
        --width 512 --height 512 \\
        --output engines/unet_controlnet_canny.engine
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

import _export_diagnostics as diag

# Optional: ONNX simplifier from streamdiffusion's polygraphy fork.
# Imported read-only — we don't modify that subtree.
try:
    from streamdiffusion.acceleration.tensorrt.models import Optimizer  # type: ignore
    HAS_OPTIMIZER = True
except Exception:
    HAS_OPTIMIZER = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="stabilityai/sd-turbo",
                        help="HuggingFace base UNet model id")
    parser.add_argument("--controlnet", default="lllyasviel/sd-controlnet-canny",
                        help="HuggingFace ControlNet model id")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="v1 fixes batch_size=1; values != 1 rejected")
    parser.add_argument("--text-seq-len", type=int, default=77)
    parser.add_argument("--text-hidden-dim", type=int, default=1024,
                        help="SD-Turbo: 1024; SD1.5: 768")
    parser.add_argument("--output", required=True, help="Output engine path")
    parser.add_argument("--onnx-path", default=None,
                        help="Where to save the intermediate .onnx (default: <output>.onnx)")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--workspace-mb", type=int, default=8192)
    parser.add_argument("--skip-build", action="store_true",
                        help="Export ONNX only; skip TRT engine build")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class CombinedUNetControlNet(nn.Module):
    """Wraps base UNet + ControlNet into a single forward pass.

    Inputs match the standard UNet plus two extras:
      - control_image       [B, 3, H, W]  fp16
      - controlnet_strength [1]           fp16  (scalar multiplier)
    """

    def __init__(self, unet: nn.Module, controlnet: nn.Module) -> None:
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_image: torch.Tensor,
        controlnet_strength: torch.Tensor,
    ) -> torch.Tensor:
        # ControlNet returns a list of down-block residuals + a mid-block residual.
        cn_out = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_image,
            return_dict=False,
        )
        down_residuals, mid_residual = cn_out

        # Apply runtime strength. Cast scale to the residual dtype (fp16) so
        # the Mul stays in fp16 — sample is fp32 to match train-lora.py's
        # plain unet engine, but residuals come out of ControlNet in fp16
        # and we want the skip-connection Add to fuse cleanly.
        scale = controlnet_strength.to(mid_residual.dtype).reshape(())
        down_residuals = [r * scale for r in down_residuals]
        mid_residual = mid_residual * scale

        out = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
            return_dict=False,
        )[0]
        return out


def load_models(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    print(f"[load] base UNet:   {args.base_model}")
    print(f"[load] ControlNet:  {args.controlnet}")
    from diffusers import UNet2DConditionModel, ControlNetModel
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model, subfolder="unet", torch_dtype=dtype
    ).to(device).eval()
    cn = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype).to(device).eval()
    return unet, cn


def make_dummy_inputs(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    b, h, w = args.batch_size, args.height, args.width
    lh, lw = h // 8, w // 8
    # sample MUST be fp32 to match the plain unet.engine (train-lora.py via
    # streamdiffusion hardcodes torch.float32 here regardless of fp16 mode).
    # The librediffusion C++ wrapper allocates a fp32 sample_buffer_ and
    # binds it to the engine's "sample" input — if this is exported as fp16
    # the wrapper feeds the engine garbage bytes, producing blue-noise
    # output that's invariant to ControlNet inputs.
    sample = torch.randn(b, 4, lh, lw, device=device, dtype=torch.float32)
    timestep = torch.tensor([1.0] * b, device=device, dtype=torch.float32)
    encoder = torch.randn(b, args.text_seq_len, args.text_hidden_dim, device=device, dtype=dtype)
    # IMPORTANT: avoid trivial dummy values. With control_image=zeros + strength=1
    # and torch.onnx do_constant_folding=True, the optimizer can prune the
    # ControlNet branch and fold away the residual*scale Muls -- the engine
    # ends up with declared but disconnected control_image / controlnet_strength
    # inputs, so runtime output is invariant to those bindings.
    control_image = torch.rand(b, 3, h, w, device=device, dtype=dtype)
    strength = torch.tensor([0.5], device=device, dtype=dtype)
    return sample, timestep, encoder, control_image, strength


def export_onnx(model: nn.Module, dummies, onnx_path: Path, opset: int = 17) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[onnx] exporting to {onnx_path}")
    sample, timestep, encoder, control_image, strength = dummies

    input_names = ["sample", "timestep", "encoder_hidden_states",
                   "control_image", "controlnet_strength"]
    output_names = ["latent"]

    # dynamo=False -> legacy jit.trace path. Default (True) emits an
    # empty initializer (val_1683 nulled by TRT) and produces blue-noise
    # latents. autocast lets the fp32 sample / timestep flow through fp16
    # model weights at op-level, same pattern as streamdiffusion's
    # acceleration/tensorrt/utilities.py:575.
    print(f"[onnx] exporter: torch.onnx.export(dynamo=False), opset={opset}")
    with torch.autocast("cuda"):
        torch.onnx.export(
            model,
            (sample, timestep, encoder, control_image, strength),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"[onnx] wrote {onnx_path.stat().st_size / 1e6:.1f} MB")


def simplify_onnx(onnx_path: Path) -> None:
    if not HAS_OPTIMIZER:
        print("[onnx] streamdiffusion Optimizer not available — skipping simplify")
        return
    print("[onnx] simplifying via streamdiffusion Optimizer")
    try:
        opt = Optimizer(str(onnx_path))
        opt.fold_constants()
        opt.infer_shapes()
        opt.cleanup()
        opt.save(str(onnx_path))
        print("[onnx] simplify OK")
    except Exception as e:
        print(f"[onnx] simplify failed: {e}; continuing with un-simplified ONNX")


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool, workspace_mb: int) -> dict:
    print(f"[trt]  building engine -> {engine_path}")
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    # TF32 for matmul (Ampere+). train-lora.py's pipeline enables this and the
    # plain unet.engine relies on the resulting numerics. Matching it here
    # keeps the combined engine's UNet path bit-for-bit comparable.
    config.set_flag(trt.BuilderFlag.TF32)

    # Use parse_from_file so external-data sidecars (the *.weight files torch
    # writes alongside large ONNX graphs) resolve relative to the .onnx path.
    parser = trt.OnnxParser(network, logger)
    # NATIVE_INSTANCENORM: have TRT use its native InstanceNorm op rather than
    # emulating it via subgraphs. Without this, SD-Turbo's UNet normalization
    # layers compile with different numerics from the plain unet.engine -> the
    # combined engine produces over-magnitude outputs (the "blue noise" symptom)
    # even when ControlNet is fully zeroed via strength=0.
    parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
    ok = parser.parse_from_file(str(onnx_path))
    if not ok:
        for i in range(parser.num_errors):
            print(f"[trt]  parse error: {parser.get_error(i)}")
        raise RuntimeError("ONNX parse failed")

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed (no serialized network)")
    elapsed = time.time() - t0

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[trt]  wrote {engine_path.stat().st_size / 1e6:.1f} MB in {elapsed:.1f}s")

    fusion_status = inspect_layer_fusion(serialized, logger)
    layer_info_path = engine_path.with_suffix(engine_path.suffix + ".layers.json")
    layer_dump = diag.dump_trt_layer_info(serialized, logger, layer_info_path)
    print(f"[trt]  layers: {layer_dump['num_layers']} total, "
          f"{layer_dump['controlnet_subgraph_layers']} /controlnet/, "
          f"{layer_dump['controlnet_cond_embedding_layers']} cond-embed, "
          f"{layer_dump['mul_layers']} mul -> {layer_info_path}")
    if layer_dump["controlnet_subgraph_layers"] == 0:
        print("[trt]  !! WARNING: no /controlnet/ layers in engine — "
              "TRT folded the entire ControlNet subgraph out")
    return {
        "tensorrt_version": trt.__version__,
        "build_seconds": round(elapsed, 2),
        "fusion": fusion_status,
        "layer_dump": layer_dump,
    }


def inspect_layer_fusion(serialized_engine: bytes, logger) -> dict:
    """Best-effort: enumerate the engine layers and count Mul/Add ops.

    A small Mul count near the controlnet_strength input suggests the
    expected fusion happened. If we see many un-fused Muls per residual,
    the fallback is to bake strength=1.0 at export time (S0).
    """
    try:
        import tensorrt as trt
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        # TRT 10 exposes inspector; not all platforms expose layer info reliably.
        inspector = engine.create_engine_inspector()
        info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        try:
            info_json = json.loads(info) if isinstance(info, str) else info
            layers = info_json.get("Layers", []) if isinstance(info_json, dict) else []
        except Exception:
            layers = []
        mul_count = sum(1 for L in layers if "Mul" in str(L))
        return {
            "num_layers": len(layers),
            "mul_layer_count": mul_count,
            "note": "Low mul_layer_count near controlnet_strength suggests fusion succeeded.",
        }
    except Exception as e:
        return {"error": f"inspector failed: {e}"}


def write_sidecar(engine_path: Path, args: argparse.Namespace, diagnostics: dict) -> None:
    meta = {
        "engine_kind": "combined_unet_controlnet",
        "base_model": args.base_model,
        "controlnet": args.controlnet,
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
