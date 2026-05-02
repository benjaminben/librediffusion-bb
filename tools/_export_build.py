"""
Graph-build concerns of the combined UNet+ControlNet exporter:
  - torch.onnx.export wrapper
  - streamdiffusion-Optimizer-based ONNX simplification (best-effort)
  - TensorRT engine build + post-build layer inspection

The split keeps ONNX/TRT plumbing isolated from model loading -- TRT
version bumps and ONNX opset changes touch only this file.
"""

import json
import time
from pathlib import Path

import torch.nn as nn

import _export_diagnostics as diag

# Optional: ONNX simplifier from streamdiffusion's polygraphy fork.
# Imported read-only — we don't modify that subtree.
try:
    from streamdiffusion.acceleration.tensorrt.models import Optimizer  # type: ignore
    HAS_OPTIMIZER = True
except Exception:
    HAS_OPTIMIZER = False


def export_onnx(model: nn.Module, dummies, onnx_path: Path, opset: int = 17) -> None:
    import torch
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
