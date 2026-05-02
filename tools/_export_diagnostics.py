"""Diagnostic helpers for combined UNet+ControlNet export.

A single export run takes ~3 minutes (mostly TRT build), so we want to
extract every signal the run can give us. Each function here returns a
JSON-serializable dict; the orchestrator collects them into the engine's
.meta.json sidecar so the entire diagnostic bundle can be diffed across
runs after the fact.

Keep this file dependency-light: only torch + onnx are required at import
time. tensorrt is imported lazily inside dump_trt_layer_info.
"""

import json
from pathlib import Path

import torch


# --- Per-tensor / per-module fingerprints --------------------------------------

def summarize_tensor(t, name=""):
    if t is None:
        return {"name": name, "is_none": True}
    f = t.detach().float()
    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "min": float(f.min().item()),
        "max": float(f.max().item()),
        "mean": float(f.mean().item()),
        "std": float(f.std().item()) if t.numel() > 1 else 0.0,
        "num_nan": int(torch.isnan(f).sum().item()),
        "num_inf": int(torch.isinf(f).sum().item()),
        "num_zero": int((f == 0).sum().item()),
        "numel": int(t.numel()),
    }


def summarize_module(module, name):
    """Module fingerprint. A randomly-initialized model and a pretrained
    one have identical structure, but their abs_weight_sum differs by
    orders of magnitude — cheap proof from_pretrained populated weights.
    """
    n_params, abs_sum, n_nan, n_inf, n_zero, n_tensors = 0, 0.0, 0, 0, 0, 0
    for p in module.parameters():
        f = p.detach().float()
        n_params += p.numel()
        abs_sum += float(f.abs().sum().item())
        n_nan += int(torch.isnan(f).sum().item())
        n_inf += int(torch.isinf(f).sum().item())
        n_zero += int((f == 0).sum().item())
        n_tensors += 1
    return {
        "name": name,
        "num_params": n_params,
        "num_param_tensors": n_tensors,
        "abs_weight_sum": abs_sum,
        "abs_weight_sum_per_param": abs_sum / max(n_params, 1),
        "num_nan_weights": n_nan,
        "num_inf_weights": n_inf,
        "num_zero_weights": n_zero,
    }


# --- PyTorch-eager sanity ------------------------------------------------------

def eager_sanity_check(model, dummies, dtype):
    """Three forward passes through the wrapper in eager PyTorch:
        A: (control_image=A, strength=1)
        B: (control_image=B, strength=1)   — should differ from A
        C: (control_image=A, strength=0)   — should differ from A
    If the eager model is sensitive but the engine is not, the export
    pipeline lost the dependency. If eager itself is invariant, the
    wrapper / model combo is the bug.
    """
    sample, timestep, encoder, control_image, _ = dummies
    device = sample.device

    ci_a = control_image
    ci_b = torch.rand_like(control_image)
    s_one = torch.tensor([1.0], device=device, dtype=dtype)
    s_zero = torch.tensor([0.0], device=device, dtype=dtype)

    model.eval()
    # autocast bridges fp32 sample / timestep with fp16 model weights, the
    # same way streamdiffusion's export_onnx (utilities.py:575) does.
    with torch.no_grad(), torch.autocast("cuda"):
        out_a1 = model(sample, timestep, encoder, ci_a, s_one)
        out_b1 = model(sample, timestep, encoder, ci_b, s_one)
        out_a0 = model(sample, timestep, encoder, ci_a, s_zero)

    def diff(x, y):
        return float((x.float() - y.float()).abs().mean().item())

    return {
        "out_canny_strength1": summarize_tensor(out_a1, "canny@s=1"),
        "out_ood_strength1":   summarize_tensor(out_b1, "ood@s=1"),
        "out_canny_strength0": summarize_tensor(out_a0, "canny@s=0"),
        "delta_control_image": diff(out_a1, out_b1),
        "delta_strength":      diff(out_a1, out_a0),
        "_expectation": "delta_control_image and delta_strength must be > 0",
    }


# --- ONNX inspection -----------------------------------------------------------

def scan_onnx_graph(onnx_path):
    """Look for known ONNX-side footguns:
      - empty-data initializers with non-empty dims (TRT silently nulls
        these — the export-engine.log val_1683 warning is exactly this).
      - NaN / Inf inside initializer raw_data.
      - op-type histogram (compare across exporter paths).
    """
    import onnx
    import numpy as np

    model = onnx.load(str(onnx_path), load_external_data=False)
    g = model.graph

    empty_initializers = []
    nan_initializers = []
    init_count = 0
    for init in g.initializer:
        init_count += 1
        dims = list(init.dims)
        expected = 1
        for d in dims:
            expected *= d
        has_raw = bool(init.raw_data)
        has_fields = bool(init.float_data) or bool(init.int32_data) \
            or bool(init.int64_data) or bool(init.double_data) \
            or bool(init.uint64_data)
        has_external = bool(getattr(init, "external_data", []))
        if expected > 0 and not has_raw and not has_fields and not has_external:
            empty_initializers.append({
                "name": init.name,
                "dims": dims,
                "expected_numel": expected,
            })
        if has_raw and init.data_type in (1, 10, 11):  # FLOAT, FLOAT16, DOUBLE
            try:
                arr = np.frombuffer(init.raw_data, dtype={
                    1: np.float32, 10: np.float16, 11: np.float64,
                }[init.data_type])
                if arr.size and (np.isnan(arr).any() or np.isinf(arr).any()):
                    nan_initializers.append({
                        "name": init.name,
                        "dims": dims,
                        "num_nan": int(np.isnan(arr).sum()),
                        "num_inf": int(np.isinf(arr).sum()),
                    })
            except Exception:
                pass

    op_hist = {}
    for n in g.node:
        op_hist[n.op_type] = op_hist.get(n.op_type, 0) + 1

    def shape_of(vi):
        return [d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in vi.type.tensor_type.shape.dim]

    return {
        "num_nodes": len(g.node),
        "num_initializers": init_count,
        "num_empty_initializers": len(empty_initializers),
        "empty_initializers": empty_initializers[:50],
        "num_nan_initializers": len(nan_initializers),
        "nan_initializers": nan_initializers[:20],
        "op_type_histogram": dict(sorted(op_hist.items(), key=lambda x: -x[1])),
        "inputs":  [{"name": i.name, "shape": shape_of(i)} for i in g.input],
        "outputs": [{"name": o.name, "shape": shape_of(o)} for o in g.output],
        "ir_version": model.ir_version,
        "producer": f"{model.producer_name} {model.producer_version}",
        "opset_imports": {imp.domain or "ai.onnx": imp.version for imp in model.opset_import},
    }


def reachability_check(onnx_path):
    """Per-input forward BFS through node consumers. Inline equivalent
    of tools/_inspect_onnx_reachability.py — embedded here so a single
    export run produces the reachability verdict in its meta sidecar.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    g = model.graph
    output_names = [o.name for o in g.output]

    consumers = {}
    for n in g.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)

    def forward_reach(start):
        seen, produced = set(), set()
        frontier = [start]
        while frontier:
            t = frontier.pop()
            for n in consumers.get(t, ()):
                if n.name in seen:
                    continue
                seen.add(n.name)
                for out in n.output:
                    if out in produced:
                        continue
                    produced.add(out)
                    frontier.append(out)
        return len(seen), produced

    out = {}
    for inp in g.input:
        n_nodes, produced = forward_reach(inp.name)
        reachable = [o for o in output_names if o in produced]
        out[inp.name] = {
            "forward_reach_nodes": n_nodes,
            "forward_reach_tensors": len(produced),
            "reachable_outputs": reachable,
            "verdict": "REACHABLE" if reachable else "DISCONNECTED",
        }
    return out


# --- TRT engine inspection -----------------------------------------------------

def dump_trt_layer_info(serialized_engine, logger, output_path):
    """Deserialize the freshly-built engine and write the full layer JSON
    to a sidecar. Lets us grep post-build for "is the strength Mul
    present?", layer-count parity vs the plain-unet engine, etc.
    """
    import tensorrt as trt
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    inspector = engine.create_engine_inspector()
    info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(info if isinstance(info, str) else json.dumps(info))

    try:
        info_json = json.loads(info) if isinstance(info, str) else info
        layers = info_json.get("Layers", []) if isinstance(info_json, dict) else []
    except Exception:
        layers = []

    # TRT 10's LayerInformationFormat.JSON returns just a list of layer-name
    # strings (not structured dicts), so the only signal we can extract is
    # name-prefix matching. Count layers attributable to the ControlNet
    # subgraph and the Mul ops that should fall near each residual scale.
    def name_of(L):
        if isinstance(L, str):
            return L
        if isinstance(L, dict):
            return L.get("Name") or json.dumps(L)
        return str(L)

    names = [name_of(L) for L in layers]
    cn_count = sum(1 for n in names if "/controlnet/" in n)
    cond_count = sum(1 for n in names if "controlnet_cond" in n)
    mul_count = sum(1 for n in names if "/Mul" in n or "PWN_Mul" in n)
    return {
        "num_layers": len(layers),
        "layer_info_path": str(output_path),
        "controlnet_subgraph_layers": cn_count,
        "controlnet_cond_embedding_layers": cond_count,
        "mul_layers": mul_count,
    }


# --- Bundling ------------------------------------------------------------------

def run_model_diagnostics(unet, controlnet, model, dummies, dtype):
    """Run + print all pre-ONNX diagnostics. Returns the bundle."""
    out: dict = {}
    out["unet"] = summarize_module(unet, "unet")
    out["controlnet"] = summarize_module(controlnet, "controlnet")
    for k in ("unet", "controlnet"):
        m = out[k]
        print(f"[diag] {k:>10}: {m['num_params']:>12,} params, "
              f"abs_sum={m['abs_weight_sum']:.3e}, "
              f"nan={m['num_nan_weights']}, inf={m['num_inf_weights']}")
    if out["controlnet"]["abs_weight_sum"] < 1e3:
        print("[diag] !! controlnet weight magnitude suspiciously small — "
              "from_pretrained may have fallen back to random init")

    names = ["sample", "timestep", "encoder_hidden_states",
             "control_image", "controlnet_strength"]
    out["dummies"] = [summarize_tensor(t, n) for t, n in zip(dummies, names)]
    for d in out["dummies"]:
        print(f"[diag] dummy {d['name']:>22}: shape={d['shape']} "
              f"dtype={d['dtype']} min={d['min']:.4f} max={d['max']:.4f}")

    print("[diag] eager sanity (3 forward passes)...")
    out["eager_sanity"] = eager_sanity_check(model, dummies, dtype)
    e = out["eager_sanity"]
    print(f"[diag] eager delta(control_image A vs B, s=1): {e['delta_control_image']:.6e}")
    print(f"[diag] eager delta(strength 1 vs 0, ci=A):     {e['delta_strength']:.6e}")
    if e["delta_control_image"] == 0:
        print("[diag] !! eager invariant to control_image — wrapper/model broken pre-export")
    if e["delta_strength"] == 0:
        print("[diag] !! eager invariant to strength — controlnet contribution dead pre-export")
    return out


def run_onnx_diagnostics(onnx_path):
    """Scan ONNX + reachability. Prints summaries, returns the bundle."""
    out: dict = {}
    print("[diag] scanning ONNX graph...")
    out["onnx_scan"] = scan_onnx_graph(onnx_path)
    s = out["onnx_scan"]
    print(f"[diag] onnx: {s['num_nodes']} nodes, {s['num_initializers']} initializers, "
          f"empty={s['num_empty_initializers']}, nan={s['num_nan_initializers']}, "
          f"opset={s['opset_imports']}, producer={s['producer']}")
    if s["empty_initializers"]:
        print("[diag] !! EMPTY INITIALIZERS in ONNX (TRT will null these to garbage):")
        for ei in s["empty_initializers"][:10]:
            print(f"[diag]    {ei}")
    if s["nan_initializers"]:
        print("[diag] !! NaN/Inf initializers in ONNX:")
        for ni in s["nan_initializers"][:10]:
            print(f"[diag]    {ni}")

    print("[diag] reachability check...")
    out["reachability"] = reachability_check(onnx_path)
    for name, info in out["reachability"].items():
        print(f"[diag] reach[{name:>22}]: {info['verdict']:>12} "
              f"({info['forward_reach_nodes']} ops, "
              f"{info['forward_reach_tensors']} tensors)")
    return out


def write_diagnostics_log(diag, log_path):
    """Pretty-print every diagnostic block to a sidecar text file so it's
    grep-friendly without having to crack open the JSON.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        for k, v in diag.items():
            f.write(f"\n========== {k} ==========\n")
            f.write(json.dumps(v, indent=2, default=str))
            f.write("\n")
    return str(log_path)
