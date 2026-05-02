# PRD: ControlNet Support (v1) — librediffusion

## Scope

Add ControlNet support to librediffusion as a minimally-destructive, additive set of changes that can be merged back into upstream `jcelerier/librediffusion` cleanly. v1 targets SD-Turbo + Canny ControlNet for img2img inference, single-step, single-batch — the realtime live-visuals configuration used by `td-librediffusion`.

This PRD covers library, engine-export tooling, and validation tooling changes. The TD operator changes that consume this work are specified in `td-librediffusion/PRD-CONTROLNET-V1.md`.

## Guiding principles

1. **Per-frame inference cost is the primary constraint.** This work targets realtime live concert visuals, not preview/iteration workflows.
2. **Zero perf cost when ControlNet is not enabled.** Users running today's plain UNet pipeline must see no measurable regression.
3. **Minimally destructive to upstream.** Prefer additive changes (new files, new methods, new C API entries). Avoid signature changes on existing functions.
4. **TD setup ergonomics is secondary** to the above two.

## Architecture

### Engine packaging — combined UNet + ControlNet

The ControlNet is folded into the UNet ONNX graph at export time, producing a single TensorRT engine (`unet_controlnet.engine`) whose schema extends today's UNet engine with two additional inputs:

- `control_image`: `[1, 3, H, W]` fp16 — the preprocessed control image
- `controlnet_strength`: `[1]` fp16 — scalar multiplier applied to ControlNet residuals before they are added to UNet skip connections

ControlNet residual addition happens **inside** the engine, so per-frame C++ changes are minimal: one extra input binding for the control image, one for the strength scalar. Residual `Mul` (strength) and `Add` (skip-connection injection) ops are expected to be fused by TensorRT — verified at engine build time (see "Strength control" below).

### Why combined-engine and not separate engines (B)

Separate-engine ControlNet (`controlnet.engine` + modified `unet.engine` with `down_block_additional_residuals` inputs) is the more flexible architecture and is the right path for multi-controlnet support. It is **explicitly deferred to v2/v3.** v1's combined-engine choice trades that flexibility for:

- Substantially smaller C++ surface (no `ControlNetWrapper` class, no residual-buffer plumbing, no two-stage forward, no CFG batching of residuals)
- TRT cross-boundary fusion (slightly faster per-frame)
- Smaller upstream merge diff

The cost: switching ControlNet variants (canny → depth → openpose) requires rebuilding the combined engine. Acceptable for v1 since the use case is "build the engine for the show, run it." Multi-controlnet (combinations) is unworkable under this architecture and is the trigger for v2's switch to separate engines.

### Preprocessing lives outside librediffusion

The control image is supplied to librediffusion already preprocessed (canny edges, depth map, etc.). librediffusion does not include canny/depth/openpose implementations. TouchDesigner has native primitives for this (`Edge TOP`, GLSL TOPs, CUDA TOPs), so the preprocessing burden lives there. This keeps librediffusion's scope clean and the same combined engine works for any preprocessing the user wires.

### v1 inference configuration lock

When the combined engine is loaded, the pipeline must be configured as:

- `batch_size = 1`
- `denoising_steps = 1`
- `frame_buffer_size = 1`
- `cfg_type ∈ {none, self}` (matches today's TD operator default of `self` with `guidance_scale=1.2`)
- `guidance_scale` is runtime-controllable (existing setter)

`init_buffers()` validates these and returns `LIBREDIFFUSION_ERROR_INVALID_ARGUMENT` with a clear message if any differ. Other configurations require the (B) separate-engine path (v2+).

This lock means `unet_batch_size = 1` always — no control-image tiling, no strength broadcast, fixed-shape TRT engine for max throughput. Streamdiffusion's batched-denoising mode (`frame_buffer_size > 1`) is the most consequential capability v1 gives up; users who rely on it for SD-Turbo image quality will see a regression.

## C API additions

All additions are purely additive. No existing symbol changes signature or behavior.

### Config

```c
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_config_set_combined_unet_controlnet_engine(
    librediffusion_config_handle config, const char* path);
```

When set (non-null, non-empty), this overrides `unet_engine_path` and switches the pipeline into combined-engine mode. When unset, pipeline behavior is **byte-for-byte identical to today** — same engine, same code path, same per-frame cost.

### Per-frame inputs

```c
/* CPU RGBA NHWC uint8 — convenience, matches librediffusion_img2img */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_control_image(
    librediffusion_pipeline_handle pipeline,
    const uint8_t* cpu_rgba_input,
    int width, int height);

/* GPU device pointer — hot path, matches librediffusion_img2img_gpu_half */
LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_control_image_gpu(
    librediffusion_pipeline_handle pipeline,
    const uint8_t* device_rgba_input,
    int width, int height,
    librediffusion_stream_t stream);

LIBREDIFFUSION_API librediffusion_error_t LIBREDIFFUSION_CALL
librediffusion_set_controlnet_strength(
    librediffusion_pipeline_handle pipeline, float strength);
```

#### Lifecycle — set-then-run (L1)

The control image is bound to a persistent internal buffer at `set_control_image*` time and consumed by the next `librediffusion_img2img_gpu_half` (or other inference function) call.

- If combined-engine mode is **off**: `set_control_image*` is a no-op. `set_controlnet_strength` is a no-op.
- If combined-engine mode is **on** and no control image has ever been bound: inference returns `LIBREDIFFUSION_ERROR_NOT_INITIALIZED`.
- If combined-engine mode is **on** and a previous control image was bound: reuse the previous one. Caller is not required to re-bind every frame.

#### Implementation constraint — no staging copies

`set_control_image_gpu` MUST perform NHWC RGBA uint8 → NCHW RGB fp16 conversion **directly into** the persistent control-image NCHW fp16 buffer. No intermediate device-to-device memcpy. No staging buffer. The conversion kernel writes to the engine input buffer in one pass.

#### Resolution policy

Control image dimensions must match the pipeline's configured `width` × `height`. Mismatches return `LIBREDIFFUSION_ERROR_INVALID_DIMENSIONS`. Auto-resize is the caller's responsibility (`test_engine.py` does this; the TD plugin enforces match at the operator level).

## Internal implementation notes

### LibreDiffusionConfig changes

Add one optional field:

```cpp
std::string combined_unet_controlnet_engine_path;  // empty == disabled
```

### LibreDiffusionPipeline changes

Add persistent buffers (allocated only when combined-engine mode is active):

```cpp
std::unique_ptr<CUDATensor<__half>> control_image_nchw_;     // [1, 3, H, W]
std::unique_ptr<CUDATensor<__half>> controlnet_strength_;    // [1]
```

`init_engines()` chooses `unet_engine_path` vs `combined_unet_controlnet_engine_path` based on which is set. The `UNetWrapper` either stays as today (plain UNet) or wraps a combined engine — the engine has more inputs, but the wrapper class can stay singular by parameterizing on whether the extra bindings exist. Prefer not to introduce a `CombinedUNetWrapper` class; minimize new types.

`predict_x0_batch` (in `librediffusion.unet.cpp`) gains one early branch:

```cpp
if (combined_engine_mode_) {
    // bind control_image_nchw_->data() to engine input "control_image"
    // bind controlnet_strength_->data() to engine input "controlnet_strength"
}
```

When `combined_engine_mode_` is false, this branch is a single predicted-not-taken comparison per inference — the zero-cost-when-off invariant.

### Strength control (S2 with TRT fusion verification)

Strength is implemented as a scalar engine input rather than baked at export time, to allow runtime modulation (audio-reactive, MIDI-mapped) which is a first-class live-visuals workflow. The expected runtime cost after TRT fuses the `Mul`/`Add` is effectively zero.

**Build-time verification step:** the export script dumps the TRT layer profile after building the engine and checks that the strength `Mul` is fused with adjacent `Add` operations. If fusion fails (unlikely), the measured per-frame cost (~10–30 μs on residual tensors) is logged. If non-trivial cost is observed, the fallback is to bake strength=1.0 at export (S0) and remove the runtime input — this is a re-export-only change, no C++ changes required.

### Zero-cost-when-off invariant

A baseline benchmark assertion is added: with `combined_unet_controlnet_engine_path` unset, the average per-frame inference time over N frames must match the pre-ControlNet baseline within ±2%. This guards against accidental regressions in the plain UNet code path during development.

## Engine export tooling

### New file: `tools/export_combined_unet_controlnet.py`

Standalone script (additive — does not modify `src/streamdiffusion/acceleration/tensorrt/`, which is the upstream-NVIDIA fork). May import `Optimizer` from that module read-only for ONNX simplification.

```
tools/export_combined_unet_controlnet.py

Args:
  --base-model      huggingface model id (e.g. stabilityai/sd-turbo)
  --controlnet      huggingface model id (e.g. lllyasviel/sd-controlnet-canny)
  --width, --height image dimensions (default 512)
  --batch-size      fixed at 1 for v1 (rejected if != 1)
  --output          output engine path (e.g. unet_controlnet_canny.engine)
  --fp16            (default true)
  --workspace-mb    TRT builder workspace (default 8192)

Steps:
  1. Load base UNet from diffusers
  2. Load ControlNet from ControlNetModel.from_pretrained
  3. Wrap in CombinedUNetControlNet(nn.Module):
       def forward(sample, timestep, encoder_hidden_states, control_image, controlnet_strength):
           down_residuals, mid_residual = controlnet(
               sample, timestep, encoder_hidden_states, control_image)
           down_residuals = [r * controlnet_strength for r in down_residuals]
           mid_residual = mid_residual * controlnet_strength
           return unet(sample, timestep, encoder_hidden_states,
                       down_block_additional_residuals=down_residuals,
                       mid_block_additional_residual=mid_residual).sample
  4. torch.onnx.export with fixed shapes for unet_batch_size=1
  5. ONNX simplification via Optimizer (from streamdiffusion.acceleration.tensorrt.models)
  6. TensorRT engine build
  7. Save engine + .meta.json sidecar
  8. Dump TRT layer profile, verify Mul/Add fusion (S2 verification)
```

### Sidecar (`<engine>.meta.json`)

Generated by the export script with every build. Contains:

```json
{
  "engine_kind": "combined_unet_controlnet",
  "base_model": "stabilityai/sd-turbo",
  "controlnet": "lllyasviel/sd-controlnet-canny",
  "width": 512,
  "height": 512,
  "batch_size": 1,
  "precision": "fp16",
  "tensorrt_version": "10.x.y",
  "built_at": "2026-05-01T..."
}
```

The TD plugin does **not** read the sidecar in v1 — routing is convention-based on engine filename. Sidecar exists for provenance and future-version validation/info display.

### v1 reference variant

Build target for v1 ship: `lllyasviel/sd-controlnet-canny` × `stabilityai/sd-turbo` → `unet_controlnet_canny.engine`.

### Escape hatch — smaller variants

If measured per-frame cost (~30–50% UNet overhead from full-size ControlNet) is unacceptable, fall back paths in order of distance:

1. **Distilled / pruned ControlNets** — same input/output contract as standard ControlNet; swap the `--controlnet` arg only. No code changes.
2. **T2I-Adapter** — different residual structure (encoder-only, fewer outputs). Requires a parallel export script (`tools/export_combined_unet_t2iadapter.py`); ~1–2 days of work. Per-frame cost drops to ~5–15% UNet overhead.
3. **ControlNet-LoRA** — most aggressive reduction; quality variable. Last resort.

All three keep the v1 C API and TD plugin unchanged — they're export-time choices.

## Validation tooling

### New file: `tools/test_engine.py`

ctypes-based Python script that loads the librediffusion shared library and runs one inference against a checked-in fixture. Purpose: smoke-test newly-built combined engines before loading them into TouchDesigner.

```
test_engine.py --engine PATH [other args]

Required:
  --engine PATH               combined engine to test
  --vae-encoder PATH
  --vae-decoder PATH
  --clip PATH

Optional with defaults:
  --source PATH               default: tools/test_data/source.png
  --control PATH              default: tools/test_data/canny.png
  --prompt STR                default: "a watercolor painting of a mountain landscape"
  --negative STR              default: "blurry, low quality, distorted"
  --strength FLOAT            default: 1.0
  --seed INT                  default: 42
  --output PATH               default: ./test_output.png
  --display                   open the result image after writing
```

Behavior: load DLL → build config → init pipeline + CLIP → load fixtures → run inference → save output → print timing. Auto-resize fixtures to match engine dimensions (the C API enforces resolution-must-match at the binding layer; the script resizes before the API call so one fixture set works against any engine resolution).

### Fixtures (committed)

- `tools/test_data/source.png` — 1024×1024 RGBA source image
- `tools/test_data/canny.png` — 1024×1024 RGBA canny edges of source (in-domain control signal)
- `tools/test_data/canny_of_different.png` — 1024×1024 RGBA canny from an unrelated image (out-of-domain control signal — the test that actually verifies ControlNet is consuming the control image)

### Output (not committed)

`test_output.png` and `tools/test_data/output*.png` added to `.gitignore`.

### Validation depth (v1)

Smoke-only — visual inspection. Numerical reference comparison against diffusers Python (V2/V3) is explicitly deferred. If smoke tests reveal subtle issues, the script can be extended later to optionally compare against a diffusers reference.

## Phasing

v1 (this PRD) → SD-Turbo + canny + img2img + single ControlNet + (A) combined engine.

Future phases, in likely order:

- **v2: SDXL combined engine.** Same architecture, separate engine file. Adds SDXL-specific conditioning (`text_embeds`, `time_ids`) to the combined-engine wrapper. No C API changes beyond a parallel `set_combined_unet_controlnet_engine` for SDXL (or the same setter, with model type detected from existing config).
- **v3: txt2img + ControlNet support in the TD operator.** Library already supports both — `librediffusion_txt2img_gpu` invokes the same `predict_x0_batch` and inherits combined-engine support automatically. Pure plugin-side addition.
- **v4: (B) separate-engine path + multi-controlnet.** Triggered when multi-controlnet becomes a real requirement. Adds `controlnet.engine` + modified `unet.engine` with `down_block_additional_residuals` inputs. New `ControlNetWrapper` class. New C API for loading N ControlNets and binding N control images. Multi-slot operator parameters in TD plugin.

## Out of scope for v1

- SDXL combined engine
- txt2img + ControlNet (library supports it; TD operator does not expose it)
- Multi-ControlNet
- `denoising_steps > 1`, `frame_buffer_size > 1`, `cfg_type = full/initialize`
- ControlNet `start/end` timestep range (would require both engines loaded simultaneously, which contradicts (A))
- Per-block strength (single global scalar only)
- Numerical reference validation against diffusers
- Auto-detection of engine type from sidecar or engine introspection
- Python bindings as a product (test_engine.py uses ctypes inline; no separate bindings module)

## Acceptance criteria

1. `tools/export_combined_unet_controlnet.py` produces a working `unet_controlnet_canny.engine` from `stabilityai/sd-turbo` + `lllyasviel/sd-controlnet-canny` at 512×512.
2. The export script writes `<engine>.meta.json` sidecar.
3. The export script reports TRT layer fusion status for the strength `Mul` operation.
4. `tools/test_engine.py` runs the engine end-to-end against the committed fixtures and produces a non-garbage output image.
5. The output image with `--control canny_of_different.png` visibly follows the canny structure rather than the source structure.
6. The C API additions (`set_combined_unet_controlnet_engine`, `set_control_image`, `set_control_image_gpu`, `set_controlnet_strength`) are added to `librediffusion_c.h` and implemented in `librediffusion_c.cpp`.
7. `init_buffers()` validates the v1 inference configuration lock and returns a clear error on violation.
8. Baseline benchmark assertion: with combined-engine mode off, average per-frame inference time matches pre-ControlNet baseline within ±2%.
9. The control image NHWC→NCHW conversion writes directly into the persistent NCHW fp16 buffer; no intermediate copies present in the code path.
10. All changes are additive in the diff against upstream — no existing function signatures, fields, or behaviors are modified.
