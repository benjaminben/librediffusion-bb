# librediffusion

A C++ / CUDA / TensorRT implementation of StreamDiffusion.

Designed as a real-time inference library, integrated into [ossia score](https://ossia.io) as the `StreamDiffusion` process.

## Requirements

- NVIDIA GPU, Turing architecture (sm_75) or newer
- NVIDIA driver supporting CUDA 13 (R570 or newer)
- CUDA Toolkit 13.0+
- TensorRT 10.14+ — the **development package** with headers and import libraries (the pip `tensorrt` wheel is not sufficient for building)
- CMake 3.18+ and Ninja
- Rust toolchain (stable, with the platform's default linker)
- Boost (header-only — only `boost::unordered::concurrent_flat_map` is used)
- Python 3.13 (for the engine compilation script)
- Visual Studio 2022, version 17.11 or newer (required for C++23 support)
- The "Desktop development with C++" workload installed
- [vcpkg](https://github.com/microsoft/vcpkg) for installing Boost

## Building the library

Clone the repo with submodules (CUTLASS):

```cmd
git clone --recurse-submodules https://github.com/jcelerier/librediffusion
cd librediffusion
```

Install Boost once via vcpkg:

```cmd
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg install boost-unordered:x64-windows
```

Then from the **"x64 Native Tools Command Prompt for VS 2022"**:

```cmd
cmake -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake ^
  -DTENSORRT_INCLUDE_DIR=<TensorRT-root>\include ^
  -DTENSORRT_LIB_DIR=<TensorRT-root>\lib
cmake --build build
```

Output: `build\librediffusion.dll`.

## Building TensorRT engines

Engines are model + GPU + TensorRT-version specific and must be built on (or for) the target machine. The Python script `train-lora.py` handles model download, ONNX export, and TensorRT compilation.

Set up the Python environment:

```cmd
uv sync
```

Or, in an existing conda/venv with Python 3.13:

```cmd
pip install uv
uv pip install -e . --index-strategy unsafe-best-match
```

Build engines for SD-Turbo at 512×512, single-step inference:

```cmd
python train-lora.py ^
  --type sd15 ^
  --model stabilityai/sd-turbo ^
  --output .\engines\sd-turbo-512 ^
  --min-batch 1 --max-batch 1 --opt-batch 1 ^
  --min-resolution 512 --max-resolution 512 ^
  --opt-height 512 --opt-width 512
```

The output folder will contain four `.engine` files: `clip.engine`, `unet.engine`, `vae_encoder.engine`, `vae_decoder.engine`.

### Choosing batch size

The `--min-batch / --max-batch / --opt-batch` flags control the batch dimension the engines support. When all three are equal **and** all resolution flags are equal, engines are built with **static shapes** for maximum performance.

The Score wrapper calls CLIP at batch=1 (one prompt at a time) and UNet at batch=N (where N is the number of denoising timesteps configured on the node). Choose engines accordingly:

- **Single-step (recommended for SD-Turbo):** `--min-batch 1 --max-batch 1 --opt-batch 1`. Set Score's `Timesteps` to a single value (e.g. `25`).
- **Multi-step (e.g. 2-step LCM):** `--min-batch 1 --max-batch 2 --opt-batch 2`. Set Score's `Timesteps` to two values (e.g. `15, 25`). Engines are dynamic-batch; `opt=2` means UNet's hot path is still fully optimized.

Building with `--min-batch 2 --max-batch 2` is **incompatible** with the Score wrapper: CLIP refuses batch-1 calls and inference fails silently.

## Using with ossia score

Score's package manager installs the addon's source repository without pre-built binaries. To run librediffusion in Score, the built shared library plus its TensorRT and CUDA runtime dependencies need to be staged into the addon folder.

### Locate the addon folder

After installing the LibreDiffusion package via Score's package manager, the addon folder is at:

```
%USERPROFILE%\Documents\ossia\score\packages\librediffusion
```

### Stage the runtime libraries

Copy these into the addon folder:

- `librediffusion.dll` (from your `build\` directory)
- All `*.dll` from `<TensorRT-root>\bin\` (the TensorRT runtime libraries — the `.lib` files in `lib\` are import libraries used at link time, the actual DLLs live in `bin\`)
- From the CUDA Toolkit's `bin\x64\` directory:
  - `cudart64_13.dll`
  - `curand64_10.dll`
  - `cublas64_13.dll`
  - `cublasLt64_13.dll`
  - `nppig64_13.dll`
  - `nppc64_13.dll`

### Launch Score with the addon folder on PATH

Score's `LoadLibrary` call only searches the executable directory and `PATH`. The addon folder is on neither by default, so launch Score with a wrapper that prepends the addon folder:

```cmd
@echo off
set ADDON=%USERPROFILE%\Documents\ossia\score\packages\librediffusion
set PATH=%ADDON%;%PATH%
"C:\Program Files\ossia score X.Y.Z\score.exe"
```

### Connect the StreamDiffusion node

In Score, add a `StreamDiffusion` process. Configure:

- `Engines`: path to the folder containing the four `.engine` files
- `Workflow`: variant matching your model — e.g. `SDTURBO_IMG2IMG` for SD-Turbo, `SDXL_IMG2IMG` for SDXL-Turbo
- `Resolution`: must match the resolution the engines were compiled for
- `Timesteps`: matched to the batch dimension your engines support (see "Choosing batch size" above)
- `Prompt +` and `Prompt -`: non-empty strings
- `Guidance`: typically `1.0` for Turbo models

Wire a Spout In device's texture to the node's `In` port, and route the `Out` port to a Spout Out device or any texture consumer.

## License

See [LICENSE](LICENSE).
