# 3. GPU/accelerator execution provider strategy

## Status

Accepted

## Context

motioncap is distributed as a single binary intended to run on hardware the author
doesn't control: NVIDIA desktops, AMD desktops/laptops, Intel-integrated-graphics
laptops, and Raspberry Pi devices. YOLO inference performance and idle power draw
both depend heavily on whether a GPU/accelerator is available and which vendor it's
from. The binary must work correctly (if more slowly) on hardware with no
acceleration at all.

The development machine has an NVIDIA GTX 1650 with drivers installed but no CUDA
toolkit (`nvcc`), and a system-installed `onnxruntime` package that ships only the CPU
execution provider (`libonnxruntime_providers_cuda.so` is not present) — confirming
that a system ONNX Runtime install cannot be relied on for GPU acceleration and the
`ort` crate's own provider binaries are needed instead.

## Decision

Use the `ort` crate (ONNX Runtime Rust bindings) with execution providers registered
in priority order: **CUDA → ROCm → OpenVINO → CPU**, built with `ort`'s `cuda`,
`rocm`, `openvino`, and `load-dynamic` Cargo features. `load-dynamic` allows the
binary to probe for each provider's runtime library at process start and skip
unavailable ones rather than failing to launch — this is what makes one binary work
across all target hardware. The active provider is logged at startup (e.g. `"NVIDIA
CUDA detected: using GPU inference"` / `"No supported accelerator found: falling back
to CPU inference"`) so it's visible which mode is active on a given machine.

Raspberry Pi has no CUDA/ROCm/OpenVINO-applicable GPU and falls back to CPU. This is
accepted as an adequate v1 baseline — YOLOv8n on CPU at reduced fps is sufficient for
a security trigger that doesn't need 30fps detection. Adding a Coral Edge TPU (or
similar USB accelerator) execution path for Pi was considered and explicitly deferred,
not designed for in v1.

OpenVINO was initially considered unnecessary complexity (narrower target: just an
NVIDIA desktop), then added once the target hardware matrix was clarified to include
laptops and Pi devices generally, not just the author's desktop.

## Consequences

- Each non-CPU provider requires its corresponding runtime library on the target
  machine (CUDA runtime, ROCm runtime, or OpenVINO runtime). Because `ort` detects
  availability per-provider and falls back gracefully, these are NOT treated as hard
  startup requirements the way `ffmpeg`/OpenCV are (see ADR 5) — their absence just
  means falling further down the priority list, not failing to start.
- The ONNX model file path must be configurable (not hardcoded), since it's a
  separately-obtained artifact (see model export note in the plan) and the shared
  binary can't assume where it lives on someone else's machine.
- Verifying which provider is actually active (vs. silently falling back) should be
  checked explicitly during testing (e.g. via `nvidia-smi` utilization during a
  detection event on an NVIDIA machine), since a silent fallback to CPU would be a
  performance regression that's easy to miss without watching for it.
