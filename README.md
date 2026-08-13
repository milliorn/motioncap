# motioncap

A webcam-based security motion-capture tool written in Rust. It runs as a long-lived
background process: continuous camera/audio capture, background-subtraction motion
detection, YOLO object-detection confirmation, and H.264+AAC video recording with a
pre/post event buffer.

It's distributed as a single binary meant to run on hardware the author doesn't
control (NVIDIA/AMD/Intel/Raspberry Pi), so graceful degradation across missing
accelerators/dependencies is a first-class concern rather than an edge case.

## How it works

1. Camera and audio are captured continuously into a rolling ring buffer.
2. A background-subtraction motion gate (OpenCV MOG2) watches for changed pixels.
3. When the gate trips, a YOLO model confirms a living thing (person or any COCO
   animal class) is actually present before anything is recorded.
4. On confirmation, a clip starts — seeded with footage from *before* the trigger —
   and keeps recording through a quiet post-event window before closing.

Motion alone never triggers a recording; only a confirmed YOLO detection does. See
[`docs/adr/0002-two-path-trigger-design.md`](docs/adr/0002-two-path-trigger-design.md)
for the reasoning.

## Requirements

- **`ffmpeg`** must be on `PATH`. Used for video encoding and audio muxing (invoked as
  a subprocess). Checked at startup — motioncap prints install instructions and exits
  rather than installing anything itself.
- **OpenCV** must be installed and discoverable via pkg-config at *compile* time.
  Development builds target OpenCV 5.0 (`opencv5.pc`), which is newer than the
  `opencv4` pkg-config name many guides assume.
- **A YOLO model in ONNX format** at `models/yolov8n.onnx` (configurable via
  `--model-path`). Not bundled — obtain it separately.

See [`docs/adr/0001-rust-preferred-not-absolute.md`](docs/adr/0001-rust-preferred-not-absolute.md)
and [`docs/adr/0005-missing-dependency-handling.md`](docs/adr/0005-missing-dependency-handling.md)
for why these are handled this way.

## Build

```fish
cargo build                      # debug build
cargo build --release            # release build
```

## Run

```fish
cargo run -- [flags]
```

Useful flags (full list in `src/config.rs`):

| Flag | Description |
| --- | --- |
| `--preview` | Opens a live OpenCV window showing the raw feed (diagnostic only, never affects recording) |
| `--force-cpu` | Skip GPU execution provider probing |
| `--camera-device /dev/videoN` | Pin a camera instead of auto-detecting |
| `--output-dir <dir>` | Where recordings are written (default `./recordings`) |
| `--model-path <path>` | Path to the YOLO ONNX model (default `./models/yolov8n.onnx`) |
| `--pre-buffer-secs <n>` | Seconds of footage to keep buffered before a trigger (default 10) |
| `--post-buffer-secs <n>` | Seconds to keep recording after the last trigger (default 15) |
| `--detection-confidence <0.0-1.0>` | Minimum YOLO confidence to confirm a detection (default 0.3) |
| `--motion-threshold <0.0-1.0>` | Minimum changed-pixel ratio for the motion gate to trip (default 0.01) |

GPU acceleration is probed in order CUDA → ROCm → OpenVINO → CPU and falls back
automatically; see
[`docs/adr/0003-gpu-execution-provider-strategy.md`](docs/adr/0003-gpu-execution-provider-strategy.md).

### Enabling GPU acceleration (optional)

Falling back to CPU is a normal, supported outcome — motioncap works either way, just
slower on CPU. Nothing below is required to build or run it.

GPU acceleration only activates if the ONNX Runtime shared library motioncap loads at
startup was itself built with that GPU vendor's execution provider compiled in. A
Linux distro's packaged `onnxruntime` (e.g. Arch's `extra/onnxruntime`) is commonly
CPU-only — installing just the NVIDIA/AMD/Intel driver is **not** sufficient, and the
fallback to CPU happens silently, with no warning or error.

To enable NVIDIA CUDA specifically:

1. Install a CUDA toolkit + cuDNN matching your driver (e.g. on Arch: `sudo pacman -S
   cuda cudnn`). Check the
   [ONNX Runtime CUDA execution provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
   for the CUDA/cuDNN version your ONNX Runtime release expects.
2. Download the matching GPU release tarball from the
   [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases)
   (e.g. `onnxruntime-linux-x64-gpu_cuda13-<version>.tgz`) and extract it somewhere
   local, such as `vendor/` in this repo (gitignored — not something you commit).
3. Point motioncap at that build's `libonnxruntime.so` via the `ORT_DYLIB_PATH`
   environment variable before running it:

   ```fish
   set -x ORT_DYLIB_PATH /path/to/onnxruntime-linux-x64-gpu_cuda13-<version>/lib/libonnxruntime.so
   cargo run --release
   ```

   Set it permanently with `set -Ux ORT_DYLIB_PATH ...` (fish) if you want every
   future run to use it without re-exporting it each session.
4. Verify it actually took effect — the fallback is silent, so a successful launch
   alone doesn't confirm GPU is in use. While motioncap is running, check:

   ```fish
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
   ```

   motioncap should appear in that list with nonzero GPU memory.

ROCm and OpenVINO follow the same pattern (a GPU-enabled ONNX Runtime build,
discoverable via `ORT_DYLIB_PATH`), against their own vendor's toolkit/runtime
instead of CUDA.

## Output layout

```sh
<output_dir>/<YYYY-MM-DD>/<YYYY-MM-DD_HH-MM-SS>_<class1>_<class2>.mp4
```

Each clip has a same-named `.json` sidecar recording per-detection offsets, classes,
and confidence scores, so future search/debugging never needs to re-run inference on
archived footage. Classes in the filename are deduplicated and sorted alphabetically.
Retention/pruning is out of scope — disk space is managed manually. See
[`docs/adr/0004-storage-layout-and-retention.md`](docs/adr/0004-storage-layout-and-retention.md).

## Development

```fish
cargo clippy     # lint
cargo fmt        # format
```

There is no test suite yet.

Design rationale for anything non-obvious in the codebase is recorded in
[`docs/adr/`](docs/adr/) — read the relevant ADR before changing behavior in that area.

## License

MIT — see [LICENSE](LICENSE).
