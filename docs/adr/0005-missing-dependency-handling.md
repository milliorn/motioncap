# 5. Missing hard-dependency handling: detect and abort, never auto-install

## Status

Accepted

## Context

`ffmpeg` and OpenCV (see ADR 1) are hard runtime requirements that cannot be bundled
into the binary. Since motioncap is distributed to people running unknown systems,
either dependency may be missing on a given machine. Several ways of handling a
missing dependency were considered: silently auto-installing via the system package
manager, interactively prompting to install (possibly via `sudo`), or detecting and
failing with instructions.

## Decision

At startup, probe for `ffmpeg` (via subprocess invocation) and the configured ONNX
model file path. If anything required is missing, print exactly what's missing and
the precise install command for common platforms (e.g. `sudo apt install ffmpeg`),
then exit non-zero. **The program never invokes `sudo` or a package manager itself**,
whether silently or via an interactive prompt.

OpenCV is not part of this runtime check: unlike `ffmpeg` (invoked as a subprocess at
run time), OpenCV is linked at compile time via the `opencv` crate. If OpenCV isn't
installed, the binary fails to *build* with a clear linker/pkg-config error — there is
no scenario where a successfully-built motioncap binary is running without OpenCV
present, so a runtime check for it would be dead code.

This was chosen over auto-install/interactive-install for two concrete reasons:

- **Headless/service-mode breakage.** `sudo` needs a TTY to prompt for a password (or
  a pre-configured passwordless-sudo/polkit setup). motioncap is meant to potentially
  run as an always-on background service (e.g. a systemd unit) — a very plausible
  deployment mode for a security tool — where there is no terminal for `sudo` to
  prompt on. An install flow that depends on an interactive prompt would simply hang
  or fail in exactly the deployment mode this project is aimed at.
- **Trust cost.** A security-relevant tool that invokes `sudo` on its own initiative,
  even with a per-run `y/N` confirmation, asks more of a stranger's trust than a tool
  that only ever tells them the exact command to run themselves. Since this binary is
  shared with people who didn't write it, earning that trust by never touching the
  system uninvited was judged more valuable than a smoother first-run experience.

## Consequences

- Setup requires one extra manual step from the user (running the printed install
  command) on a machine missing `ffmpeg`, rather than the program fixing it
  automatically.
- The dependency-check code only needs to detect presence and construct instruction
  text per platform — it never needs privilege-escalation or package-manager
  invocation logic, keeping that code path simpler and safer.
- This does not apply to GPU execution providers (CUDA/ROCm/OpenVINO runtime
  libraries) — see ADR 3 — since those degrade gracefully to a CPU fallback rather
  than being hard requirements, so there's nothing to abort over.
