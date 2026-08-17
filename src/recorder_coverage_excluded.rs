//! Holds `RecordingEvent::start`/`finish` and the two ffmpeg-spawning free
//! functions they call (`spawn_video_encoder`, `mux_audio_into_video`), split
//! out of `recorder.rs` because each contains at least one `Err` arm that
//! genuinely cannot be reached by any safe test:
//!
//! - `spawn_video_encoder`'s and `mux_audio_into_video`'s `Command::spawn`/
//!   `.output()` calls failing to exec `ffmpeg` at all (as opposed to running
//!   and exiting nonzero, which is a distinct, testable arm) only happens if
//!   `ffmpeg` is absent from `PATH`, the same condition
//!   `startup::depcheck::check_ffmpeg`'s "not found" branch already refuses
//!   to start without (ADR 5). Faking that from a test would mean mutating
//!   the process's real `PATH` via `std::env::set_var`, which requires
//!   `unsafe` on current stable Rust; this crate denies `unsafe_code`
//!   outright (see `docs/adr/0006-coverage-exclusions.md`), and mutating
//!   global process state mid-suite would race every other test that shells
//!   out, regardless.
//! - `finish`'s `Child::wait` returning `Err` is, per the standard library's
//!   own Unix implementation, only reachable if the process was already
//!   reaped by something else calling `waitpid` on the same PID first
//!   (`ECHILD`). `RecordingEvent` is the sole owner of its `Child` and this
//!   crate has no `unsafe` escape hatch to reap a PID out from under it, so
//!   that arm is equally unreachable from safe code.
//!
//! Everything else in `start`/`finish` (temp-file creation, exit-status
//! handling, the mux step's own nonzero-exit handling, the class-list
//! rename, and the sidecar write) *is* reachable and was covered by tests
//! before this split; those tests moved here along with the functions they
//! exercise; `recorder.rs`'s own tests keep covering `ClipState`,
//! `resample_to_frame_rate`, `Sidecar`'s serde shape, and every other
//! `RecordingEvent` method (`seed`, `drain_frames`, `drain_audio`,
//! `write_frame`, `write_audio`, `record_detection`, `touch`,
//! `record_motion`, `quiet_for`, `camera_stalled`), none of which have an
//! irreducible gap of this kind.
//!
//! Split out of `recorder.rs` (same convention as `coverage_excluded.rs` at
//! the crate root, and `capture/audio_coverage_excluded.rs`) so that file
//! reports genuine 100% coverage instead of being held down by a handful of
//! `Err`-only arms nothing can safely trigger. See
//! `docs/adr/0006-coverage-exclusions.md`.

use std::process::{Child, Command, Stdio};

use anyhow::{Context, Result, bail};

use crate::paths::{clip_path, sidecar_path};
use crate::recorder::{ClipState, RecordingEvent, RecordingEventParams, Sidecar};

impl RecordingEvent {
    /// Starts a new event: launches the video-encoding ffmpeg process and
    /// creates the temp audio file. `final_clip_path` should already reflect
    /// the trigger's classes/timestamp per the output layout convention
    /// (ADR 4); this module only performs the actual encoding/muxing, not
    /// path naming.
    ///
    /// `audio_sample_rate`/`audio_channels` must match whatever format the
    /// caller's audio samples were captured/converted to (see
    /// `capture::audio::AudioStreamInfo`), since the mux step needs the real
    /// values to interpret the buffered PCM correctly.
    ///
    /// Deliberately does *not* write the pre-buffer here. See `seed`.
    /// Writing dozens of frames synchronously on the caller's thread would
    /// block it for long enough that real wall-clock time passes with
    /// nothing being recorded, which shows up as a skip/jump right at the
    /// start of the clip once live writing resumes. Callers should invoke
    /// `seed` from whatever thread is responsible for steady-paced frame
    /// writing instead.
    ///
    /// `clip_timeline_start` should be the capture timestamp of the earliest
    /// pre-buffer frame that will be seeded into this clip (i.e. the actual
    /// start of the video file), not the trigger instant. It's the zero
    /// point `DetectionRecord::offset_secs` is measured from, so detections
    /// recorded against footage that predates the trigger (the pre-buffer
    /// window) get correct nonzero offsets instead of all reading ~0.
    ///
    /// `output_dir`/`started_at` must be the same values `final_clip_path`
    /// was originally built from via `clip_path`, so `finish` can recompute
    /// the filename with the full class list accumulated over the clip's
    /// lifetime (ADR 4) and rename to it if it grew since `start`.
    pub fn start(params: RecordingEventParams) -> Result<Self> {
        let RecordingEventParams {
            final_clip_path,
            output_dir,
            started_at,
            width,
            height,
            frame_rate,
            audio_sample_rate,
            audio_channels,
            clip_timeline_start,
        } = params;

        let video_tmp_path = final_clip_path.with_extension("video.tmp.mp4");
        let audio_tmp_path = final_clip_path.with_extension("audio.tmp.pcm");

        let ffmpeg_video = spawn_video_encoder(&video_tmp_path, width, height, frame_rate)?;
        let audio_file = std::fs::File::create(&audio_tmp_path).with_context(|| {
            format!(
                "failed to create temp audio file {}",
                audio_tmp_path.display()
            )
        })?;

        Ok(Self {
            ffmpeg_video,
            audio_tmp_path,
            audio_file,
            final_clip_path,
            output_dir,
            started_at,
            video_tmp_path,
            audio_sample_rate,
            audio_channels,
            state: ClipState::new(frame_rate, clip_timeline_start),
        })
    }

    /// Stops encoding, muxes the buffered audio into the video, and writes
    /// the JSON sidecar alongside the final clip.
    pub fn finish(mut self) -> Result<()> {
        drop(self.ffmpeg_video.stdin.take());

        let mut stderr = String::new();

        if let Some(mut stderr_pipe) = self.ffmpeg_video.stderr.take() {
            let _ = std::io::Read::read_to_string(&mut stderr_pipe, &mut stderr);
        }

        let status = self
            .ffmpeg_video
            .wait()
            .context("ffmpeg video encoder failed")?;

        if !status.success() {
            bail!(
                "ffmpeg video encoder exited with {status}: {}",
                stderr.trim()
            );
        }

        drop(self.audio_file);

        mux_audio_into_video(
            &self.video_tmp_path,
            &self.audio_tmp_path,
            &self.final_clip_path,
            self.audio_sample_rate,
            self.audio_channels,
        )?;

        let _ = std::fs::remove_file(&self.video_tmp_path);
        let _ = std::fs::remove_file(&self.audio_tmp_path);

        let (all_classes_set, detections, motion_events) = self.state.into_parts();
        let all_classes: Vec<&str> = all_classes_set.iter().copied().collect();
        let renamed_path = clip_path(&self.output_dir, self.started_at, &all_classes)?;

        if renamed_path != self.final_clip_path {
            std::fs::rename(&self.final_clip_path, &renamed_path).with_context(|| {
                format!(
                    "failed to rename clip {} to {} (reflecting every class detected during the clip, per ADR 4)",
                    self.final_clip_path.display(),
                    renamed_path.display()
                )
            })?;
            self.final_clip_path = renamed_path;
        }

        let sidecar = Sidecar {
            detections,
            motion_events,
        };

        let sidecar_json =
            serde_json::to_string_pretty(&sidecar).context("failed to serialize sidecar")?;

        std::fs::write(sidecar_path(&self.final_clip_path), sidecar_json)
            .context("failed to write sidecar file")?;

        Ok(())
    }
}

/// Spawns ffmpeg to encode raw RGB frames fed via stdin into an H.264 file.
fn spawn_video_encoder(
    path: &std::path::Path,
    width: u32,
    height: u32,
    frame_rate: u32,
) -> Result<Child> {
    #[cfg(unix)]
    use std::os::unix::process::CommandExt;

    let mut command = Command::new("ffmpeg");

    command
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "rawvideo", "-pixel_format", "rgb24"])
        .args(["-video_size", &format!("{width}x{height}")])
        .args(["-framerate", &frame_rate.to_string()])
        .args(["-i", "-"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    // Put ffmpeg in its own process group so a terminal SIGINT (Ctrl+C)
    // doesn't reach it directly. It shares the foreground process group
    // with motioncap by default, so without this, Ctrl+C kills ffmpeg at
    // the same instant as motioncap's own ctrlc handler tries to close it
    // gracefully (closing stdin, then waiting), racing ffmpeg's own SIGINT
    // handling and producing a nonzero exit even when the output file is
    // actually complete and valid. Graceful shutdown should be the only
    // thing that ever tells ffmpeg to stop.
    #[cfg(unix)]
    command.process_group(0);

    command
        .spawn()
        .context("failed to spawn ffmpeg video encoder")
}

/// Muxes the buffered raw audio into the encoded video. `-shortest` is
/// deliberately not used: with independently-accumulated video/audio streams,
/// whichever stream is shorter due to minor drift would otherwise have the
/// *other* stream silently truncated to match, losing recorded content.
/// Instead, the audio stream is padded with silence (`apad`) to at least the
/// video's duration and `-shortest` is applied only to that padded output, so
/// the result is exactly the video's length with no dropped video frames.
fn mux_audio_into_video(
    video_path: &std::path::Path,
    audio_path: &std::path::Path,
    output_path: &std::path::Path,
    sample_rate: u32,
    channels: u16,
) -> Result<()> {
    let output = Command::new("ffmpeg")
        .args(["-y", "-i"])
        .arg(video_path)
        .args([
            "-f",
            "f32le",
            "-ar",
            &sample_rate.to_string(),
            "-ac",
            &channels.to_string(),
        ])
        .arg("-i")
        .arg(audio_path)
        .args(["-c:v", "copy", "-af", "apad", "-c:a", "aac", "-shortest"])
        .arg(output_path)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .context("failed to spawn ffmpeg for audio muxing")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);

        bail!(
            "ffmpeg audio mux exited with {}: {}",
            output.status,
            stderr.trim()
        );
    }

    Ok(())
}
