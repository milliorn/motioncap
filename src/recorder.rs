use std::io::Write;
use std::process::Child;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Local};

use crate::buffer::{TimestampedAudio, TimestampedFrame};
use crate::clip_state::ClipState;
use crate::ffmpeg::{mux_audio_into_video, resample_to_frame_rate, spawn_video_encoder};
use crate::paths::{clip_path, sidecar_path};
use crate::sidecar::Sidecar;

/// Construction parameters for `RecordingEvent::start`, grouped into a
/// struct since the individual values (video dimensions, audio format,
/// path-naming inputs, timeline anchor) don't share a natural owner.
pub struct RecordingEventParams {
    /// Where the finished, muxed clip is written, before any class-list
    /// rename (see `RecordingEvent::all_classes`).
    pub final_clip_path: std::path::PathBuf,
    /// Directory recordings are written under (ADR 4), needed alongside
    /// `started_at` to recompute the filename at `finish` if the class list
    /// grows over the clip's lifetime.
    pub output_dir: std::path::PathBuf,
    /// Wall-clock time the clip started; must match what `final_clip_path`
    /// was built from via `clip_path`.
    pub started_at: DateTime<Local>,
    /// Video frame width in pixels.
    pub width: u32,
    /// Video frame height in pixels.
    pub height: u32,
    /// Configured video frame rate the encoder is spawned with.
    pub frame_rate: u32,
    /// Sample rate of the audio that will be written, needed to mux correctly.
    pub audio_sample_rate: u32,
    /// Channel count of the audio that will be written, needed to mux correctly.
    pub audio_channels: u16,
    /// Capture timestamp of the clip's first frame (the start of the
    /// pre-buffer window, not the trigger instant), used as the zero point
    /// for `DetectionRecord::offset_secs`.
    pub clip_timeline_start: Instant,
}

/// Manages the lifecycle of a single recorded clip: seeds the file with the
/// pre-event buffer, accepts live frames as they arrive, and tracks the
/// post-event quiet window so the caller knows when to close it (ADR 2, ADR 4).
///
/// Audio is buffered to a temporary raw PCM file and muxed in by ffmpeg only
/// when the clip closes (`finish`), since ffmpeg needs the final duration of
/// both streams up front to mux them correctly; video frames, by contrast,
/// are streamed live via ffmpeg's stdin as they arrive.
pub struct RecordingEvent {
    /// The running ffmpeg process encoding video, fed live via its stdin.
    ffmpeg_video: Child,
    /// Path to the temporary raw PCM file audio is buffered into.
    audio_tmp_path: std::path::PathBuf,
    /// Open handle to `audio_tmp_path`, written to as audio arrives.
    audio_file: std::fs::File,
    /// Where the finished, muxed clip is written on `finish`, before any
    /// class-list rename (see `all_classes`).
    final_clip_path: std::path::PathBuf,
    /// Directory recordings are written under, needed to recompute
    /// `final_clip_path` at `finish` if `all_classes` grew since `start`.
    output_dir: std::path::PathBuf,
    /// Wall-clock time the clip started, needed (alongside `output_dir`) to
    /// recompute `final_clip_path` at `finish` via the same `clip_path`
    /// naming convention it was originally built with.
    started_at: DateTime<Local>,
    /// Path to the temporary video-only file ffmpeg encodes into.
    video_tmp_path: std::path::PathBuf,
    /// Sample rate of the buffered audio, needed to mux correctly.
    audio_sample_rate: u32,
    /// Channel count of the buffered audio, needed to mux correctly.
    audio_channels: u16,
    /// Pure timing/bookkeeping state for this clip (see `ClipState`).
    state: ClipState,
}

impl RecordingEvent {
    /// Exposes the pure `ClipState` for tests elsewhere (e.g.
    /// `event_lifecycle.rs`) that need to reach its test-only helpers (e.g.
    /// `backdate_last_real_frame_at_past_stall`) directly.
    #[cfg(test)]
    pub(crate) const fn state_mut(&mut self) -> &mut ClipState {
        &mut self.state
    }

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
    ///
    /// Contains at least one `Err` arm (`spawn_video_encoder`'s/
    /// `Command::spawn`'s exec failure) reachable only if `ffmpeg` is absent
    /// from `PATH`, the same condition `startup::depcheck::check_ffmpeg`'s
    /// "not found" branch already refuses to start without (ADR 5). Faking
    /// that from a test would mean mutating the process's real `PATH` via
    /// `std::env::set_var`, which requires `unsafe` on current stable Rust;
    /// this crate denies `unsafe_code` outright, and mutating global process
    /// state mid-suite would race every other test that shells out,
    /// regardless. Everything else here (temp-file creation) is reachable
    /// and covered by tests below.
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
    ///
    /// Contains one further irreducible `Err` arm: `Child::wait` returning
    /// `Err` is, per the standard library's own Unix implementation, only
    /// reachable if the process was already reaped by something else
    /// calling `waitpid` on the same PID first (`ECHILD`). `RecordingEvent`
    /// is the sole owner of its `Child` and this crate has no `unsafe`
    /// escape hatch to reap a PID out from under it, so that arm is equally
    /// unreachable from safe code. Every other path here (ffmpeg exit
    /// status, the mux step's own nonzero-exit handling, the class-list
    /// rename, and the sidecar write) is reachable and covered by
    /// fault-injection tests below.
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

    /// Writes the pre-event buffer (frames captured before the trigger
    /// fired) into a freshly-started event. Must be called once, as the
    /// first write against a new event, before any live `write_frame`/
    /// `drain_audio` calls. See the note on `start` for why this is
    /// separate from event construction.
    ///
    /// The ring buffer accumulates frames at the camera's native capture
    /// rate, which may be higher than this event's configured `frame_rate`
    /// (set in `start`, the rate the video encoder was spawned with).
    /// Pre-buffered frames are resampled down to that rate using their
    /// timestamps before writing, so the pre-buffer portion of the clip plays
    /// back at the correct real-time duration instead of being stretched by
    /// writing every captured frame 1:1.
    pub fn seed(
        &mut self,
        pre_frames: &[TimestampedFrame],
        pre_audio: &[TimestampedAudio],
    ) -> Result<()> {
        let selected = resample_to_frame_rate(pre_frames, self.state.frame_rate);

        for frame in &selected {
            self.write_frame(&frame.image)?;
        }

        if !selected.is_empty() {
            self.state.last_real_frame_at = Instant::now();
        }

        if let Some(last) = selected.last() {
            self.state.anchor_next_tick_after_seed(last.timestamp);
        }

        for chunk in pre_audio {
            self.write_audio(&chunk.samples)?;
        }

        if let Some(last) = pre_audio.last() {
            self.state.last_audio_drain_at = last.timestamp;
        }

        Ok(())
    }

    /// Writes every frame captured since the last drain (or since `seed`,
    /// for the first call), resampled to this event's configured frame rate
    /// using a persistent tick anchor carried across calls. Must be polled
    /// periodically while the event is active.
    ///
    /// Draining *all* new frames (not just the newest) matters because the
    /// camera's delivery rate can momentarily exceed the writer's poll rate;
    /// reading only the latest frame each poll silently skips over whatever
    /// arrived and was superseded in between, which produces a visible jump
    /// in the subject's position despite otherwise-correct frame timing.
    ///
    /// This is a security recording: it must never paper over a gap in
    /// coverage by fabricating footage. If the camera doesn't deliver a real
    /// frame for a tick that's already due by wall-clock `Instant::now()`,
    /// this does *not* duplicate the last frame to fill it. It stops
    /// writing and leaves `camera_stalled` reporting true so the caller ends
    /// the recording instead of the clip silently containing footage that
    /// was never actually captured.
    pub fn drain_frames(
        &mut self,
        ring_buffer: &std::sync::Mutex<crate::buffer::RingBuffer>,
    ) -> Result<()> {
        let new_frames = {
            let buf = ring_buffer.lock().expect("ring buffer lock poisoned");
            buf.frames_since(self.state.last_frame_drain_at)
        };

        if let Some(last) = new_frames.last() {
            self.state.last_frame_drain_at = last.timestamp;
        }

        if !new_frames.is_empty() {
            self.state.last_real_frame_at = Instant::now();
        }

        for frame in &new_frames {
            if self.state.should_write_frame(frame.timestamp) {
                self.write_frame(&frame.image)?;
            }
        }

        Ok(())
    }

    /// True once the camera has gone `MAX_FRAME_STALL` without delivering a
    /// real frame. Callers should treat this as a signal to end the
    /// recording: `drain_frames` never fabricates a frame to cover for the
    /// camera, and the motion gate has nothing new to evaluate once frames
    /// stop arriving, so nothing else will naturally close the clip.
    pub fn camera_stalled(&self) -> bool {
        self.state.camera_stalled()
    }

    /// Writes any audio captured since the last drain (or since the event
    /// started, for the first call) into the temp PCM file. Must be polled
    /// periodically while the event is active so live audio keeps
    /// accumulating for the clip's full duration, not just the pre-buffer
    /// window written in `start`.
    pub fn drain_audio(
        &mut self,
        ring_buffer: &std::sync::Mutex<crate::buffer::RingBuffer>,
    ) -> Result<()> {
        let new_chunks = {
            let buf = ring_buffer.lock().expect("ring buffer lock poisoned");
            buf.audio_since(self.state.last_audio_drain_at)
        };

        if let Some(last) = new_chunks.last() {
            self.state.last_audio_drain_at = last.timestamp;
        }

        for chunk in &new_chunks {
            self.write_audio(&chunk.samples)?;
        }

        Ok(())
    }

    /// Streams one raw video frame to ffmpeg's stdin.
    pub fn write_frame(&mut self, image: &image::RgbImage) -> Result<()> {
        let stdin = self
            .ffmpeg_video
            .stdin
            .as_mut()
            .context("ffmpeg stdin was already closed")?;

        stdin
            .write_all(image.as_raw())
            .context("failed to write frame to ffmpeg")?;
        Ok(())
    }

    /// Appends raw PCM samples to the temp audio file.
    pub fn write_audio(&mut self, samples: &[f32]) -> Result<()> {
        let mut bytes = Vec::with_capacity(samples.len().saturating_mul(size_of::<f32>()));

        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        self.audio_file
            .write_all(&bytes)
            .context("failed to write audio samples to temp file")?;
        Ok(())
    }

    /// Records a detection into the sidecar and resets the post-buffer quiet
    /// window. `frame_timestamp` is the capture timestamp of the frame the
    /// detection ran against (from the ring buffer), used to compute
    /// `offset_secs` relative to the clip's actual timeline start rather than
    /// wall-clock time at the moment this function happens to run.
    pub fn record_detection(
        &mut self,
        class_name: &'static str,
        confidence: f32,
        frame_timestamp: Instant,
    ) {
        self.state
            .record_detection(class_name, confidence, frame_timestamp);
    }

    /// Resets the post-buffer quiet window without recording a new detection
    /// (e.g. motion continues but wasn't re-confirmed by YOLO this tick).
    pub fn touch(&mut self) {
        self.state.touch();
    }

    /// Records a motion-gate trip into the sidecar for later audit (e.g.
    /// distinguishing a fan/curtain repeatedly tripping the gate from an
    /// actual subject). Purely diagnostic bookkeeping. It does not itself
    /// touch the post-buffer quiet window. Callers already call
    /// `touch`/`record_detection` for that as needed.
    pub fn record_motion(&mut self, changed_ratio: f32, frame_timestamp: Instant) {
        self.state.record_motion(changed_ratio, frame_timestamp);
    }

    /// How long it's been since the last trigger/touch.
    pub fn quiet_for(&self) -> Duration {
        self.state.quiet_for()
    }
}

#[cfg(test)]
mod tests {
    //! Inline ffmpeg-backed round-trip and fault-injection tests for
    //! `RecordingEvent::start`/`seed`/`drain_frames`/`drain_audio`/`finish`
    //! (no top-level `tests/` directory exists for this binary-only crate,
    //! see ADR 7). Pure bookkeeping tests for `ClipState` and
    //! `resample_to_frame_rate` live in `clip_state.rs` and `ffmpeg.rs`
    //! respectively, alongside the code they exercise.
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        clippy::unchecked_time_subtraction,
        reason = "test assertions favor unwrap/indexing/plain time arithmetic for clarity; test \
                   durations are small hardcoded constants, so underflow is not reachable"
    )]

    use std::process::Command;

    use super::*;
    use crate::paths::{clip_path, sidecar_path};

    // --- RecordingEvent end-to-end (real ffmpeg subprocess) ---

    /// Runs `ffprobe` against `path` and returns its parsed JSON stream info,
    /// to confirm the muxed output is a real, playable file rather than just
    /// a file that happens to exist on disk.
    fn ffprobe_streams(path: &std::path::Path) -> serde_json::Value {
        let output = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
            ])
            .arg(path)
            .output()
            .expect("failed to spawn ffprobe");
        // Built unconditionally (not passed as a lazy assert! message arg) so
        // this line is exercised on every run, not just an ffprobe failure.
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        assert!(output.status.success(), "ffprobe failed: {stderr}");
        serde_json::from_slice(&output.stdout).expect("ffprobe produced invalid JSON")
    }

    /// A 2x2 timestamped frame; libx264 requires even width/height, so this
    /// is used for the ffmpeg round-trip tests below.
    fn even_sized_frame_at(timestamp: Instant) -> TimestampedFrame {
        TimestampedFrame {
            timestamp,
            image: std::sync::Arc::new(image::RgbImage::new(2, 2)),
        }
    }

    /// Starts a `RecordingEvent` with the field values shared by most tests
    /// in this module (2x2 frames, 5fps, 8kHz mono audio), overriding only
    /// the fields a given test needs to vary.
    fn start_event(
        dir: &std::path::Path,
        final_clip_path: std::path::PathBuf,
        started_at: DateTime<Local>,
        clip_timeline_start: Instant,
        width: u32,
        height: u32,
        audio_sample_rate: u32,
    ) -> Result<RecordingEvent> {
        RecordingEvent::start(RecordingEventParams {
            final_clip_path,
            output_dir: dir.to_path_buf(),
            started_at,
            width,
            height,
            frame_rate: 5,
            audio_sample_rate,
            audio_channels: 1,
            clip_timeline_start,
        })
    }

    /// Seeds `event` with a single even-sized frame and a small audio chunk,
    /// both timestamped at `clip_timeline_start`. Used by tests that only
    /// need *some* pre-buffer content written, not its exact size or timing.
    fn seed_one_frame(event: &mut RecordingEvent, clip_timeline_start: Instant) {
        event
            .seed(
                &[even_sized_frame_at(clip_timeline_start)],
                &[TimestampedAudio {
                    timestamp: clip_timeline_start,
                    samples: vec![0.0; 100],
                }],
            )
            .unwrap();
    }

    #[test]
    fn recording_event_round_trip_produces_valid_mp4_and_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();

        // finish() recomputes the final filename from the full accumulated
        // class list (ADR 4) via the same clip_path() helper; the initial
        // final_clip_path only matters as the pre-rename name, so predict the
        // real final path with clip_path() up front rather than guessing it.
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();
        let expected_final_path = clip_path(dir.path(), started_at, &["person"]).unwrap();

        let mut event = RecordingEvent::start(RecordingEventParams {
            final_clip_path: initial_path,
            output_dir: dir.path().to_path_buf(),
            started_at,
            width: 2,
            height: 2,
            frame_rate: 5,
            audio_sample_rate: 8000,
            audio_channels: 1,
            clip_timeline_start,
        })
        .unwrap();

        let pre_frames = vec![
            even_sized_frame_at(clip_timeline_start),
            even_sized_frame_at(clip_timeline_start + Duration::from_millis(250)),
        ];
        let pre_audio = vec![TimestampedAudio {
            timestamp: clip_timeline_start,
            samples: vec![0.0; 800],
        }];

        event.seed(&pre_frames, &pre_audio).unwrap();

        event.record_motion(0.02, clip_timeline_start);
        event.record_detection("person", 0.95, clip_timeline_start);
        assert!(event.quiet_for() < Duration::from_secs(1));
        event.touch();

        event.finish().unwrap();

        // Built unconditionally (not passed as a lazy assert! message arg) so
        // this line is exercised on every run, not just a missing-file failure.
        let expected_final_path_display = expected_final_path.display().to_string();
        assert!(
            expected_final_path.exists(),
            "expected clip at {expected_final_path_display}"
        );

        let sidecar_json = std::fs::read_to_string(sidecar_path(&expected_final_path)).unwrap();
        let sidecar: serde_json::Value = serde_json::from_str(&sidecar_json).unwrap();
        assert_eq!(sidecar["detections"].as_array().unwrap().len(), 1);
        assert_eq!(sidecar["motion_events"].as_array().unwrap().len(), 1);
        assert_eq!(sidecar["detections"][0]["class_name"], "person");

        let probe = ffprobe_streams(&expected_final_path);
        let streams = probe["streams"].as_array().unwrap();
        assert!(
            streams.iter().any(|s| s["codec_type"] == "video"),
            "expected a video stream in {probe}"
        );
        assert!(
            streams.iter().any(|s| s["codec_type"] == "audio"),
            "expected an audio stream in {probe}"
        );
    }

    #[test]
    fn drain_frames_and_drain_audio_write_newly_buffered_content() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();

        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        let mut event = start_event(
            dir.path(),
            initial_path,
            started_at,
            clip_timeline_start,
            2,
            2,
            8000,
        )
        .unwrap();

        seed_one_frame(&mut event, clip_timeline_start);

        // At 5fps, seed() sets next_frame_due to clip_timeline_start + 200ms;
        // sleeping past that before pushing ensures the newly-buffered frame
        // is actually due and gets written (not silently skipped for arriving
        // before its tick), exercising drain_frames' write branch for real.
        std::thread::sleep(Duration::from_millis(250));

        let ring_buffer =
            std::sync::Mutex::new(crate::buffer::RingBuffer::new(Duration::from_secs(30)));
        {
            let mut buf = ring_buffer.lock().unwrap();
            buf.push_frame(image::RgbImage::new(2, 2));
            buf.push_audio(vec![0.5; 100]);
        }

        // Both must succeed without error against real newly-buffered content
        // pushed after seed(); this is the steady-poll path the recording
        // writer thread drives in production.
        event.drain_frames(&ring_buffer).unwrap();
        event.drain_audio(&ring_buffer).unwrap();

        assert!(!event.camera_stalled());

        event.finish().unwrap();
    }

    #[test]
    fn start_errors_when_audio_temp_file_cannot_be_created() {
        let dir = tempfile::tempdir().unwrap();
        let started_at = chrono::Local::now();

        // A final_clip_path under a nonexistent subdirectory makes the audio
        // temp file's std::fs::File::create fail (no such directory), while
        // spawn_video_encoder itself doesn't touch the filesystem until
        // ffmpeg actually writes, so this exercises the audio-file error arm
        // specifically rather than the video spawn.
        let missing_subdir = dir.path().join("does-not-exist");
        let final_clip_path = missing_subdir.join("clip.mp4");

        let err = start_event(
            dir.path(),
            final_clip_path,
            started_at,
            Instant::now(),
            2,
            2,
            8000,
        )
        .err()
        .unwrap();

        assert!(err.to_string().contains("failed to create temp audio file"));
    }

    #[test]
    fn finish_errors_when_video_encoder_exits_nonzero() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        // width/height of 0 makes ffmpeg reject "-video_size 0x0" and exit
        // nonzero immediately, without needing any frames written to stdin.
        let event = start_event(
            dir.path(),
            initial_path,
            started_at,
            clip_timeline_start,
            0,
            0,
            8000,
        )
        .unwrap();

        let err = event.finish().unwrap_err();

        assert!(err.to_string().contains("ffmpeg video encoder exited with"));
    }

    #[test]
    fn finish_errors_when_audio_mux_fails() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        // A sample rate of 0 makes ffmpeg's mux step (-ar 0) reject the raw
        // PCM input and exit nonzero. finish() closes stdin and waits on the
        // video encoder regardless of whether any frame was ever written, so
        // no seed() call is needed to reach the mux step this targets.
        let event = start_event(
            dir.path(),
            initial_path,
            started_at,
            clip_timeline_start,
            2,
            2,
            0,
        )
        .unwrap();

        let err = event.finish().unwrap_err();

        assert!(err.to_string().contains("ffmpeg audio mux exited with"));
    }

    #[test]
    fn finish_errors_when_rename_target_already_exists_as_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        // finish() recomputes the final path from the accumulated class list
        // via clip_path() and renames into it. Pre-creating a directory at
        // that exact target path makes the rename fail (can't rename a file
        // onto an existing directory) without touching filesystem permissions.
        let renamed_path = clip_path(dir.path(), started_at, &["person"]).unwrap();
        std::fs::create_dir_all(&renamed_path).unwrap();

        let mut event = start_event(
            dir.path(),
            initial_path,
            started_at,
            clip_timeline_start,
            2,
            2,
            8000,
        )
        .unwrap();

        // Unlike finish_errors_when_audio_mux_fails' -ar 0 (which fails
        // fast), this test uses a valid sample rate so finish() actually
        // runs the full mux before reaching the rename step under test.
        // Muxing an empty audio stream against a video stream whose
        // duration ffmpeg can't determine makes the apad filter pad
        // indefinitely instead of erroring, hanging rather than completing,
        // so a real frame/audio chunk must be seeded first.
        seed_one_frame(&mut event, clip_timeline_start);

        event.record_detection("person", 0.9, clip_timeline_start);

        let err = event.finish().unwrap_err();

        assert!(
            err.to_string().contains("failed to rename clip"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn write_frame_errors_when_stdin_already_closed() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        let mut event = start_event(
            dir.path(),
            initial_path,
            started_at,
            clip_timeline_start,
            2,
            2,
            8000,
        )
        .unwrap();

        // Taking stdin out from under the event (rather than anything ffmpeg
        // does) is what write_frame's own "already closed" arm guards
        // against; drop it here to force that arm directly.
        drop(event.ffmpeg_video.stdin.take());

        let err = event.write_frame(&image::RgbImage::new(2, 2)).unwrap_err();
        assert!(err.to_string().contains("ffmpeg stdin was already closed"));

        // ffmpeg_video is still a live child with no stdin ever supplied a
        // frame; wait it out directly instead of calling finish() (which
        // would try to write to the now-closed stdin again via its own
        // drop(stdin.take())/wait() sequence, which is fine, but the process
        // never received "-video_size" data and this test only cares about
        // the write_frame error path above).
        let _ = event.ffmpeg_video.wait();
    }

    #[test]
    fn seed_errors_when_stdin_already_closed() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        let mut event = start_event(
            dir.path(),
            initial_path,
            started_at,
            clip_timeline_start,
            2,
            2,
            8000,
        )
        .unwrap();

        drop(event.ffmpeg_video.stdin.take());

        let err = event
            .seed(&[even_sized_frame_at(clip_timeline_start)], &[])
            .unwrap_err();
        assert!(err.to_string().contains("ffmpeg stdin was already closed"));

        let _ = event.ffmpeg_video.wait();
    }

    #[test]
    fn drain_frames_errors_when_stdin_already_closed() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        let mut event = start_event(
            dir.path(),
            initial_path,
            started_at,
            clip_timeline_start,
            2,
            2,
            8000,
        )
        .unwrap();

        seed_one_frame(&mut event, clip_timeline_start);
        std::thread::sleep(Duration::from_millis(250));

        let ring_buffer =
            std::sync::Mutex::new(crate::buffer::RingBuffer::new(Duration::from_secs(30)));
        {
            let mut buf = ring_buffer.lock().unwrap();
            buf.push_frame(image::RgbImage::new(2, 2));
        }

        drop(event.ffmpeg_video.stdin.take());

        let err = event.drain_frames(&ring_buffer).unwrap_err();
        assert!(err.to_string().contains("ffmpeg stdin was already closed"));

        let _ = event.ffmpeg_video.wait();
    }

    #[test]
    fn write_frame_errors_when_ffmpeg_process_has_exited() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        let mut event = start_event(
            dir.path(),
            initial_path,
            started_at,
            clip_timeline_start,
            2,
            2,
            8000,
        )
        .unwrap();

        // Kill the encoder out from under the event so its stdin pipe becomes
        // a broken pipe (the process is gone, but the Stdio::piped() handle
        // itself is still Some), reaching write_all's failure arm rather than
        // write_frame's own "stdin already closed" precondition check.
        // Deliberately never call Child::wait()/try_wait() here: std's wait()
        // takes self.stdin itself (to avoid a deadlock where the child blocks
        // writing to a full stdout/stderr pipe while the parent blocks in
        // wait() before draining it), which would trip the "already closed"
        // arm instead of the one this test targets.
        event.ffmpeg_video.kill().unwrap();

        let mut result = event.write_frame(&image::RgbImage::new(2, 2));
        // A single write can land in the pipe buffer before the kernel
        // notices the reader is gone; retry briefly until the broken pipe is
        // actually observed rather than asserting on a racy first attempt.
        for _ in 0..200 {
            if result.is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
            result = event.write_frame(&image::RgbImage::new(2, 2));
        }

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("failed to write frame to ffmpeg"),
            "unexpected error: {err}"
        );

        // Reap the killed child directly (bypassing RecordingEvent::finish,
        // which would try to write the sidecar/mux against a process this
        // test intentionally never fed valid frames to).
        let _ = event.ffmpeg_video.wait();
    }

    #[test]
    fn write_audio_errors_when_audio_file_is_not_writable() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        let mut event = start_event(
            dir.path(),
            initial_path,
            started_at,
            clip_timeline_start,
            2,
            2,
            8000,
        )
        .unwrap();

        // Swap in a read-only handle to the same temp file so write_all
        // itself fails (EBADF) rather than anything about the path/directory
        // being wrong; audio_tmp_path is private to this module, reachable
        // here as a same-module test.
        event.audio_file = std::fs::File::open(&event.audio_tmp_path).unwrap();

        let err = event.write_audio(&[0.0; 10]).unwrap_err();
        assert!(
            err.to_string()
                .contains("failed to write audio samples to temp file"),
            "unexpected error: {err}"
        );

        drop(event.ffmpeg_video.stdin.take());
        let _ = event.ffmpeg_video.wait();
    }

    #[test]
    fn seed_errors_when_audio_file_is_not_writable() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        let mut event = start_event(
            dir.path(),
            initial_path,
            started_at,
            clip_timeline_start,
            2,
            2,
            8000,
        )
        .unwrap();

        event.audio_file = std::fs::File::open(&event.audio_tmp_path).unwrap();

        let err = event
            .seed(
                &[],
                &[TimestampedAudio {
                    timestamp: clip_timeline_start,
                    samples: vec![0.0; 10],
                }],
            )
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("failed to write audio samples to temp file"),
            "unexpected error: {err}"
        );

        drop(event.ffmpeg_video.stdin.take());
        let _ = event.ffmpeg_video.wait();
    }

    #[test]
    fn drain_audio_errors_when_audio_file_is_not_writable() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        let mut event = start_event(
            dir.path(),
            initial_path,
            started_at,
            clip_timeline_start,
            2,
            2,
            8000,
        )
        .unwrap();

        seed_one_frame(&mut event, clip_timeline_start);

        let ring_buffer =
            std::sync::Mutex::new(crate::buffer::RingBuffer::new(Duration::from_secs(30)));
        {
            let mut buf = ring_buffer.lock().unwrap();
            buf.push_audio(vec![0.5; 100]);
        }

        event.audio_file = std::fs::File::open(&event.audio_tmp_path).unwrap();

        let err = event.drain_audio(&ring_buffer).unwrap_err();
        assert!(
            err.to_string()
                .contains("failed to write audio samples to temp file"),
            "unexpected error: {err}"
        );

        drop(event.ffmpeg_video.stdin.take());
        let _ = event.ffmpeg_video.wait();
    }

    #[test]
    fn finish_skips_stderr_capture_when_already_taken() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        let mut event = start_event(
            dir.path(),
            initial_path,
            started_at,
            clip_timeline_start,
            2,
            2,
            8000,
        )
        .unwrap();

        seed_one_frame(&mut event, clip_timeline_start);
        event.record_detection("person", 0.9, clip_timeline_start);

        // finish()'s `if let Some(stderr_pipe) = self.ffmpeg_video.stderr.take()`
        // only has stderr to capture the first time it runs; pre-taking it
        // here exercises the None arm (nothing to read, stderr stays empty)
        // instead of the Some arm every other finish()-calling test already
        // covers.
        drop(event.ffmpeg_video.stderr.take());

        event.finish().unwrap();
    }

    #[test]
    fn finish_errors_when_sidecar_path_already_exists_as_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        // No class is ever recorded, so finish() recomputes the same path
        // clip_path(..., &[]) already resolves to (no rename needed) and
        // proceeds straight to the sidecar write this test targets.
        let sidecar_target = sidecar_path(&initial_path);
        std::fs::create_dir_all(&sidecar_target).unwrap();

        let mut event = start_event(
            dir.path(),
            initial_path,
            started_at,
            clip_timeline_start,
            2,
            2,
            8000,
        )
        .unwrap();

        seed_one_frame(&mut event, clip_timeline_start);

        let err = event.finish().unwrap_err();
        assert!(
            err.to_string().contains("failed to write sidecar file"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn finish_errors_when_output_dir_cannot_be_created_for_rename() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        let mut event = start_event(
            dir.path(),
            initial_path,
            started_at,
            clip_timeline_start,
            2,
            2,
            8000,
        )
        .unwrap();

        seed_one_frame(&mut event, clip_timeline_start);
        event.record_detection("person", 0.9, clip_timeline_start);

        // finish() recomputes the final path via clip_path(&self.output_dir,
        // ...), which create_dir_all()s the day directory under output_dir.
        // Replacing output_dir with a path that is itself a plain file (not
        // a directory) makes that create_dir_all fail, reaching clip_path's
        // own '?' inside finish() rather than the later rename/write steps.
        let blocked_output_dir = dir.path().join("blocked");
        std::fs::write(&blocked_output_dir, b"not a directory").unwrap();
        event.output_dir = blocked_output_dir;

        let err = event.finish().unwrap_err();
        assert!(
            err.to_string()
                .contains("failed to create output directory")
        );
    }
}
