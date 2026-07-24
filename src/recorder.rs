use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Local};
use serde::Serialize;

use crate::buffer::{TimestampedAudio, TimestampedFrame};
use crate::paths::sidecar_path;

const AUDIO_SAMPLE_RATE: u32 = 48_000;
const AUDIO_CHANNELS: u32 = 1;

#[derive(Serialize)]
pub struct DetectionRecord {
    pub offset_secs: f64,
    pub class_name: String,
    pub confidence: f32,
}

#[derive(Serialize)]
pub struct Sidecar {
    pub detections: Vec<DetectionRecord>,
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
    ffmpeg_video: Child,
    audio_tmp_path: std::path::PathBuf,
    audio_file: std::fs::File,
    final_clip_path: std::path::PathBuf,
    video_tmp_path: std::path::PathBuf,
    started_at: DateTime<Local>,
    clip_started: Instant,
    last_trigger_at: Instant,
    detections: Vec<DetectionRecord>,
}

impl RecordingEvent {
    /// Starts a new event: launches the video-encoding ffmpeg process and
    /// immediately writes the pre-buffered frames/audio captured before the
    /// trigger fired. `final_clip_path` should already reflect the trigger's
    /// classes/timestamp per the output layout convention (ADR 4); this
    /// module only performs the actual encoding/muxing, not path naming.
    pub fn start(
        final_clip_path: std::path::PathBuf,
        pre_frames: Vec<TimestampedFrame>,
        pre_audio: Vec<TimestampedAudio>,
        frame_rate: u32,
    ) -> Result<Self> {
        let started_at = Local::now();
        let (width, height) = pre_frames
            .first()
            .map(|f| f.image.dimensions())
            .context("cannot start a recording with no buffered frames")?;

        let video_tmp_path = final_clip_path.with_extension("video.tmp.mp4");
        let audio_tmp_path = final_clip_path.with_extension("audio.tmp.pcm");

        let ffmpeg_video = spawn_video_encoder(&video_tmp_path, width, height, frame_rate)?;
        let audio_file = std::fs::File::create(&audio_tmp_path)
            .with_context(|| format!("failed to create temp audio file {}", audio_tmp_path.display()))?;

        let mut event = Self {
            ffmpeg_video,
            audio_tmp_path,
            audio_file,
            final_clip_path,
            video_tmp_path,
            started_at,
            clip_started: Instant::now(),
            last_trigger_at: Instant::now(),
            detections: Vec::new(),
        };

        for frame in &pre_frames {
            event.write_frame(&frame.image)?;
        }

        for chunk in &pre_audio {
            event.write_audio(&chunk.samples)?;
        }

        Ok(event)
    }

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

    pub fn write_audio(&mut self, samples: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        self.audio_file
            .write_all(&bytes)
            .context("failed to write audio samples to temp file")?;
        Ok(())
    }

    pub fn record_detection(&mut self, class_name: &str, confidence: f32) {
        let offset_secs = self.clip_started.elapsed().as_secs_f64();

        self.detections.push(DetectionRecord {
            offset_secs,
            class_name: class_name.to_string(),
            confidence,
        });

        self.last_trigger_at = Instant::now();
    }

    pub fn touch(&mut self) {
        self.last_trigger_at = Instant::now();
    }

    pub fn quiet_for(&self) -> Duration {
        self.last_trigger_at.elapsed()
    }

    pub fn started_at(&self) -> DateTime<Local> {
        self.started_at
    }

    /// Stops encoding, muxes the buffered audio into the video, and writes
    /// the JSON sidecar alongside the final clip.
    pub fn finish(mut self) -> Result<()> {
        drop(self.ffmpeg_video.stdin.take());

        let status = self.ffmpeg_video.wait().context("ffmpeg video encoder failed")?;
        
        if !status.success() {
            bail!("ffmpeg video encoder exited with {status}");
        }

        drop(self.audio_file);

        mux_audio_into_video(&self.video_tmp_path, &self.audio_tmp_path, &self.final_clip_path)?;
        
        let _ = std::fs::remove_file(&self.video_tmp_path);
        let _ = std::fs::remove_file(&self.audio_tmp_path);

        let sidecar = Sidecar {
            detections: self.detections,
        };

        let sidecar_json = serde_json::to_string_pretty(&sidecar).context("failed to serialize sidecar")?;
        
        std::fs::write(sidecar_path(&self.final_clip_path), sidecar_json)
            .context("failed to write sidecar file")?;

        Ok(())
    }
}

fn spawn_video_encoder(
    path: &std::path::Path,
    width: u32,
    height: u32,
    frame_rate: u32,
) -> Result<Child> {
    Command::new("ffmpeg")
        .args(["-y", "-f", "rawvideo", "-pixel_format", "rgb24"])
        .args(["-video_size", &format!("{width}x{height}")])
        .args(["-framerate", &frame_rate.to_string()])
        .args(["-i", "-"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to spawn ffmpeg video encoder")
}

fn mux_audio_into_video(video_path: &std::path::Path, audio_path: &std::path::Path, output_path: &std::path::Path) -> Result<()> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-i"])
        .arg(video_path)
        .args(["-f", "f32le", "-ar", &AUDIO_SAMPLE_RATE.to_string(), "-ac", &AUDIO_CHANNELS.to_string()])
        .arg("-i")
        .arg(audio_path)
        .args(["-c:v", "copy", "-c:a", "aac", "-shortest"])
        .arg(output_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("failed to spawn ffmpeg for audio muxing")?;

    if !status.success() {
        bail!("ffmpeg audio mux exited with {status}");
    }
    
    Ok(())
}
