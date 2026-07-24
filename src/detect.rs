use std::fmt::Debug;
use std::path::Path;

use anyhow::{Context, Result};
use image::{RgbImage, imageops::FilterType};
use ndarray::{Array4, s};
use ort::ep::{
    CPUExecutionProvider, CUDAExecutionProvider, OpenVINOExecutionProvider, ROCmExecutionProvider,
};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{Shape, Tensor};

/// `ort::Error` doesn't implement `std::error::Error` in a way `anyhow::Context`
/// can use directly, so ort results are converted through this helper instead.
fn ort_err<T, E: Debug>(result: std::result::Result<T, E>, msg: &str) -> Result<T> {
    result.map_err(|e| anyhow::anyhow!("{msg}: {e:?}"))
}

/// COCO class indices for "person" plus every animal class, per the
/// "every living thing" detection scope decided during planning (ADR 2).
const LIVING_THING_CLASSES: &[(usize, &str)] = &[
    (0, "person"),
    (14, "bird"),
    (15, "cat"),
    (16, "dog"),
    (17, "horse"),
    (18, "sheep"),
    (19, "cow"),
    (20, "elephant"),
    (21, "bear"),
    (22, "zebra"),
    (23, "giraffe"),
];

const MODEL_INPUT_SIZE: u32 = 640;

pub struct Detection {
    pub class_name: &'static str,
    pub confidence: f32,
}

pub struct Detector {
    session: Session,
}

impl Detector {
    /// Loads the YOLO ONNX model, registering execution providers in priority
    /// order CUDA -> ROCm -> OpenVINO -> CPU (ADR 3). `ort` probes availability
    /// per-provider and falls through automatically, so this works unmodified
    /// on hardware with any subset of these accelerators, or none at all.
    pub fn load(model_path: &Path, force_cpu: bool) -> Result<Self> {
        let builder = ort_err(
            Session::builder(),
            "failed to create ONNX Runtime session builder",
        )?;
        let mut builder = ort_err(
            builder.with_optimization_level(GraphOptimizationLevel::Level3),
            "failed to set optimization level",
        )?;

        if !force_cpu {
            builder = ort_err(
                builder.with_execution_providers([
                    CUDAExecutionProvider::default().build(),
                    ROCmExecutionProvider::default().build(),
                    OpenVINOExecutionProvider::default().build(),
                    CPUExecutionProvider::default().build(),
                ]),
                "failed to register execution providers",
            )?;
        }

        let session = ort_err(
            builder.commit_from_file(model_path),
            &format!("failed to load model from {}", model_path.display()),
        )?;

        log::info!(
            "ONNX Runtime session created (execution provider availability determined per-provider; \
             see startup logs for CUDA/ROCm/OpenVINO/CPU probing detail)"
        );

        Ok(Self { session })
    }

    /// Runs YOLO inference on a frame and returns every detected instance of
    /// a living-thing class (person or COCO animal, see `LIVING_THING_CLASSES`)
    /// above `confidence_threshold`.
    pub fn detect(
        &mut self,
        frame: &RgbImage,
        confidence_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let input = preprocess(frame);
        let input_value = ort_err(Tensor::from_array(input), "failed to build input tensor")?;

        let outputs = ort_err(
            self.session.run(ort::inputs!["images" => input_value]),
            "YOLO inference failed",
        )?;

        let output = outputs
            .get("output0")
            .or_else(|| outputs.get("0"))
            .context("model produced no recognizable output tensor")?;
        let (shape, data) = ort_err(
            output.try_extract_tensor::<f32>(),
            "failed to extract output tensor",
        )?;

        Ok(postprocess(shape, data, confidence_threshold))
    }
}

/// Letterboxes the frame into a square canvas (preserving aspect ratio, padding
/// with grey) and converts to an NCHW f32 tensor normalized to [0, 1] -- the
/// standard YOLOv8 preprocessing. A naive stretch-to-square resize distorts
/// non-square camera frames (e.g. this webcam's 640x480) enough to
/// meaningfully degrade detection confidence, since the model is trained on
/// letterboxed inputs, not stretched ones.
fn preprocess(frame: &RgbImage) -> Array4<f32> {
    let (src_w, src_h) = frame.dimensions();
    let scale =
        (MODEL_INPUT_SIZE as f32 / src_w as f32).min(MODEL_INPUT_SIZE as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round() as u32;
    let new_h = (src_h as f32 * scale).round() as u32;

    let resized = image::imageops::resize(frame, new_w, new_h, FilterType::Triangle);

    let pad_x = (MODEL_INPUT_SIZE - new_w) / 2;
    let pad_y = (MODEL_INPUT_SIZE - new_h) / 2;

    let mut tensor = Array4::<f32>::from_elem(
        (1, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize),
        114.0 / 255.0,
    );
    for (x, y, pixel) in resized.enumerate_pixels() {
        let (dst_x, dst_y) = ((x + pad_x) as usize, (y + pad_y) as usize);
        tensor[[0, 0, dst_y, dst_x]] = pixel[0] as f32 / 255.0;
        tensor[[0, 1, dst_y, dst_x]] = pixel[1] as f32 / 255.0;
        tensor[[0, 2, dst_y, dst_x]] = pixel[2] as f32 / 255.0;
    }
    tensor
}

/// Parses YOLOv8's standard `[1, 84, 8400]` output (4 box coords + 80 class
/// scores, per anchor) and filters to living-thing classes above threshold.
/// No NMS/box deduplication is performed since only class presence (not
/// precise box geometry) is needed to decide whether to trigger a recording.
fn postprocess(shape: &Shape, data: &[f32], confidence_threshold: f32) -> Vec<Detection> {
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    if dims.len() != 3 || dims[1] != 84 {
        log::warn!("unexpected YOLO output shape: {dims:?}");
        return Vec::new();
    }
    let num_anchors = dims[2];

    let view = ndarray::ArrayView2::from_shape((84, num_anchors), data)
        .expect("output tensor size should match its own reported shape");

    let mut best_per_class: std::collections::HashMap<usize, f32> =
        std::collections::HashMap::new();

    for anchor in 0..num_anchors {
        let scores = view.slice(s![4..84, anchor]);
        for (class_idx, &score) in scores.iter().enumerate() {
            if score < confidence_threshold {
                continue;
            }
            best_per_class
                .entry(class_idx)
                .and_modify(|best| {
                    if score > *best {
                        *best = score;
                    }
                })
                .or_insert(score);
        }
    }

    LIVING_THING_CLASSES
        .iter()
        .filter_map(|&(class_idx, name)| {
            best_per_class.get(&class_idx).map(|&confidence| Detection {
                class_name: name,
                confidence,
            })
        })
        .collect()
}
