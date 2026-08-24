use std::fmt::Debug;
use std::path::Path;

use anyhow::{Context, Result};
use image::{RgbImage, imageops::FilterType};
use ndarray::{Array4, s};
use ort::ep::{CPU, CUDA, OpenVINO, ROCm};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{Shape, Tensor};

/// `ort::Error` doesn't implement `std::error::Error` in a way `anyhow::Context`
/// can use directly, so ort results are converted through this helper instead.
pub fn ort_err<T, E: Debug>(result: std::result::Result<T, E>, msg: &str) -> Result<T> {
    result.map_err(|e| anyhow::anyhow!("{msg}: {e:?}"))
}

/// Serializes every `#[ignore]`d test across the crate that constructs a real
/// `Detector`/ONNX Runtime session and/or a real `MotionGate` (both here and
/// in `main.rs`'s test module). `cargo test`'s default parallelism ran
/// multiple such tests concurrently and reliably produced a heap-corruption
/// abort ("corrupted double-linked list") within a handful of runs. Neither
/// `ort`'s `Session` nor `OpenCV`'s `BackgroundSubtractorMOG2` are documented
/// as safe to construct/run concurrently across independent instances in
/// separate threads, and this crate's production code never does so (YOLO
/// inference and the motion gate both run single-threaded inside the
/// detection worker). Every `#[ignore]`d test that touches either must
/// acquire this lock for its full duration before doing anything else.
#[cfg(test)]
pub static MODEL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

/// `YOLOv8`'s fixed square input resolution.
const MODEL_INPUT_SIZE: u32 = 640;

/// Divides an unpadded dimension in half to center it within
/// `MODEL_INPUT_SIZE`'s letterbox padding.
const LETTERBOX_CENTER_DIVISOR: u32 = 2;

/// Maximum value of an 8-bit color channel, used to normalize pixel values
/// into the `[0, 1]` range the model expects.
const PIXEL_MAX_VALUE: f32 = 255.0;

/// Standard YOLO letterbox pad color (mid-grey, `114` out of `255`), matching
/// the padding value `YOLOv8` itself is trained with.
const LETTERBOX_PAD_VALUE: f32 = 114.0;

/// `YOLOv8`'s standard per-anchor output channel count: 4 box coordinates
/// plus one score per COCO class (80 classes).
const YOLO_OUTPUT_CHANNELS: usize = 84;

/// Number of leading channels per anchor that encode box coordinates (before
/// the per-class scores begin), in `YOLOv8`'s standard output layout.
const YOLO_BOX_COORD_CHANNELS: usize = 4;

/// A single YOLO detection above the confidence threshold.
pub struct Detection {
    /// The detected COCO class name (person or an animal, see `LIVING_THING_CLASSES`).
    pub class_name: &'static str,
    /// The model's reported confidence for this detection.
    pub confidence: f32,
}

/// Letterboxes the frame into a square canvas (preserving aspect ratio, padding
/// with grey) and converts to an NCHW f32 tensor normalized to [0, 1]. This is the
/// standard `YOLOv8` preprocessing; a naive stretch-to-square resize distorts
/// non-square camera frames (e.g. this webcam's 640x480) enough to
/// meaningfully degrade detection confidence, since the model is trained on
/// letterboxed inputs, not stretched ones.
#[allow(
    clippy::cast_precision_loss,
    reason = "camera frame dimensions never approach f32's 24-bit exact-integer range"
)]
pub fn preprocess(frame: &RgbImage) -> Array4<f32> {
    let (src_w, src_h) = frame.dimensions();
    let scale =
        (MODEL_INPUT_SIZE as f32 / src_w as f32).min(MODEL_INPUT_SIZE as f32 / src_h as f32);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "scale is derived from MODEL_INPUT_SIZE / src_dim, so result stays within [0, MODEL_INPUT_SIZE]"
    )]
    let new_w = (src_w as f32 * scale).round() as u32;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "scale is derived from MODEL_INPUT_SIZE / src_dim, so result stays within [0, MODEL_INPUT_SIZE]"
    )]
    let new_h = (src_h as f32 * scale).round() as u32;

    let resized = image::imageops::resize(frame, new_w, new_h, FilterType::Triangle);

    // new_w/new_h <= MODEL_INPUT_SIZE always holds: scale = min(SIZE/src_w,
    // SIZE/src_h) <= 1.0, so new_w = round(src_w * scale) <= SIZE (likewise
    // new_h). The subtraction and the pad_x/pad_y additions below therefore
    // never overflow, and the resulting tensor index is always < SIZE,
    // matching tensor's allocated shape.
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "new_w/new_h <= MODEL_INPUT_SIZE, proven above"
    )]
    let pad_x = (MODEL_INPUT_SIZE - new_w) / LETTERBOX_CENTER_DIVISOR;
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "new_w/new_h <= MODEL_INPUT_SIZE, proven above"
    )]
    let pad_y = (MODEL_INPUT_SIZE - new_h) / LETTERBOX_CENTER_DIVISOR;

    let mut tensor = Array4::<f32>::from_elem(
        (1, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize),
        LETTERBOX_PAD_VALUE / PIXEL_MAX_VALUE,
    );
    for (x, y, pixel) in resized.enumerate_pixels() {
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "x < new_w and pad_x + new_w <= MODEL_INPUT_SIZE, so x + pad_x never overflows"
        )]
        let dst_x = (x + pad_x) as usize;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "y < new_h and pad_y + new_h <= MODEL_INPUT_SIZE, so y + pad_y never overflows"
        )]
        let dst_y = (y + pad_y) as usize;
        #[allow(
            clippy::indexing_slicing,
            reason = "dst_x, dst_y < MODEL_INPUT_SIZE, matching tensor's allocated shape"
        )]
        {
            tensor[[0, 0, dst_y, dst_x]] = f32::from(pixel[0]) / PIXEL_MAX_VALUE;
            tensor[[0, 1, dst_y, dst_x]] = f32::from(pixel[1]) / PIXEL_MAX_VALUE;
            tensor[[0, 2, dst_y, dst_x]] = f32::from(pixel[2]) / PIXEL_MAX_VALUE;
        }
    }
    tensor
}

/// Parses `YOLOv8`'s standard `[1, 84, 8400]` output (4 box coords + 80 class
/// scores, per anchor) and filters to living-thing classes above threshold.
/// No NMS/box deduplication is performed since only class presence (not
/// precise box geometry) is needed to decide whether to trigger a recording.
pub fn postprocess(shape: &Shape, data: &[f32], confidence_threshold: f32) -> Vec<Detection> {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "ONNX tensor dims are always small positive values (e.g. 1, 84, 8400)"
    )]
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

    let [_, class_dim, num_anchors] = dims[..] else {
        log::warn!("unexpected YOLO output shape: {dims:?}");
        return Vec::new();
    };
    if class_dim != YOLO_OUTPUT_CHANNELS {
        log::warn!("unexpected YOLO output shape: {dims:?}");
        return Vec::new();
    }

    let view = ndarray::ArrayView2::from_shape((YOLO_OUTPUT_CHANNELS, num_anchors), data)
        .expect("output tensor size should match its own reported shape");

    let mut best_per_class: std::collections::HashMap<usize, f32> =
        std::collections::HashMap::new();

    for anchor in 0..num_anchors {
        let scores = view.slice(s![YOLO_BOX_COORD_CHANNELS..YOLO_OUTPUT_CHANNELS, anchor]);
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

/// A loaded YOLO ONNX Runtime session ready to run inference.
pub struct Detector {
    /// The underlying ONNX Runtime inference session.
    session: Session,
}

impl Detector {
    /// Loads the YOLO ONNX model, registering execution providers in priority
    /// order CUDA -> `ROCm` -> `OpenVINO` -> CPU (ADR 3). `ort` probes availability
    /// per-provider and falls through automatically, so this works unmodified
    /// on hardware with any subset of these accelerators, or none at all.
    ///
    /// Contains at least one `Err` arm (a malformed/missing model file) that
    /// no safe test can trigger without a real `models/yolov8n.onnx` and a
    /// working ONNX Runtime build; the two tests that do exercise this
    /// function (below) stay `#[ignore]`'d, requiring `MODEL_TEST_LOCK`.
    pub fn load(model_path: &Path, force_cpu: bool) -> Result<Self> {
        let builder = ort_err(
            Session::builder(),
            "failed to create ONNX Runtime session builder",
        )?;

        let mut builder = ort_err(
            builder.with_optimization_level(GraphOptimizationLevel::Level3),
            "failed to set optimization level",
        )?;

        builder = if force_cpu {
            // Register only CPU, rather than skipping registration
            // entirely: with no execution providers explicitly registered,
            // ONNX Runtime falls back to its own default provider selection,
            // which (depending on how the linked ONNX Runtime build was
            // configured) isn't guaranteed to be CPU-only, silently
            // defeating what `--force-cpu` promises.
            ort_err(
                builder.with_execution_providers([CPU::default().build()]),
                "failed to register CPU execution provider",
            )?
        } else {
            ort_err(
                builder.with_execution_providers([
                    CUDA::default().build(),
                    ROCm::default().build(),
                    OpenVINO::default().build(),
                    CPU::default().build(),
                ]),
                "failed to register execution providers",
            )?
        };

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
    ///
    /// Contains at least one `Err` arm (an inference failure, an unrecognized
    /// output tensor) that no safe test can trigger without a real model
    /// file and ONNX Runtime session; see `load`'s doc comment.
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

#[cfg(test)]
mod tests {
    //! Unit tests for `preprocess`/`postprocess`/`ort_err`, exercised without a
    //! real ONNX Runtime session or model file.
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        clippy::arithmetic_side_effects,
        clippy::cast_possible_wrap,
        reason = "test assertions favor unwrap/indexing/plain arithmetic for clarity over the \
                   production-code proof-obligation style used elsewhere in this crate; test \
                   data sizes are small hardcoded constants, so overflow/wrap is not reachable"
    )]

    use image::Rgb;
    use ort::value::Shape;

    use super::*;

    /// A representative detection confidence threshold, used wherever a test
    /// needs one but its exact value isn't the thing under test.
    const TEST_CONFIDENCE_THRESHOLD: f32 = 0.3;

    /// A score used to demonstrate "clearly above `TEST_CONFIDENCE_THRESHOLD`".
    const ABOVE_THRESHOLD_SCORE: f32 = 0.9;

    /// A score used to demonstrate "clearly below `TEST_CONFIDENCE_THRESHOLD`".
    const BELOW_THRESHOLD_SCORE: f32 = 0.1;

    /// The lower of two same-class scores across anchors, used to prove
    /// `postprocess` keeps the best (not first-seen) score per class.
    const LOWER_REPEAT_SCORE: f32 = 0.4;

    /// The higher of two same-class scores across anchors. See `LOWER_REPEAT_SCORE`.
    const HIGHER_REPEAT_SCORE: f32 = 0.95;

    /// A channel count that deliberately does not match `YOLO_OUTPUT_CHANNELS`,
    /// to exercise `postprocess`'s wrong-shape guard.
    const WRONG_CLASS_DIM: usize = 85;

    /// An anchor count used only by the wrong-shape tests, where its exact
    /// value is irrelevant to what's being proven.
    const WRONG_SHAPE_TEST_ANCHORS: usize = 10;

    /// Builds a synthetic `[1, YOLO_OUTPUT_CHANNELS, num_anchors]` YOLO output
    /// buffer where every anchor's class scores are zero except the ones set
    /// via `scores`, a list of `(anchor, class_idx, score)` triples
    /// (`class_idx` is 0-based over the 80 COCO classes, matching
    /// `LIVING_THING_CLASSES`' indices).
    fn synthetic_output(num_anchors: usize, scores: &[(usize, usize, f32)]) -> Vec<f32> {
        let mut data = vec![0.0_f32; YOLO_OUTPUT_CHANNELS * num_anchors];
        for &(anchor, class_idx, score) in scores {
            let row = YOLO_BOX_COORD_CHANNELS + class_idx;
            data[row * num_anchors + anchor] = score;
        }
        data
    }

    #[test]
    fn postprocess_filters_to_living_thing_classes_above_threshold() {
        let num_anchors = 2;
        // class_idx 0 = person (living-thing, above threshold);
        // class_idx 1 = bicycle (not in LIVING_THING_CLASSES, ignored).
        let data = synthetic_output(
            num_anchors,
            &[(0, 0, ABOVE_THRESHOLD_SCORE), (1, 1, ABOVE_THRESHOLD_SCORE)],
        );
        let shape = Shape::from(vec![1_i64, YOLO_OUTPUT_CHANNELS as i64, num_anchors as i64]);

        let detections = postprocess(&shape, &data, TEST_CONFIDENCE_THRESHOLD);

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].class_name, "person");
        assert!((detections[0].confidence - ABOVE_THRESHOLD_SCORE).abs() < f32::EPSILON);
    }

    #[test]
    fn postprocess_excludes_scores_below_threshold() {
        let num_anchors = 1;
        let data = synthetic_output(num_anchors, &[(0, 0, BELOW_THRESHOLD_SCORE)]);
        let shape = Shape::from(vec![1_i64, YOLO_OUTPUT_CHANNELS as i64, num_anchors as i64]);

        let detections = postprocess(&shape, &data, TEST_CONFIDENCE_THRESHOLD);

        assert!(detections.is_empty());
    }

    #[test]
    fn postprocess_keeps_best_score_per_class_across_anchors() {
        let num_anchors = 2;
        let data = synthetic_output(
            num_anchors,
            &[(0, 0, LOWER_REPEAT_SCORE), (1, 0, HIGHER_REPEAT_SCORE)],
        );
        let shape = Shape::from(vec![1_i64, YOLO_OUTPUT_CHANNELS as i64, num_anchors as i64]);

        let detections = postprocess(&shape, &data, TEST_CONFIDENCE_THRESHOLD);

        assert_eq!(detections.len(), 1);
        assert!((detections[0].confidence - HIGHER_REPEAT_SCORE).abs() < f32::EPSILON);
    }

    #[test]
    fn postprocess_keeps_first_best_score_when_later_anchor_scores_lower() {
        let num_anchors = 2;
        let data = synthetic_output(
            num_anchors,
            &[(0, 0, HIGHER_REPEAT_SCORE), (1, 0, LOWER_REPEAT_SCORE)],
        );
        let shape = Shape::from(vec![1_i64, YOLO_OUTPUT_CHANNELS as i64, num_anchors as i64]);

        let detections = postprocess(&shape, &data, TEST_CONFIDENCE_THRESHOLD);

        assert_eq!(detections.len(), 1);
        assert!((detections[0].confidence - HIGHER_REPEAT_SCORE).abs() < f32::EPSILON);
    }

    #[test]
    fn postprocess_returns_empty_on_wrong_class_dim() {
        let shape = Shape::from(vec![
            1_i64,
            WRONG_CLASS_DIM as i64,
            WRONG_SHAPE_TEST_ANCHORS as i64,
        ]);
        let data = vec![0.0_f32; WRONG_CLASS_DIM * WRONG_SHAPE_TEST_ANCHORS];

        assert!(postprocess(&shape, &data, TEST_CONFIDENCE_THRESHOLD).is_empty());
    }

    #[test]
    fn postprocess_returns_empty_on_wrong_rank() {
        let shape = Shape::from(vec![
            YOLO_OUTPUT_CHANNELS as i64,
            WRONG_SHAPE_TEST_ANCHORS as i64,
        ]);
        let data = vec![0.0_f32; YOLO_OUTPUT_CHANNELS * WRONG_SHAPE_TEST_ANCHORS];

        assert!(postprocess(&shape, &data, TEST_CONFIDENCE_THRESHOLD).is_empty());
    }

    /// This webcam's native landscape resolution (see `preprocess`'s doc
    /// comment), used to exercise letterbox padding on the vertical axis.
    const LANDSCAPE_WIDTH: u32 = 640;
    /// See `LANDSCAPE_WIDTH`.
    const LANDSCAPE_HEIGHT: u32 = 480;

    /// A mid-grey fixture pixel value, deliberately close to but distinct
    /// from `LETTERBOX_PAD_VALUE`'s `114`, so the pad-fill assertion in the
    /// landscape/portrait tests genuinely distinguishes "pad" from "resized
    /// pixel" rather than passing coincidentally.
    const GREY_FIXTURE_PIXEL: u8 = 200;

    /// A fixture pixel value chosen to visibly differ per channel, so the
    /// square-input test can assert "not the pad color" without ambiguity.
    const NON_GREY_FIXTURE_PIXEL: [u8; 3] = [10, 20, 30];

    /// Tolerance the square-input test uses to assert a pixel is clearly
    /// *not* the letterbox pad value (wider than the exact-match tolerance
    /// used elsewhere, since this compares two deliberately different values
    /// rather than checking equality).
    const NOT_PAD_VALUE_TOLERANCE: f32 = 0.05;

    /// An arbitrary `Ok` payload value for `ort_err`'s passthrough test; its
    /// exact value is irrelevant to what's being proven (that `Ok` values
    /// pass through unchanged).
    const ORT_ERR_OK_PAYLOAD: u32 = 42;

    /// Mid-grey fixture value for the real-model inference test; a plain
    /// neutral fill with no real subject, distinct from
    /// `GREY_FIXTURE_PIXEL`/`NON_GREY_FIXTURE_PIXEL` only because this test
    /// lives in a different logical group (real-model inference vs.
    /// synthetic-tensor unit tests).
    const NEUTRAL_FIXTURE_PIXEL: u8 = 128;

    #[test]
    fn preprocess_pads_vertically_for_landscape_input() {
        let frame = RgbImage::from_pixel(
            LANDSCAPE_WIDTH,
            LANDSCAPE_HEIGHT,
            Rgb([GREY_FIXTURE_PIXEL; 3]),
        );
        let tensor = preprocess(&frame);

        // Landscape (wider than tall): scale is bound by width, so the resized
        // image is shorter than MODEL_INPUT_SIZE and padding is added on the
        // vertical axis only. The top row should be the grey pad fill, not a
        // resized pixel.
        let pad_value = LETTERBOX_PAD_VALUE / PIXEL_MAX_VALUE;
        assert!((tensor[[0, 0, 0, 0]] - pad_value).abs() < f32::EPSILON);
    }

    #[test]
    fn preprocess_pads_horizontally_for_portrait_input() {
        let frame = RgbImage::from_pixel(
            LANDSCAPE_HEIGHT,
            LANDSCAPE_WIDTH,
            Rgb([GREY_FIXTURE_PIXEL; 3]),
        );
        let tensor = preprocess(&frame);

        let pad_value = LETTERBOX_PAD_VALUE / PIXEL_MAX_VALUE;
        assert!((tensor[[0, 0, 0, 0]] - pad_value).abs() < f32::EPSILON);
    }

    #[test]
    fn preprocess_adds_no_padding_for_square_input() {
        let frame = RgbImage::from_pixel(
            MODEL_INPUT_SIZE,
            MODEL_INPUT_SIZE,
            Rgb(NON_GREY_FIXTURE_PIXEL),
        );
        let tensor = preprocess(&frame);

        // A square input needs no letterboxing: scale = 1.0, new_w = new_h =
        // MODEL_INPUT_SIZE, so every pixel in the tensor comes from the
        // resized image rather than the grey pad fill.
        let pad_value = LETTERBOX_PAD_VALUE / PIXEL_MAX_VALUE;
        assert!((tensor[[0, 0, 0, 0]] - pad_value).abs() > NOT_PAD_VALUE_TOLERANCE);
    }

    #[test]
    fn ort_err_passes_through_ok() {
        let result: Result<u32, &str> = Ok(ORT_ERR_OK_PAYLOAD);
        assert_eq!(ort_err(result, "unused").unwrap(), ORT_ERR_OK_PAYLOAD);
    }

    #[test]
    fn ort_err_wraps_error_with_message() {
        let result: Result<u32, &str> = Err("boom");
        let err = ort_err(result, "context message").unwrap_err();
        let text = format!("{err}");
        assert!(text.contains("context message"));
        assert!(text.contains("boom"));
    }

    #[test]
    #[ignore = "requires models/yolov8n.onnx + a working ONNX Runtime build (see ORT_DYLIB_PATH in README); \
                run explicitly with `cargo test -- --ignored` on a machine that has both"]
    fn detector_loads_and_runs_inference_on_a_synthetic_frame() {
        let _guard = MODEL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let model_path = std::path::Path::new("models/yolov8n.onnx");
        let mut detector = Detector::load(model_path, true)
            .expect("failed to load model, is models/yolov8n.onnx present?");

        let frame = RgbImage::from_pixel(
            LANDSCAPE_WIDTH,
            LANDSCAPE_HEIGHT,
            Rgb([NEUTRAL_FIXTURE_PIXEL; 3]),
        );

        // A synthetic grey frame has no real subject in it, so this only
        // exercises Detector::load/detect's plumbing (session creation,
        // preprocessing, running inference, extracting the output tensor);
        // postprocess's actual filtering logic is already covered directly by
        // the postprocess_* tests above with controlled synthetic tensors.
        let detections = detector
            .detect(&frame, TEST_CONFIDENCE_THRESHOLD)
            .expect("inference should not error on a well-formed frame");

        // No assertion on detections.len(): a blank grey frame may or may not
        // produce spurious detections depending on the model build.
        drop(detections);
    }

    #[test]
    #[ignore = "requires models/yolov8n.onnx + a working ONNX Runtime build (see ORT_DYLIB_PATH in README); \
                run explicitly with `cargo test -- --ignored` on a machine that has both"]
    fn detector_load_registers_gpu_execution_providers_when_not_forced_to_cpu() {
        let _guard = MODEL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // force_cpu=false exercises the CUDA/ROCm/OpenVINO/CPU registration
        // branch (as opposed to the force_cpu=true test above, which only
        // exercises the CPU-only branch); ort probes each provider at
        // runtime and falls through to whichever is actually available
        // (ADR 3), so this should succeed identically on hardware with none
        // of those accelerators, same as the force_cpu=true case.
        let model_path = std::path::Path::new("models/yolov8n.onnx");
        Detector::load(model_path, false)
            .expect("failed to load model, is models/yolov8n.onnx present?");
    }
}
