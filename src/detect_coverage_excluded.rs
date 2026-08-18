//! Holds `Detector::load`/`Detector::detect`, the only parts of `detect.rs`
//! that require a real ONNX Runtime session and model file. Both contain at
//! least one `Err` arm (a malformed/missing model file, an inference failure,
//! an unrecognized output tensor) that no safe test can trigger without a real
//! `models/yolov8n.onnx` and a working ONNX Runtime build; the two tests that
//! do exercise them (`detector_loads_and_runs_inference_on_a_synthetic_frame`,
//! `detector_load_registers_gpu_execution_providers_when_not_forced_to_cpu`)
//! stay `#[ignore]`'d, requiring `MODEL_TEST_LOCK`. Split out of `detect.rs`
//! (same convention as `coverage_excluded.rs` at the crate root) so that file
//! reaches genuine 100% coverage on `preprocess`/`postprocess`/`ort_err`,
//! which stay there and are unit-tested with synthetic data. See
//! `docs/adr/0006-coverage-exclusions.md`.

use std::path::Path;

use anyhow::{Context, Result};
use image::RgbImage;
use ort::ep::{CPU, CUDA, OpenVINO, ROCm};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::detect::{Detection, ort_err, postprocess, preprocess};

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
    #![allow(
        clippy::unwrap_used,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap for clarity; matches convention used by other \
                   *_coverage_excluded.rs test modules"
    )]

    use super::Detector;
    use crate::detect::MODEL_TEST_LOCK;
    use image::RgbImage;

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

        let frame = RgbImage::from_pixel(640, 480, image::Rgb([128, 128, 128]));

        // A synthetic grey frame has no real subject in it, so this only
        // exercises Detector::load/detect's plumbing (session creation,
        // preprocessing, running inference, extracting the output tensor);
        // postprocess's actual filtering logic is already covered directly by
        // the postprocess_* tests in detect.rs with controlled synthetic tensors.
        let detections = detector
            .detect(&frame, 0.3)
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
