use crate::detect::Detection;

/// Converts a set of confirmed YOLO detections into a recording trigger.
/// Returns `None` if nothing was confirmed, meaning whatever tripped the
/// motion gate (e.g. a ceiling fan) was correctly not treated as an event
/// (ADR 2): recording only ever starts on a confirmed living-thing
/// classification, never on the motion gate alone.
pub fn evaluate(detections: Vec<Detection>) -> Option<Vec<Detection>> {
    if detections.is_empty() {
        None
    } else {
        Some(detections)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the motion-gate-to-recording-trigger conversion.
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap/indexing for clarity; panics here fail the test, which is the intended behavior"
    )]

    use super::*;

    /// Fixture confidence for a synthetic test detection; its exact value
    /// doesn't matter, only that `evaluate` passes it through unchanged.
    const FIXTURE_CONFIDENCE: f32 = 0.9;

    #[test]
    fn empty_detections_yield_no_trigger() {
        assert!(evaluate(Vec::new()).is_none());
    }

    #[test]
    fn nonempty_detections_are_passed_through_unchanged() {
        let detections = vec![Detection {
            class_name: "person",
            confidence: FIXTURE_CONFIDENCE,
        }];

        let result = evaluate(detections);

        let result = result.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].class_name, "person");
    }
}
