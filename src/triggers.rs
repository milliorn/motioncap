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
