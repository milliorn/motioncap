use anyhow::{Context, Result};

/// Log file name written under `--output-dir` (see `init_logging`).
pub const LOG_FILE_NAME: &str = "motioncap.log";

/// Writes every log line to both the given file and stderr, since this
/// process runs long-lived and unattended. Stderr alone
/// is lost the moment the terminal/session that launched it goes away, but
/// keeping stderr too means interactive/`--preview` runs still see live
/// diagnostics without needing to tail the file.
pub struct TeeWriter {
    /// The persistent log file under `--output-dir`.
    file: std::fs::File,
}

impl TeeWriter {
    /// Wraps an already-opened log file for tee'd write/flush to both it and stderr.
    pub const fn new(file: std::fs::File) -> Self {
        Self { file }
    }
}

/// Runs both fallible sink operations unconditionally, never short-circuiting
/// with `?` after just the first, and propagates the first error, if any.
/// One sink failing (e.g. a full disk, or stderr closed under a supervisor)
/// must never silently suppress the other from being attempted.
fn both(a: std::io::Result<()>, b: std::io::Result<()>) -> std::io::Result<()> {
    a?;
    b?;
    Ok(())
}

impl std::io::Write for TeeWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        both(self.file.write_all(buf), std::io::stderr().write_all(buf))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        both(self.file.flush(), std::io::stderr().flush())
    }
}

/// Initializes logging to write every line to both `<output_dir>/motioncap.log`
/// and stderr (see `TeeWriter`), honoring `RUST_LOG` exactly as a bare
/// `env_logger::init()` would otherwise. `output_dir` is created if it
/// doesn't exist yet, since this may run before anything else has created it.
/// Calls `env_logger::Builder::init()`, which sets the process-global logger
/// and panics if called more than once in a process; `cargo test` runs the
/// entire suite in one process, so this can only be invoked here, in the
/// real `main` path, without an order-dependent risk of poisoning every
/// other test's use of the `log::` macros.
pub fn init_logging(output_dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create output directory {}", output_dir.display()))?;

    let log_path = output_dir.join(LOG_FILE_NAME);

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("failed to open log file {}", log_path.display()))?;

    env_logger::Builder::from_env(env_logger::Env::default())
        .target(env_logger::Target::Pipe(Box::new(TeeWriter::new(file))))
        .init();

    Ok(())
}

#[cfg(test)]
mod tests {
    //! Unit tests for `TeeWriter`'s read/write logic and the `both` helper.
    //!
    //! `init_logging` itself is not unit-tested: it calls
    //! `env_logger::Builder::init()`, which sets the process-global logger
    //! and panics if called more than once, since `cargo test` runs the
    //! whole suite in one process, exercising it here would either poison
    //! every other test's logging or only be safely callable once, in an
    //! order-dependent way.
    #![allow(
        clippy::unwrap_used,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap for clarity; panics here fail the test, which is the intended behavior"
    )]

    use std::io;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    #[test]
    fn tee_writer_write_appends_to_file_and_returns_full_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tee.log");
        let file = std::fs::File::create(&path).unwrap();
        let mut tee = TeeWriter::new(file);

        let n = std::io::Write::write(&mut tee, b"hello\n").unwrap();

        assert_eq!(n, 6);
        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents, "hello\n");
    }

    #[test]
    fn tee_writer_flush_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tee.log");
        let file = std::fs::File::create(&path).unwrap();
        let mut tee = TeeWriter::new(file);

        std::io::Write::write_all(&mut tee, b"data").unwrap();
        assert!(std::io::Write::flush(&mut tee).is_ok());
    }

    #[test]
    fn both_returns_ok_when_both_succeed() {
        assert!(both(Ok(()), Ok(())).is_ok());
    }

    #[test]
    fn both_returns_first_error_when_first_fails() {
        let err = both(
            Err(io::Error::other("first")),
            Err(io::Error::other("second")),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "first");
    }

    #[test]
    fn both_returns_second_error_when_only_second_fails() {
        let err = both(Ok(()), Err(io::Error::other("second"))).unwrap_err();
        assert_eq!(err.to_string(), "second");
    }

    #[test]
    fn both_attempts_second_even_when_first_fails() {
        // The second closure's side effect (incrementing this counter) must
        // still run even though the first result is already an error;
        // `both` must not short-circuit before attempting both sinks.
        let attempted = Arc::new(AtomicBool::new(false));
        let attempted_clone = Arc::clone(&attempted);

        let second_result: io::Result<()> = {
            attempted_clone.store(true, Ordering::SeqCst);
            Ok(())
        };

        let _ = both(Err(io::Error::other("first")), second_result);
        assert!(attempted.load(Ordering::SeqCst));
    }
}
