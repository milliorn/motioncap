use std::sync::{Condvar, Mutex};

/// Signals when the writer thread has completed its post-shutdown
/// last-chance drain (see `run_recording_writer_loop`), so the detection
/// loop's shutdown path can block on it instead of finalizing the active
/// event prematurely. A condvar rather than a busy-polled flag, since this
/// is a one-shot handshake during shutdown, not a recurring cadence.
#[derive(Default)]
pub struct WriterDrained {
    /// Set to `true` once the writer thread's final drain has completed.
    done: Mutex<bool>,
    /// Notified when `done` is set, to wake `wait`'s blocked receiver.
    condvar: Condvar,
}

impl WriterDrained {
    /// Marks the final drain as complete and wakes any thread blocked in `wait`.
    pub fn signal(&self) {
        *self.done.lock().expect("writer-drained lock poisoned") = true;
        self.condvar.notify_one();
    }

    /// Blocks until `signal` has been called.
    pub fn wait(&self) {
        let guard = self.done.lock().expect("writer-drained lock poisoned");

        drop(
            self.condvar
                .wait_while(guard, |done| !*done)
                .expect("writer-drained lock poisoned"),
        );
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the `WriterDrained` shutdown handshake primitive.
    #![allow(
        clippy::unwrap_used,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap for clarity; panics here fail the test, which is the intended behavior"
    )]

    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

    /// Delay before signaling, in the blocking test, long enough that
    /// `wait()` would return immediately (a bug) rather than actually
    /// blocking if it didn't work.
    const SIGNAL_DELAY: Duration = Duration::from_millis(50);

    #[test]
    fn writer_drained_signal_then_wait_returns_immediately() {
        let drained = WriterDrained::default();
        drained.signal();
        drained.wait(); // must not block
    }

    #[test]
    fn writer_drained_wait_blocks_until_signaled() {
        let drained = Arc::new(WriterDrained::default());
        let signaler = Arc::clone(&drained);

        let handle = std::thread::spawn(move || {
            std::thread::sleep(SIGNAL_DELAY);
            signaler.signal();
        });

        drained.wait();
        handle.join().unwrap();
    }
}
