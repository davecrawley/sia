use crate::model::MonotonicTimestamp;
use std::{collections::VecDeque, fmt, sync::Mutex};

pub const LINUX_MONOTONIC_DOMAIN: &str = "linux_clock_monotonic";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClockError(pub String);
impl fmt::Display for ClockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ClockError {}

pub trait Clock: Send + Sync {
    fn domain(&self) -> &'static str;
    fn now(&self) -> Result<MonotonicTimestamp, ClockError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LinuxMonotonicClock;
impl Clock for LinuxMonotonicClock {
    fn domain(&self) -> &'static str {
        LINUX_MONOTONIC_DOMAIN
    }
    fn now(&self) -> Result<MonotonicTimestamp, ClockError> {
        linux_monotonic_nanoseconds().map(MonotonicTimestamp)
    }
}

#[cfg(target_os = "linux")]
fn linux_monotonic_nanoseconds() -> Result<u64, ClockError> {
    use std::os::raw::{c_int, c_long};
    #[repr(C)]
    struct Timespec {
        tv_sec: c_long,
        tv_nsec: c_long,
    }
    const CLOCK_MONOTONIC: c_int = 1;
    extern "C" {
        fn clock_gettime(clock_id: c_int, time: *mut Timespec) -> c_int;
    }
    let mut time = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if unsafe { clock_gettime(CLOCK_MONOTONIC, &mut time) } != 0 {
        return Err(ClockError(std::io::Error::last_os_error().to_string()));
    }
    if time.tv_sec < 0 || time.tv_nsec < 0 {
        return Err(ClockError(
            "CLOCK_MONOTONIC returned a negative value".into(),
        ));
    }
    Ok((time.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(time.tv_nsec as u64))
}

#[cfg(not(target_os = "linux"))]
fn linux_monotonic_nanoseconds() -> Result<u64, ClockError> {
    Err(ClockError(
        "linux_clock_monotonic is only available on Linux".into(),
    ))
}

#[derive(Debug)]
pub struct DeterministicClock {
    times: Mutex<VecDeque<MonotonicTimestamp>>,
}
impl DeterministicClock {
    pub fn new(times: impl IntoIterator<Item = MonotonicTimestamp>) -> Self {
        Self {
            times: Mutex::new(times.into_iter().collect()),
        }
    }
}
impl Clock for DeterministicClock {
    fn domain(&self) -> &'static str {
        LINUX_MONOTONIC_DOMAIN
    }
    fn now(&self) -> Result<MonotonicTimestamp, ClockError> {
        self.times
            .lock()
            .map_err(|_| ClockError("deterministic clock lock was poisoned".into()))?
            .pop_front()
            .ok_or_else(|| ClockError("deterministic clock has no remaining timestamps".into()))
    }
}
