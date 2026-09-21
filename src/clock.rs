use crate::{Clock, Timestamp};

#[cfg(target_os = "linux")]
#[derive(Default)]
pub struct MonotonicClock;

#[cfg(target_os = "linux")]
impl Clock for MonotonicClock {
    fn now(&self) -> Timestamp {
        use std::os::raw::{c_int, c_long};

        #[repr(C)]
        struct Timespec {
            seconds: c_long,
            nanoseconds: c_long,
        }

        extern "C" {
            fn clock_gettime(clock_id: c_int, time: *mut Timespec) -> c_int;
        }

        let mut time = Timespec {
            seconds: 0,
            nanoseconds: 0,
        };
        // CLOCK_MONOTONIC is 1 on Linux. The pointer refers to a writable,
        // correctly laid out timespec for the duration of this call.
        let result = unsafe { clock_gettime(1, &mut time) };
        assert_eq!(result, 0, "CLOCK_MONOTONIC is unavailable");
        Timestamp {
            ns: (time.seconds as u64)
                .saturating_mul(1_000_000_000)
                .saturating_add(time.nanoseconds as u64),
            clock_domain: "linux_clock_monotonic".into(),
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub struct MonotonicClock {
    origin: std::time::Instant,
}

#[cfg(not(target_os = "linux"))]
impl Default for MonotonicClock {
    fn default() -> Self {
        Self {
            origin: std::time::Instant::now(),
        }
    }
}

#[cfg(not(target_os = "linux"))]
impl Clock for MonotonicClock {
    fn now(&self) -> Timestamp {
        Timestamp {
            ns: self.origin.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            clock_domain: "process_local_monotonic".into(),
        }
    }
}

impl MonotonicClock {
    pub fn new() -> Self {
        Self::default()
    }
}
