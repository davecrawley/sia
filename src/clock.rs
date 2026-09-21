//! Observation time is supplied independently of the requested sampling period.

use crate::model::ClockIdentity;
use std::io;

pub trait Clock {
    fn now_ns(&mut self) -> io::Result<u64>;
    fn identity(&self) -> ClockIdentity;
}

pub struct SystemClock {
    identity: ClockIdentity,
    #[cfg(not(target_os = "linux"))]
    origin: std::time::Instant,
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemClock {
    pub fn new() -> Self {
        #[cfg(target_os = "linux")]
        let identity = ClockIdentity {
            domain: "linux_clock_monotonic".into(),
            boot_id: std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
                .ok()
                .map(|id| id.trim().to_owned()),
            time_namespace: std::fs::read_link("/proc/self/ns/time")
                .ok()
                .map(|path| path.to_string_lossy().into_owned()),
        };
        #[cfg(not(target_os = "linux"))]
        let identity = ClockIdentity {
            domain: "process_relative_monotonic".into(),
            boot_id: None,
            time_namespace: None,
        };
        Self {
            identity,
            #[cfg(not(target_os = "linux"))]
            origin: std::time::Instant::now(),
        }
    }
}

impl Clock for SystemClock {
    fn now_ns(&mut self) -> io::Result<u64> {
        #[cfg(target_os = "linux")]
        {
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
            // Linux CLOCK_MONOTONIC is 1. The initialized, writable structure
            // has the C timespec layout used by this libc entry point.
            let result = unsafe { clock_gettime(1, &mut time) };
            if result != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(time.seconds as u64 * 1_000_000_000 + time.nanoseconds as u64)
        }
        #[cfg(not(target_os = "linux"))]
        {
            Ok(self.origin.elapsed().as_nanos().min(u64::MAX as u128) as u64)
        }
    }

    fn identity(&self) -> ClockIdentity {
        self.identity.clone()
    }
}
