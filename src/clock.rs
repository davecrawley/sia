use crate::model::Timestamp;
use std::io;

pub trait Clock {
    fn now(&mut self) -> io::Result<Timestamp>;
}

/// Reads the host clock and its identity, without initializing graphics.
#[derive(Default)]
pub struct NativeClock;

#[cfg(target_os = "linux")]
impl Clock for NativeClock {
    fn now(&mut self) -> io::Result<Timestamp> {
        use crate::model::ClockIdentity;
        use std::os::raw::{c_int, c_long};

        #[repr(C)]
        struct Timespec {
            tv_sec: c_long,
            tv_nsec: c_long,
        }

        extern "C" {
            fn clock_gettime(clock_id: c_int, value: *mut Timespec) -> c_int;
        }

        let boot_id = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?
            .trim()
            .to_owned();
        let time_namespace_id = std::fs::read_link("/proc/self/ns/time")?
            .to_string_lossy()
            .into_owned();
        let mut value = Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // Linux CLOCK_MONOTONIC is 1. The C ABI writes one initialized timespec.
        if unsafe { clock_gettime(1, &mut value) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let seconds = u64::try_from(value.tv_sec)
            .map_err(|_| io::Error::other("negative monotonic clock"))?;
        let nanos = u64::try_from(value.tv_nsec)
            .map_err(|_| io::Error::other("negative clock nanoseconds"))?;
        if nanos >= 1_000_000_000 {
            return Err(io::Error::other("invalid clock nanoseconds"));
        }
        let mono_ns = seconds
            .checked_mul(1_000_000_000)
            .and_then(|seconds| seconds.checked_add(nanos))
            .ok_or_else(|| io::Error::other("monotonic timestamp overflow"))?;
        Ok(Timestamp {
            identity: ClockIdentity {
                domain: "linux_clock_monotonic".into(),
                boot_id,
                time_namespace_id,
            },
            mono_ns,
        })
    }
}

#[cfg(not(target_os = "linux"))]
impl Clock for NativeClock {
    fn now(&mut self) -> io::Result<Timestamp> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native collection requires Linux CLOCK_MONOTONIC",
        ))
    }
}
