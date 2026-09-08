use crate::model::{ObservationTime, LINUX_MONOTONIC_CLOCK};

#[cfg(target_os = "linux")]
use std::ffi::{c_int, c_long};
#[cfg(not(target_os = "linux"))]
use std::sync::OnceLock;
#[cfg(not(target_os = "linux"))]
use std::time::Instant;

#[cfg(not(target_os = "linux"))]
const PROCESS_RELATIVE_CLOCK: &str = "process_relative_instant";

pub trait Clock {
    fn domain(&self) -> &'static str;
    fn now(&self) -> ObservationTime;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeClock;

#[cfg(target_os = "linux")]
#[repr(C)]
struct Timespec {
    tv_sec: c_long,
    tv_nsec: c_long,
}

#[cfg(target_os = "linux")]
extern "C" {
    fn clock_gettime(clock_id: c_int, time: *mut Timespec) -> c_int;
}

impl Clock for NativeClock {
    fn domain(&self) -> &'static str {
        #[cfg(target_os = "linux")]
        {
            LINUX_MONOTONIC_CLOCK
        }
        #[cfg(not(target_os = "linux"))]
        {
            PROCESS_RELATIVE_CLOCK
        }
    }

    fn now(&self) -> ObservationTime {
        #[cfg(target_os = "linux")]
        {
            const CLOCK_MONOTONIC: c_int = 1;
            let mut value = Timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            // SAFETY: `value` is writable for the duration of the call and
            // CLOCK_MONOTONIC requires no additional caller-managed state.
            let result = unsafe { clock_gettime(CLOCK_MONOTONIC, &mut value) };
            assert_eq!(result, 0, "clock_gettime(CLOCK_MONOTONIC) failed");
            assert!(
                value.tv_sec >= 0 && (0..1_000_000_000).contains(&value.tv_nsec),
                "clock_gettime(CLOCK_MONOTONIC) returned an invalid timespec"
            );
            ObservationTime {
                monotonic_ns: (value.tv_sec as u64)
                    .saturating_mul(1_000_000_000)
                    .saturating_add(value.tv_nsec as u64),
                clock_domain: self.domain(),
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            static ORIGIN: OnceLock<Instant> = OnceLock::new();
            let elapsed = ORIGIN.get_or_init(Instant::now).elapsed();
            ObservationTime {
                monotonic_ns: elapsed.as_nanos().min(u64::MAX as u128) as u64,
                clock_domain: self.domain(),
            }
        }
    }
}
