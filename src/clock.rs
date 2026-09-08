use crate::model::ObservationTime;

pub const LINUX_MONOTONIC_CLOCK: &str = "linux_clock_monotonic";

pub trait Clock: Send + Sync {
    fn now(&self) -> ObservationTime;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeClock;

impl Clock for NativeClock {
    fn now(&self) -> ObservationTime {
        #[cfg(target_os = "linux")]
        {
            let mut value = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            // SAFETY: `value` is a valid writable timespec and CLOCK_MONOTONIC
            // requires no additional caller-managed state.
            let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut value) };
            assert_eq!(result, 0, "clock_gettime(CLOCK_MONOTONIC) failed");

            ObservationTime {
                monotonic_ns: (value.tv_sec as u64)
                    .saturating_mul(1_000_000_000)
                    .saturating_add(value.tv_nsec as u64),
                clock_domain: LINUX_MONOTONIC_CLOCK,
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            panic!("the native monotonic clock is currently supported only on Linux")
        }
    }
}
