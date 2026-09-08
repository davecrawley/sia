use crate::model::{ObservationTime, LINUX_MONOTONIC_CLOCK};

pub trait Clock {
    fn domain(&self) -> &'static str;
    fn now(&self) -> ObservationTime;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeClock;

#[cfg(target_os = "linux")]
#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

#[cfg(target_os = "linux")]
extern "C" {
    fn clock_gettime(clock_id: i32, time: *mut Timespec) -> i32;
}

impl Clock for NativeClock {
    fn domain(&self) -> &'static str {
        LINUX_MONOTONIC_CLOCK
    }

    fn now(&self) -> ObservationTime {
        #[cfg(target_os = "linux")]
        {
            const CLOCK_MONOTONIC: i32 = 1;
            let mut value = Timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            // SAFETY: `value` is a valid writable timespec and CLOCK_MONOTONIC
            // requires no additional caller-managed state.
            let result = unsafe { clock_gettime(CLOCK_MONOTONIC, &mut value) };
            assert_eq!(result, 0, "clock_gettime(CLOCK_MONOTONIC) failed");
            assert!(
                value.tv_sec >= 0 && value.tv_nsec >= 0,
                "clock_gettime(CLOCK_MONOTONIC) returned an invalid timespec"
            );
            ObservationTime {
                monotonic_ns: (value.tv_sec as u64)
                    .saturating_mul(1_000_000_000)
                    .saturating_add(value.tv_nsec as u64),
                clock_domain: LINUX_MONOTONIC_CLOCK,
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            panic!("the native Linux monotonic clock is supported only on Linux")
        }
    }
}
