use crate::model::{ObservationTime, LINUX_MONOTONIC_CLOCK};
use std::sync::OnceLock;
use std::time::Instant;

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
            let result = unsafe { clock_gettime(CLOCK_MONOTONIC, &mut value) };
            if result == 0 && value.tv_sec >= 0 && value.tv_nsec >= 0 {
                ObservationTime {
                    monotonic_ns: (value.tv_sec as u64)
                        .saturating_mul(1_000_000_000)
                        .saturating_add(value.tv_nsec as u64),
                    clock_domain: LINUX_MONOTONIC_CLOCK,
                }
            } else {
                fallback_time()
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            fallback_time()
        }
    }
}

fn fallback_time() -> ObservationTime {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    let elapsed = ORIGIN.get_or_init(Instant::now).elapsed();
    ObservationTime {
        monotonic_ns: elapsed.as_nanos().min(u64::MAX as u128) as u64,
        clock_domain: LINUX_MONOTONIC_CLOCK,
    }
}
