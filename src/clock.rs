use std::io;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClockIdentity {
    pub domain: String,
    pub boot_id: String,
    pub time_namespace: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Timestamp {
    pub identity: ClockIdentity,
    pub nanoseconds: u64,
}

impl Timestamp {
    pub fn elapsed_since(&self, previous: &Self) -> Option<u64> {
        if self.identity != previous.identity {
            return None;
        }
        self.nanoseconds.checked_sub(previous.nanoseconds)
    }
}

pub trait Clock {
    fn now(&mut self) -> io::Result<Timestamp>;
}

/// Native Linux CLOCK_MONOTONIC, accompanied by boot and time-namespace identity.
#[derive(Default)]
pub struct LinuxClock;

#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
impl Clock for LinuxClock {
    fn now(&mut self) -> io::Result<Timestamp> {
        use std::os::raw::{c_int, c_long};

        #[repr(C)]
        struct Timespec {
            seconds: c_long,
            nanoseconds: c_long,
        }

        extern "C" {
            fn clock_gettime(clock_id: c_int, value: *mut Timespec) -> c_int;
        }

        fn identity() -> io::Result<ClockIdentity> {
            let boot_id = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
            let namespace = match std::fs::read_link("/proc/self/ns/time") {
                Ok(path) => path.to_string_lossy().into_owned(),
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    // Kernels predating time namespaces have one initial namespace.
                    "initial_namespace_without_time_namespace_support".to_string()
                }
                Err(error) => return Err(error),
            };
            Ok(ClockIdentity {
                domain: "linux_clock_monotonic".to_string(),
                boot_id: boot_id.trim().to_string(),
                time_namespace: namespace,
            })
        }

        let before = identity()?;
        let mut value = Timespec {
            seconds: 0,
            nanoseconds: 0,
        };
        // SAFETY: value is a writable native Linux 64-bit timespec. The call
        // writes it synchronously; CLOCK_MONOTONIC is clock ID 1 on Linux.
        if unsafe { clock_gettime(1, &mut value) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let after = identity()?;
        if before != after {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "clock identity changed during observation",
            ));
        }
        if value.seconds < 0 || !(0..1_000_000_000).contains(&value.nanoseconds) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid CLOCK_MONOTONIC timestamp",
            ));
        }
        let nanoseconds = (value.seconds as u64)
            .checked_mul(1_000_000_000)
            .and_then(|seconds| seconds.checked_add(value.nanoseconds as u64))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "clock overflow"))?;
        Ok(Timestamp {
            identity: after,
            nanoseconds,
        })
    }
}

#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
impl Clock for LinuxClock {
    fn now(&mut self) -> io::Result<Timestamp> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native collection requires a 64-bit Linux CLOCK_MONOTONIC implementation",
        ))
    }
}
