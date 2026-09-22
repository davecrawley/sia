pub mod clock;
pub mod collection;
pub mod model;
pub mod providers;

use clock::NativeClock;
use collection::Collector;
use providers::{NativeProvider, NvidiaProvider};

pub fn native_collector() -> Collector<NativeClock> {
    Collector::new(
        NativeClock,
        vec![
            Box::new(NativeProvider::new()),
            Box::new(NvidiaProvider::native()),
        ],
    )
}
