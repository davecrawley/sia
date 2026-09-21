//! Hardware access belongs exclusively to providers.

mod system;
pub use system::SystemProvider;

#[cfg(feature = "nvidia")]
pub mod nvidia;

use crate::collection::Provider;

pub fn local_providers() -> Vec<Box<dyn Provider>> {
    let providers: Vec<Box<dyn Provider>> = vec![Box::new(SystemProvider::new())];
    #[cfg(feature = "nvidia")]
    {
        let mut providers = providers;
        providers.push(nvidia::local_provider());
        providers
    }
    #[cfg(not(feature = "nvidia"))]
    {
        providers
    }
}
