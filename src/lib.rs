#![allow(non_snake_case)]
pub mod force_atlas;
pub mod io;
pub mod run_script;
pub mod util;

#[cfg(test)]
mod test {
    use crate::run_script::*;
    use std::path::Path;

    #[test]
    fn ca_netscience() {
        let path = Path::new("res/ca-netscience/ca-netscience.mtx");
        run_script(path, 2)
    }
}
