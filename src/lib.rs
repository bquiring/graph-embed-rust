#![allow(non_snake_case)]

pub mod forceatlas;
pub mod util;
pub mod run_script;
pub mod io;

#[cfg(test)]
mod test {
    use std::path::Path;
    use crate::run_script::*;

    #[test]
    fn ca_netscience() {
        let path = Path::new("res/ca-netscience/ca-netscience.mtx");
        run_script (path, 2)
    }
}
