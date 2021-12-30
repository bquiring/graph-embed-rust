use std::{path::Path, time::Instant};

mod run_script;

fn ca_netscience() {
    println!("running ca-netscience example...");
    let path = Path::new("res/ca-netscience/ca-netscience.mtx");
    let start = Instant::now();
    run_script::run_script(path, 2);
    let duration = start.elapsed();
    println!("elapsed: {:?}", duration);
}

fn main() {
    ca_netscience();
}
