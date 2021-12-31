use std::path::Path;

mod run_script;

fn ca_netscience() {
    println!("running ca-netscience example...");
    let path = Path::new("res/ca-netscience/ca-netscience.mtx");
    run_script::run_script(path, 2);
}

fn main() {
    ca_netscience();
}
