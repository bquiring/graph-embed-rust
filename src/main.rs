#![allow(non_snake_case)]
use std::path::Path;

mod run_script;

fn ca_netscience() {
    println!("running ca-netscience example...");
    let path = Path::new("res/ca-netscience/ca-netscience.mtx");
    let zero_indexed = false;
    run_script::run_script(path, zero_indexed, 2, true);
}

fn road_usroads() {
    println!("running road-usroads example...");
    let path = Path::new("res/road-usroads/road-usroads.mtx");
    let zero_indexed = false;
    run_script::run_script(path, zero_indexed, 2, true);
}

fn main() {
    ca_netscience();
    //road_usroads();
}
