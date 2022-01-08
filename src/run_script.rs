use graph_embed_rust::{community::*, force_atlas::*, io::*};
use nalgebra::base::DMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use rand::{distributions::Uniform, Rng};
use std::{fs::File, io::Write, path::Path, process::Command, time::Instant};

pub fn run_script(graph_path: &Path, dim: usize) {
    let mm = MatrixMarket::read_from_path(graph_path).unwrap();
    let coo = mm.to_sym_coo();
    let m = CsrMatrix::from(&coo);

    assert_eq!(m.nrows(), m.ncols());
    let n = m.nrows();

    let mut rng = rand::thread_rng();
    let dist = Uniform::from(-1.0..1.0);
    let mut rand_elems = Vec::with_capacity(n * dim);
    rand_elems.extend((0..n * dim).map(|_| rng.sample(&dist)));
    let mut coords = DMatrix::from_vec(n, dim, rand_elems);

    let start = Instant::now();
    force_atlas(&m, dim, 1, &mut coords, &ForceAtlasArgs::default());
    //coords = coords.normalize();
    let duration = start.elapsed();
    println!("force atlas time elapsed: {:?}", duration);

    let start = Instant::now();
    let levels = louvain(&m, 0.000001);
    let duration = start.elapsed();
    println!("community time elapsed: {:?}", duration);

    let part_path = graph_path.with_extension("part");
    {
        let start = Instant::now();
        let k = 1; // #partition level
        let mut part_file = File::create(&part_path).unwrap();
        // print #vertices then #partition levels
        writeln!(part_file, "{} {}", n, k).unwrap();
        // print the partitions
        for level in levels.iter().take(k) {
            // print the size of each partition
            writeln!(part_file, "{}", level.num_comm()).unwrap();
            for comm in 0..level.num_comm() {
                for node in level.iter().filter(|(_, c)| *c == comm).map(|(n, _)| n) {
                    write!(part_file, "{} ", node).unwrap();
                }
                writeln!(part_file).unwrap();
            }
        }

        // for comm in 0..level.num_comm() {
        //     for node in level.nodes(comm) {
        //         write!(part_file, "{} ", node).unwrap();
        //     }
        //     writeln!(part_file).unwrap();
        // }
        // writeln!(part_file).unwrap();
        let duration = start.elapsed();
        println!("partition time elapsed: {:?}", duration);
    }

    let coords_path = graph_path.with_extension("coords");
    {
        let mut coords_file = File::create(&coords_path).unwrap();
        for i in 0..n {
            for k in 0..dim {
                write!(coords_file, "{} ", coords[(i, k)]).unwrap();
            }
            writeln!(coords_file).unwrap();
        }
    }

    let plot_path = graph_path.with_extension("plot");

    println!(
        "python3 scripts/plot-graph.py -graph {} -part {} -coords {} -o {}",
        graph_path.to_str().unwrap(),
        part_path.to_str().unwrap(),
        coords_path.to_str().unwrap(),
        plot_path.to_str().unwrap()
    );

    let output = Command::new("python3")
        .args([
            "scripts/plot-graph.py",
            "-graph",
            graph_path.to_str().unwrap(),
            "-part",
            part_path.to_str().unwrap(),
            "-coords",
            coords_path.to_str().unwrap(),
            "-o",
            plot_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to execute process");

    println!("{:?}", output);
}
