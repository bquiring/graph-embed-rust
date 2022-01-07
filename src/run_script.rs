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
    force_atlas(&m, dim, 1000, &mut coords, &ForceAtlasArgs::default());
    //coords = coords.normalize();
    let duration = start.elapsed();
    println!("force atlas time elapsed: {:?}", duration);

    let levels = louvain(&m, 0.000001);
    for (i, level) in levels.iter().enumerate() {
        println!("level {}:", i);
        for (node, comm) in level.sorted() {
            println!("{} {}", node, comm);
        }
        println!();
    }

    // #partition levels
    // FIXME
    let K = 1;
    let part_path = graph_path.with_extension("part");
    {
        let mut part_file = File::create(part_path.clone()).unwrap();
        // print #vertices then #partition levels
        writeln!(part_file, "{} {}", n, K).unwrap();
        // print the size of each partition
        // FIXME
        // TODO: SHOULD BE ALL ON ONE LINE
        for p in 0..K {
            writeln!(part_file, "{}", n).unwrap();
        }
        // print the partitions
        for p in 0..K {
            // FIXME: should be the size of level K
            let n = n;
            for i in 0..n {
                // write associated vertex index in level p+1
                writeln!(part_file, "{}", 0).unwrap();
            }
        }
        writeln!(part_file).unwrap();
    }

    let coords_path = graph_path.with_extension("coords");
    {
        let mut coords_file = File::create(coords_path.clone()).unwrap();
        for i in 0..n {
            for k in 0..dim {
                write!(coords_file, "{} ", coords[(i, k)]).unwrap();
            }
            writeln!(coords_file).unwrap();
        }
    }

    let plot_path = graph_path.with_extension("plot");

    println!("python3 scripts/plot-graph.py -graph {} -part {} -coords {} -o {}",
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
