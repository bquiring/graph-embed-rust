
use nalgebra_sparse::csr::CsrMatrix;
use std::process::Command;
use rand;
use std::path::Path;
use std::fs::File;
use std::io::Write;

fn run_script (graph_path : &Path, dim : usize) {

    let mm = MatrixMarket::read(&path).unwrap();
    let coo = mm.to_coo();
    let m = CsrMatrix::from(&coo);

    assert!(m.nrows() == m.ncols());
    let n = m.nrows();

    let mut coords = vec![vec![0.0; dim]; m.nrows()];

    let mut rng = rand::task_rng();
    for i in 0..n {
        coords[i] = rng.gen_range (-1.0, 1.0);
    }
    
    force_atlas (&m, dim, &coords, 1000, ForceAtlasArgs::default())

    let part_path = graph_path.with_extension ("part");
    {
        let mut part_file = File::create(part_path)?;
        write!(part_file, format!("{} 1\n", n));
        write!(part_file, format!("{}\n", n));
        for i in 0..n {
            write!(part_file, format!("{} ", i));
        }
        write!("\n");
    }

    let coords_path = graph_path.with_extension ("coord");
    {
        let mut coords_file = File::create(coords_path)?;
        for i in 0..n {
            for k in 0..dim {
                write!(coords_file, format!("{} ", coords[i][k]));
            }
            write!("\n");
        }
    }

    let plot_path = graph_path.with_extension ("plot");

    let output =
        Command::new("python3")
        .args(["scripts/plot-graph.py",
               "-graph", graph_path,
               "-part", partpath,
               "-coords", coordspath,
               "-o", plot_path])
        .output()
        .expect("failed to execute process");



}
