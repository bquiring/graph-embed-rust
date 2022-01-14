use graph_embed_rust::{community::*, embed::*, force_atlas::*, io::*};
use nalgebra_sparse::csr::CsrMatrix;
use std::{fs::File, io::Write, path::Path, process::Command, time::Instant};

fn make_PT(level: &Level) -> CsrMatrix<f64> {
    let n = level.num_nodes();
    let m = level.num_comm();

    // Form P transpose
    let mut PT_I = vec![0; m + 1];
    let mut PT_J = vec![0; n];
    let PT_D = vec![1.0; n];

    for a in 0..m {
        PT_I[a + 1] = level.comm_size(a).unwrap();
    }

    for a in 0..m {
        PT_I[a + 1] += PT_I[a];
    }

    let mut count = vec![0; m];
    for i in 0..n {
        let a = level.comm_of(i).unwrap();
        // println! ("{:?} {:?}", a, m);
        assert!(count[a] < PT_I[a + 1] - PT_I[a]);
        PT_J[PT_I[a] + count[a]] = i;
        count[a] += 1;
    }
    CsrMatrix::try_from_csr_data(m, n, PT_I, PT_J, PT_D).unwrap()
}

pub fn run_script(graph_path: &Path, zero_indexed: bool, dim: usize, plot: bool) {
    println!("---");
    let start = Instant::now();
    let mm = MatrixMarket::read_from_path(graph_path, zero_indexed).unwrap();
    let coo = mm.to_sym_coo();
    let A = CsrMatrix::from(&coo);
    println!("input time elapsed: {:?}", start.elapsed());

    assert_eq!(A.nrows(), A.ncols());
    let n = A.nrows();

    let start = Instant::now();
    let levels = louvain(&A, 0.000001);
    println!("community time elapsed: {:?}", start.elapsed());

    println!("---");
    let partial = levels.len();
    let mut As = Vec::with_capacity(partial + 1);
    let mut PTs = Vec::with_capacity(partial + 1);
    println!("level 0 has {} vertices", A.nrows());
    As.push(A);
    for (i, level) in levels.iter().enumerate().take(partial) {
        let PT = make_PT(level);
        let A = &As[i];

        // form the quotient graph
        let Ac = &PT * A * PT.transpose();

        println!("level {} has {} vertices", i + 1, Ac.nrows());
        // remember these
        PTs.push(PT);
        As.push(Ac);
    }
    println!("---");

    let start = Instant::now();
    let coords = embed_multilevel(
        &As,
        &PTs,
        dim,
        |A, dim, coords| {
            force_atlas(A, dim, 300, coords, &ForceAtlasArgs::default());
        },
        |A, dim, coords, coords_Ac, PT| {
            force_atlas_multilevel(
                A,
                dim,
                50,
                coords,
                coords_Ac,
                PT,
                &ForceAtlasArgs::default(),
            );
        },
    );
    println!("embedding time elapsed: {:?}", start.elapsed());

    let part_path = graph_path.with_extension("part");
    {
        let start = Instant::now();
        let k = PTs.len();
        let mut part_file = File::create(&part_path).unwrap();
        // print #vertices then #partition levels
        writeln!(part_file, "{} {}", n, k).unwrap();
        // print the partitions
        for PT in PTs.iter() {
            // print the size of each partition
            writeln!(part_file, "{}", PT.nrows()).unwrap();
            let PT_I = PT.row_offsets();
            let PT_J = PT.col_indices();
            for a in 0..PT.nrows() {
                for index in PT_I[a]..PT_I[a + 1] {
                    write!(part_file, "{} ", PT_J[index]).unwrap();
                }
                writeln!(part_file).unwrap();
            }
            //for comm in 0..level.num_comm() {
            //    for node in level.iter().filter(|(_, c)| *c == comm).map(|(n, _)| n) {
            //        write!(part_file, "{} ", node).unwrap();
            //    }
            //    writeln!(part_file).unwrap();
            //}
        }

        // for comm in 0..level.num_comm() {
        //     for node in level.nodes(comm) {
        //         write!(part_file, "{} ", node).unwrap();
        //     }
        //     writeln!(part_file).unwrap();
        // }
        // writeln!(part_file).unwrap();
        println!("partition time elapsed: {:?}", start.elapsed());
    }

    let scale = 10.0;
    let coords_path = graph_path.with_extension("coords");
    {
        let mut coords_file = File::create(&coords_path).unwrap();
        for i in 0..n {
            for k in 0..dim {
                write!(coords_file, "{} ", scale * coords[i][k]).unwrap();
            }
            writeln!(coords_file).unwrap();
        }
    }

    let plot_path = graph_path.with_extension("plot");

    println!("---");
    let mut args = vec![
        "scripts/plot-graph.py",
        "-graph",
        graph_path.to_str().unwrap(),
        "-part",
        part_path.to_str().unwrap(),
        "-coords",
        coords_path.to_str().unwrap(),
        "-o",
        plot_path.to_str().unwrap(),
    ];

    if !zero_indexed {
        args.push("--not-zero-indexed");
    }

    println!("python3 {}", args.join(" "));
    println!("---");
    if plot {
        let output = Command::new("python3")
            .args(args)
            .output()
            .expect("failed to execute process");
        println!("{:?}", output);
    }
}
