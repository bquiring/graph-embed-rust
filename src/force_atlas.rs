use crate::grid::Grid;
use crate::util::*;
//use nalgebra::base::DMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use rayon::prelude::*;

pub struct ForceAtlasArgs {
    pub ks: f64,
    pub ksmax: f64,
    pub repel: f64,
    pub attract: f64,
    pub gravity: f64,
    //pub delta: f64,
    pub tolerate: f64,
    pub use_weights: bool,
    pub linlog: bool,
    pub no_hubs: bool,
}

impl Default for ForceAtlasArgs {
    fn default() -> Self {
        Self {
            ks: 0.1,
            ksmax: 1.0,
            repel: 1.0,
            attract: 1.0,
            gravity: 1.0,
            //delta: 1.0,
            tolerate: 1.0,
            use_weights: true,
            linlog: false,
            no_hubs: false,
        }
    }
}

pub fn force_atlas(
    matrix: &CsrMatrix<f64>,
    dim: usize,
    iter: usize,
    //    coords: &mut DMatrix<f64>,
    coords: &mut Grid<f64>,
    args: &ForceAtlasArgs,
) {
    let epsilon = 0.00001;
    let n = coords.nrows();

    assert_eq!(matrix.nrows(), matrix.ncols());
    assert_eq!(matrix.nrows(), n);
    //    for i in 0..n {
    //        assert_eq!(coords[i].len(), dim);
    //    }
    assert_eq!(coords.ncols(), dim);

    let I = matrix.row_offsets();
    let J = matrix.col_indices();
    let D = matrix.values();

    let mut deg = vec![0.0; n];
    if args.use_weights {
        for i in 0..n {
            let sum: f64 = D[I[i]..I[i + 1]].iter().sum();
            deg[i] = sum + 1.0;
        }
    } else {
        for i in 0..n {
            deg[i] = (I[i + 1] - I[i]) as f64 + 1.0;
        }
    }

    let mut forces_prev = Grid::new(n, dim);
    let mut forces = Grid::new(n, dim);
    let mut swing = vec![0.0; n];

    for _ in 0..iter {
        // TODO: parallelize here... no idea if this is right
        forces.par_iter_mut().enumerate().for_each(|(i, row)| {
            let mut force_i = vec![0.0; dim];
            for j in 0..n {
                if i != j {
                    //                    let dis_ij = coords.row(i).metric_distance(&coords.row(j)).max(epsilon);
                    let dis_ij = distance(&coords[i], &coords[j]).max(epsilon);
                    let Fr_ij = deg[i] * deg[j] * args.repel / (dis_ij * dis_ij);

                    for k in 0..dim {
                        let direction = -(coords[(j, k)] - coords[(i, k)]) / dis_ij;
                        let Fr_sum = direction * Fr_ij;
                        force_i[k] += Fr_sum;
                    }
                }
            }

            for k2 in I[i]..I[i + 1] {
                let j = J[k2];
                //                let dis_ij = coords.row(i).metric_distance(&coords.row(j)).max(epsilon);
                let dis_ij = distance(&coords[i], &coords[j]).max(epsilon);
                let mut fa_ij = if args.linlog { dis_ij.log2() } else { dis_ij };
                let a_ij = if args.use_weights { D[k2] } else { 1.0 };

                // if args.delta == 1.0 /* fix */ {
                //     fa_ij *= a_ij;
                // } else if args.delta != 0.0 {
                //     fa_ij = (if a_ij < 0 { -1 } else { 1 }) * a_ij.abs().pow(delta) * fa_ij;
                // }
                fa_ij *= a_ij;

                if args.no_hubs {
                    fa_ij /= deg[i];
                }

                let Fa_ij = args.attract * fa_ij;
                for k in 0..dim {
                    let direction = (coords[(j, k)] - coords[(i, k)]) / dis_ij;
                    let Fa_sum = direction * Fa_ij;
                    force_i[k] += Fa_sum;
                }
            }

            //            let mag = coords.row(i).magnitude();
            let mag = magnitude(&coords[i]);
            for (k, force) in row.iter_mut().enumerate() {
                let Fg_ki = -coords[(i, k)] / mag * args.gravity * deg[i];
                *force = force_i[k] + Fg_ki;
            }
        });

        // TODO: parallize here
        swing
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = distance(&forces[i], &forces_prev[i]));

        //let mut global_swing = 0.0;
        //for i in 0..n {
        //    globalSwing += ((deg[i] + 1.0) * swing[i]).max(epsilon);
        //}
        let global_swing = 1.0;

        //let mut global_traction = 0.0;
        //for i in 0..n {
        //    let traction_i = distance (&forces[i], &forces_prev[i]) / 2.0;
        //    global_traction += (deg[i]+1.0) / traction_i;
        //}
        let global_traction = 1.0;
        let global_speed = args.tolerate * global_traction / global_swing;

        // TODO: parallelize here
        //        coords.row_iter_mut().enumerate().for_each(|(i, mut row)| {
        coords.par_iter_mut().enumerate().for_each(|(i, row)| {
            for (k, coord) in row.iter_mut().enumerate() {
                let totalF_i = magnitude(&forces[i]);
                let speed_i = args.ks * global_speed / (1.0 + global_speed * swing[i].sqrt());
                *coord += forces[(i, k)] * speed_i.min(args.ksmax / totalF_i);
            }
        });

        for i in 0..n {
            for k in 0..dim {
                forces_prev[(i, k)] = forces[(i, k)];
                forces[(i, k)] = 0.0;
            }
        }
    }
}
