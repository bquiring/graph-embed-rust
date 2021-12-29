use crate::util::*;
use nalgebra_sparse::csr::CsrMatrix;

pub struct ForceAtlasArgs {
    ks: f64,
    ksmax: f64,
    repel: f64,
    attract: f64,
    gravity: f64,
    //delta: f64,
    tolerate: f64,
    use_weights: bool,
    linlog: bool,
    nohubs: bool,
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
            nohubs: false,
        }
    }
}

pub fn force_atlas(
    matrix: &CsrMatrix<f64>,
    dim: usize,
    iter: usize,
    coords: &mut Vec<Vec<f64>>,
    args: &ForceAtlasArgs,
) {
    let epsilon = 0.00001;
    let n = coords.len();

    assert_eq!(matrix.nrows(), matrix.ncols());
    assert_eq!(matrix.nrows(), n);
    for i in 0..n {
        assert_eq!(coords[i].len(), dim);
    }

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

    let mut forces_prev = vec![vec![0.0; dim]; n];
    let mut forces = vec![vec![0.0; dim]; n];
    let mut swing = vec![0.0; n];

    for _ in 0..iter {
        // TODO: parallelize here
        for i in 0..n {
            let mut force_i = vec![0.0; dim];
            for j in 0..n {
                if i != j {
                    let dis_ij = distance(&coords[i], &coords[j]).max(epsilon);
                    let Fr_ij = deg[i] * deg[j] * args.repel / (dis_ij * dis_ij);

                    for k in 0..dim {
                        let direction = -(coords[j][k] - coords[i][k]) / dis_ij;
                        let Fr_sum = direction * Fr_ij;
                        force_i[k] += Fr_sum;
                    }
                }
            }

            for k2 in I[i]..I[i + 1] {
                let j = J[k2];
                let dis_ij = distance(&coords[i], &coords[j]).max(epsilon);
                let mut fa_ij = if args.linlog { dis_ij.log2() } else { dis_ij };
                let a_ij = if args.use_weights { D[k2] } else { 1.0 };

                // if args.delta == 1.0 {
                //     fa_ij = fa_ij * a_ij;
                // } else if args.delta != 0.0 {
                //     fa_ij = (if a_ij < 0 {-1} else {1}) * a_ij.abs().pow (delta) * fa_ij;
                // }
                fa_ij *= a_ij;

                if args.nohubs {
                    fa_ij /= deg[i];
                }

                let Fa_ij = args.attract * fa_ij;
                for k in 0..dim {
                    let direction = (coords[j][k] - coords[i][k]) / dis_ij;
                    let Fa_sum = direction * Fa_ij;
                    force_i[k] += Fa_sum;
                }
            }

            let mag = magnitude(&coords[i]);
            for k in 0..dim {
                let Fg_ki = (-coords[i][k] / mag) * args.gravity * deg[i];
                forces[i][k] = force_i[k] + Fg_ki;
            }
        }

        // TODO: parallize here
        for i in 0..n {
            swing[i] = distance(&forces[i], &forces_prev[i]);
        }

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
        for i in 0..n {
            let totalF_i = magnitude(&forces[i]);
            let mut speed_i = args.ks * global_speed / (1.0 + global_speed * swing[i].sqrt());
            let speed_constraint_i = args.ksmax / totalF_i;
            speed_i = speed_i.min(speed_constraint_i);

            for k in 0..dim {
                let displacement_ik = forces[i][k] * speed_i;
                coords[i][k] += displacement_ik;
            }
        }
    }

    for i in 0..n {
        for k in 0..dim {
            forces_prev[i][k] = forces[i][k];
            forces[i][k] = 0.0;
        }
    }
}
