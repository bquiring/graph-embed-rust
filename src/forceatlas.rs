use crate::util::*;
use nalgebra_sparse::csr::CsrMatrix;

pub struct ForceAtlasArgs {
    ks: f64,
    ksmax: f64,
    repel: f64,
    attract: f64,
    gravity: f64,
    delta: f64,
    tolerate: f64,
    useWeights: bool,
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
            delta: 1.0,
            tolerate: 1.0,
            useWeights: true,
            linlog: false,
            nohubs: false,
        }
    }
}

pub fn force_atlas(
    matrix: &CsrMatrix<f64>,
    dim: usize,
    coords: &mut Vec<Vec<f64>>,
    iterations: usize,
    args: ForceAtlasArgs,
) {
    let epsilon = 0.00001;

    let n = coords.len();
    assert!(matrix.nrows() == matrix.ncols());
    assert!(matrix.nrows() == n);

    for i in 0..n {
        assert!(coords[i].len() == dim);
    }

    let I = matrix.row_offsets();
    let J = matrix.col_indices();
    let D = matrix.values();

    let mut deg = vec![0.0; n];
    if args.useWeights {
        for i in 0..n {
            let sum : f64 = D[I[i]..I[i + 1]].iter().sum();
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

    for iter in 0..iterations {
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
                let a_ij = if args.useWeights { D[k2] } else { 1.0 };

                // if args.delta == 1.0 {
                //     fa_ij = fa_ij * a_ij;
                // } else if args.delta != 0.0 {
                //     fa_ij = (if a_ij < 0 {-1} else {1}) * a_ij.abs().pow (delta) * fa_ij;
                // }
                fa_ij *= a_ij;

                if args.nohubs {
                    fa_ij = fa_ij / deg[i];
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

        let mut globalSwing = 0.0;
        //for i in 0..n {
        //    globalSwing += ((deg[i] + 1.0) * swing[i]).max(epsilon);
        //}
        globalSwing = 1.0;

        let mut globalTraction = 0.0;
        //for i in 0..n {
        //    let traction_i = distance (&forces[i], &forces_prev[i]) / 2.0;
        //    globalTraction += (deg[i]+1.0) / traction_i;
        //}
        globalTraction = 1.0;

        let globalSpeed = args.tolerate * globalTraction / globalSwing;

        // TODO: parallelize here
        for i in 0..n {
            let totalF_i = magnitude(&forces[i]);
            let mut speed_i = args.ks * globalSpeed / (1.0 + globalSpeed * swing[i].sqrt());
            let speedConstraint_i = args.ksmax / totalF_i;
            if speed_i > speedConstraint_i {
                speed_i = speedConstraint_i;
            }

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
