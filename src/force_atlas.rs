use crate::grid::Grid;
use crate::util::*;
//use nalgebra::base::DMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use rayon::prelude::*;
use rand::{distributions::Uniform, Rng};

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
    A: &CsrMatrix<f64>,
    dim: usize,
    iter: usize,
    // coords: &mut Grid<f64>,
    coords: &mut Vec<Vec<f64>>,
    args: &ForceAtlasArgs,
) {
    let epsilon = 0.00001;
    // let n = coords.nrows();
    let n = coords.len();

    assert_eq!(A.nrows(), A.ncols());
    assert_eq!(A.nrows(), n);
    // assert_eq!(coords.ncols(), dim);

    let I = A.row_offsets();
    let J = A.col_indices();
    let D = A.values();

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

    /*
    let mut forces_prev = Grid::new(n, dim);
    let mut forces = Grid::new(n, dim);
    let mut swing = vec![0.0; n];
    */
    let mut forces_prev = vec![vec![0.0; dim]; n];
    let mut forces = vec![vec![0.0; dim]; n];
    let mut swing = vec![0.0; n];

    for _ in 0..iter {
        // TODO: parallelize here... no idea if this is right
        //forces.par_iter_mut().enumerate().for_each(|(i, row)| {
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
                    let direction = (coords[j][k] - coords[i][k]) / dis_ij;
                    let Fa_sum = direction * Fa_ij;
                    force_i[k] += Fa_sum;
                }
            }

            let mag = magnitude(&coords[i]);
            //for (k, force) in row.iter_mut().enumerate() {
            for k in 0..dim {
                let Fg_ki = -coords[i][k] / mag * args.gravity * deg[i];
                //*force = force_i[k] + Fg_ki;
                forces[i][k] = force_i[k] + Fg_ki;
            }
        };

        // TODO: parallize here
        /*
        swing
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = distance(&forces[i], &forces_prev[i]));
        */
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
        //}[
        let global_traction = 1.0;
        let global_speed = args.tolerate * global_traction / global_swing;

        // TODO: parallelize here
        //        coords.row_iter_mut().enumerate().for_each(|(i, mut row)| {
        /*
        coords.par_iter_mut().enumerate().for_each(|(i, row)| {
            for (k, coord) in row.iter_mut().enumerate() {
                let totalF_i = magnitude(&forces[i]);
                let speed_i = args.ks * global_speed / (1.0 + global_speed * swing[i].sqrt());
                *coord += forces[(i, k)] * speed_i.min(args.ksmax / totalF_i);
            }
        });
         */
        for i in 0..n {
            let totalF_i = magnitude(&forces[i]);
            let speed_i = args.ks * global_speed / (1.0 + global_speed * swing[i].sqrt());
            for k in 0..dim {
                coords[i][k] += forces[i][k] * speed_i.min(args.ksmax / totalF_i);
            }
        }

        for i in 0..n {
            for k in 0..dim {
                forces_prev[i][k] = forces[i][k];
                forces[i][k] = 0.0;
            }
        }
    }
}



pub fn force_atlas_multilevel (
    A: &CsrMatrix<f64>,
    dim: usize,
    iter: usize,
    //coords: &mut Grid<f64>,
    //coords_Ac: &mut Grid<f64>,
    coords: &mut Vec<Vec<f64>>,
    coords_Ac: &Vec<Vec<f64>>,
    PT: &CsrMatrix<f64>,
    args: &ForceAtlasArgs,
) {
    let epsilon = 0.00001;
    // let n = coords.nrows();
    let n = coords.len();
    let m = PT.nrows();

    assert_eq!(A.nrows(), A.ncols());
    assert_eq!(A.nrows(), n);

    // assert_eq!(coords.ncols(), dim);
    // TODO: more checks

    let I = A.row_offsets();
    let J = A.col_indices();
    let D = A.values();

    let PT_I = PT.row_offsets();
    let PT_J = PT.col_indices();

    let mut commOf = vec![0; n];
    for a in 0..m {
        for c in PT_I[a]..PT_I[a+1] {
            commOf[PT_J[c]] = a
        }
    }

    let mut global_to_local = vec![0; n];
    for a in 0..m {
        let mut count = 0;
        for c in PT_I[a]..PT_I[a+1] {
            global_to_local[PT_J[c]] = count;
            count += 1;
        }
    }

    for a in 0..m {
        let n = PT_I[a+1] - PT_I[a];
        let mut local_to_global = vec![0; n];
        for c in PT_I[a]..PT_I[a+1] {
            local_to_global[c - PT_I[a]] = PT_J[c];
        }

        let mut coords_loc = vec![vec![0.0; dim]; n]; //Grid::new(n, dim);

        let mut rng = rand::thread_rng();
        let dist = Uniform::from(-1.0..1.0);
        for i in 0..n {
            for k in 0..dim {
                coords_loc[i][k] = rng.sample(&dist);
            }
        }

        
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

        let mut forces_prev = vec![vec![0.0; dim]; n]; // Grid::new(n, dim);
        let mut forces = vec![vec![0.0; dim]; n]; // Grid::new(n, dim);
        let mut swing = vec![0.0; n];

        for _ in 0..iter {
            // TODO: parallelize here... no idea if this is right
            for i in 0..n {
                let mut force_i = vec![0.0; dim];
                for j in 0..n {
                    if i != j {
                        let dis_ij = distance(&coords_loc[i], &coords_loc[j]).max(epsilon);
                        let Fr_ij = deg[i] * deg[j] * args.repel / (dis_ij * dis_ij);

                        for k in 0..dim {
                            let direction = -(coords_loc[j][k] - coords_loc[i][k]) / dis_ij;
                            let Fr_sum = direction * Fr_ij;
                            force_i[k] += Fr_sum;
                        }
                    }
                }

                let mag = magnitude(&coords_loc[i]);
                for k2 in I[local_to_global[i]]..I[local_to_global[i] + 1] {
                    let j = J[k2];
                    if a == commOf[j] {
                        let j = global_to_local[j];
                        let dis_ij = distance(&coords_loc[i], &coords_loc[j]).max(epsilon);
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
                            let direction = (coords_loc[j][k] - coords_loc[i][k]) / dis_ij;
                            let Fa_sum = direction * Fa_ij;
                            force_i[k] += Fa_sum;
                        }
                    } else {
                        let pull = 100.0;
	                let dis_ij = distance (&coords_Ac[a], &coords_Ac[commOf[j]]).max(epsilon);
	                let Fo_ij = pull;
	                
	                for k in 0..dim {
		            let direction = (coords_Ac[commOf[j]][k] - coords_Ac[a][k]) / dis_ij;
		            let Fo_sum = direction * Fo_ij / mag;
		            force_i[k] += Fo_sum;
	                }
                        
                    }
                }

                for k in 0..dim {
                    let Fg_ki = -coords_loc[i][k] / mag * args.gravity * deg[i];
                    forces[i][k] = force_i[k] + Fg_ki;
                }
            };

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
            //}[
            let global_traction = 1.0;
            let global_speed = args.tolerate * global_traction / global_swing;

            // TODO: parallelize here
            for i in 0..n {
                let totalF_i = magnitude(&forces[i]);
                let speed_i = args.ks * global_speed / (1.0 + global_speed * swing[i].sqrt());
                let speed_i = speed_i.min(args.ksmax / totalF_i);
                for k in 0..dim {
                    coords_loc[i][k] += forces[i][k] * speed_i;
                }
            }

            for i in 0..n {
                for k in 0..dim {
                    forces_prev[i][k] = forces[i][k];
                    forces[i][k] = 0.0;
                }
            }
        }
        for i in 0..n {
            for k in 0..dim {
                coords[local_to_global[i]][k] = coords_loc[i][k];
            }
        }
    }
}
