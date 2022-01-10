use crate::grid::Grid;
use crate::util::*;
use nalgebra_sparse::csr::CsrMatrix;
use rand::{distributions::Uniform, Rng};

// the algorithm

pub fn computeRadii(
    A: &CsrMatrix<f64>,
    //coords : &Grid<f64>,
    coords: &Vec<Vec<f64>>,
    radii: &mut Vec<f64>,
    doAll: bool,
) {
    let n = A.nrows();
    assert!(n == radii.len());
    for i in 0..n {
        radii[i] = -1.0;
    }

    let mut times = Vec::new();
    let A_I = A.row_offsets();
    let A_J = A.col_indices();
    let A_D = A.values();

    // TODO: factor in weights!
    let mut rate = vec![0.0; n];
    for i in 0..n {
        let mut found = false;
        for edge in A_I[i]..A_I[i + 1] {
            let j = A_J[edge];
            if i == j {
                rate[i] = A_D[edge];
                found = true;
                break;
            }
        }
        if !found {
            rate[i] = 1.0;
        }
    }

    for i in 0..n {
        if doAll {
            for j in i + 1..n {
                let distance_ij = distance(&coords[i], &coords[j]);
                times.push((distance_ij / (rate[i] + rate[j]), i, j));
            }
        } else {
            for edge in A_I[i]..A_I[i + 1] {
                let j = A_J[edge];
                if i < j {
                    let distance_ij = distance(&coords[i], &coords[j]);
                    times.push((distance_ij / (rate[i] + rate[j]), i, j));
                }
            }
        }
    }
    // FIXME: sort `times` based on the first component
    times.sort_by(|(t1, _, _), (t2, _, _)| t2.partial_cmp(t1).unwrap());

    let mut assignedCount = 0;
    while assignedCount < n && !times.is_empty() {
        let (time_ij, i, j) = times.pop().unwrap();
        if radii[i] <= 0.0 && radii[j] > 0.0 {
            // only i is live
            let distance_ij = time_ij * rate[i];
            radii[i] = distance_ij;
            for r in 0..times.len() {
                let (time_ij_prime, i_prime, j_prime) = times[r];
                if i_prime == i || j_prime == i {
                    let (same, other) = if i_prime == i {
                        (i_prime, j_prime)
                    } else {
                        (j_prime, i_prime)
                    };
                    // this is not the case if both are dead, but in that case we don't care
                    let distance_ij_prime = time_ij_prime * (rate[other] + rate[same]);
                    let distance_part = distance_ij * rate[same] / (rate[i] + rate[j]);
                    times[r].0 = (distance_ij_prime - distance_part) / rate[other];
                }
            }
            times.sort_by(|(t1, _, _), (t2, _, _)| t2.partial_cmp(t1).unwrap());
            assignedCount += 1;
        } else if radii[i] > 0.0 && radii[j] <= 0.0 {
            // only j is live
            let distance_ij = time_ij * rate[j];
            radii[j] = distance_ij;
            for r in 0..times.len() {
                let (time_ij_prime, i_prime, j_prime) = times[r];
                if i_prime == j || j_prime == j {
                    let (same, other) = if i_prime == j {
                        (i_prime, j_prime)
                    } else {
                        (j_prime, i_prime)
                    };
                    // this is not the case if both are dead, but in that case we don't care
                    let distance_ij_prime = time_ij_prime * (rate[other] + rate[same]);
                    let distance_part = distance_ij * rate[same] / (rate[i] + rate[j]);
                    times[r].0 = (distance_ij_prime - distance_part) / rate[other];
                }
            }
            times.sort_by(|(t1, _, _), (t2, _, _)| t2.partial_cmp(t1).unwrap());
            assignedCount += 1;
        } else if radii[i] <= 0.0 && radii[j] <= 0.0 {
            // both are live
            let distance_ij = time_ij * (rate[i] + rate[j]);
            radii[i] = distance_ij * rate[i] / (rate[i] + rate[j]);
            radii[j] = distance_ij * rate[j] / (rate[i] + rate[j]);
            for r in 0..times.len() {
                let (time_ij_prime, i_prime, j_prime) = times[r];
                if i_prime == i || j_prime == i || i_prime == j || j_prime == j {
                    let (same, other) = if i_prime == i || i_prime == j {
                        (i_prime, j_prime)
                    } else {
                        (j_prime, i_prime)
                    };
                    // this is not the case if both are dead, but in that case we don't care
                    let distance_ij_prime = time_ij_prime * (rate[other] + rate[same]);
                    let distance_part = distance_ij * rate[same] / (rate[i] + rate[j]);
                    times[r].0 = (distance_ij_prime - distance_part) / rate[other];
                }
            }
            times.sort_by(|(t1, _, _), (t2, _, _)| t2.partial_cmp(t1).unwrap());
            assignedCount += 1;
            assignedCount += 1;
        } else { // both are dead
        }
    }
}

pub fn normalizeCommunityEmbeddings(
    //coords_A : &mut Grid<f64>,
    //coords_Ac : &Grid<f64>,
    coords_A: &mut Vec<Vec<f64>>,
    coords_Ac: &Vec<Vec<f64>>,
    radii_Ac: &[f64],
    PT: &CsrMatrix<f64>,
) {
    //let n = coords_A.nrows();
    //let nc = coords_Ac.nrows();
    //let dim = coords_A.ncols();
    let n = coords_A.len();
    let nc = coords_Ac.len();
    let dim = coords_A[0].len();
    assert!(nc == radii_Ac.len());
    assert!(nc == PT.nrows());
    assert!(n == PT.ncols());
    //assert! (dim == coords_Ac.ncols());

    let PT_I = PT.row_offsets();
    let PT_J = PT.col_indices();

    let epsilon = 0.00001;

    for a in 0..nc {
        let mut max: f64 = 0.0;
        for commIndex in PT_I[a]..PT_I[a + 1] {
            let i = PT_J[commIndex];
            max = max.max(magnitude(&coords_A[i])).max(epsilon);
        }
        for commIndex in PT_I[a]..PT_I[a + 1] {
            let i = PT_J[commIndex];
            for k in 0..dim {
                coords_A[i][k] = coords_Ac[a][k] + (radii_Ac[a] / max) * coords_A[i][k];
            }
        }
    }
}

pub fn embedMultilevel<F, G>(
    As: &Vec<CsrMatrix<f64>>,
    PTs: &Vec<CsrMatrix<f64>>,
    dim: usize,
    baseEmbedder: F,
    localEmbedder: G,
) -> Vec<Vec<f64>>
where
    F: Fn(&CsrMatrix<f64>, usize, &mut Vec<Vec<f64>>) -> (),
    G: Fn(&CsrMatrix<f64>, usize, &mut Vec<Vec<f64>>, &Vec<Vec<f64>>, &CsrMatrix<f64>) -> (),
{
    let mut radii_Ac: Vec<f64> = vec![0.0; 0];

    let n0 = As[As.len() - 1].nrows();
    let mut rng = rand::thread_rng();
    let dist = Uniform::from(-1.0..1.0);
    let mut rand_elems = Vec::with_capacity(n0 * dim);
    rand_elems.extend((0..n0 * dim).map(|_| rng.sample(&dist)));
    //let mut coords_A = Grid::from_vec(n0, dim, rand_elems);
    let mut coords_A: Vec<Vec<f64>> = vec![vec![rng.sample(&dist); dim]; n0];
    for i in 0..n0 {
        for k in 0..dim {
            coords_A[i][k] = rng.sample(&dist);
        }
    }

    let mut rand_elems = Vec::with_capacity(1);
    rand_elems.extend((0..1 * 1).map(|_| rng.sample(&dist)));
    // let mut coords_Ac: Grid<f64> = Grid::from_vec(1, 1, rand_elems);
    let mut coords_Ac: Vec<Vec<f64>> = vec![vec![rng.sample(&dist); 1]; 1];
    for i in 0..1 {
        for k in 0..1 {
            coords_Ac[i][k] = rng.sample(&dist);
        }
    }

    for level in (0..As.len()).rev() {
        let A = &As[level];
        if level == As.len() - 1 {
            baseEmbedder(&A, dim, &mut coords_A);
        } else {
            let PT = &PTs[level];
            // INVARIANT:
            // - radii have been computed
            // - coordinates for Ac have been computed

            // embed all the communities in their own space
            localEmbedder(A, dim, &mut coords_A, &coords_Ac, &PT);
            // embed the local embeddings into the global embedding space
            normalizeCommunityEmbeddings(&mut coords_A, &coords_Ac, &radii_Ac, &PT);
        }

        if level != 0 {
            // shift coords_A into coords_Ac
            // setup coords_A for the next level
            coords_Ac = coords_A;
            // coords_A = Grid::new(As[level-1].nrows(), dim);
            coords_A = vec![vec![0.0; dim]; As[level - 1].nrows()];

            radii_Ac = vec![0.0; A.nrows()];
            // compute the radii for A (soon to be Ac)
            if level == As.len() - 1 {
                computeRadii(&A, &coords_Ac, &mut radii_Ac, true)
            } else {
                computeRadii(&A, &coords_Ac, &mut radii_Ac, false)
            }
        }
    }
    coords_A
}
