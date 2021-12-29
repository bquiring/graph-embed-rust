
pub fn distance (v1 : &[f32], v2 : &[f32]) -> f32 {
    assert! (v1.len() == v2.len());
    let mut sum = 0.0;
    for i in 0..v1.len() {
        let d = v2[i] - v1[i];
        sum += d * d;
    }
    sum.sqrt()
}

pub fn magnitude (v : &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..v.len() {
        let d = v[i];
        sum += d * d;
    }
    sum.sqrt()
}


fn normalize (dim : usize,
              coords : &mut Vec<Vec<f32>>) {
    let n = coords.len();

    for i in 0..n {
        assert! (coords[i].len() == dim);
    }

    // center coords at 0
    let mut avg = vec! [0.0; dim];
    for i in 0..n {
        for k in 0..dim {
            avg[k] += coords[i][k];
        }
    }
    for k in 0..dim {
        avg[k] /= n as f32;
    }

    for i in 0..n {
        for k in 0..dim {
            coords[i][k] -= avg[k]
        }
    }

    // scale coords to be within the unit circle
    let mut max_length = 0.0;
    for i in 0..n {
        let length = magnitude (&coords[i]);
        if max_length < length {
            max_length = length;
        }
    }

    for i in 0..n {
        for k in 0..dim {
            coords[i][k] = coords[i][k] / max_length;
        }
    }
}


