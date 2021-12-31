pub fn distance(v1: &[f64], v2: &[f64]) -> f64 {
    assert_eq!(v1.len(), v2.len());
    let mut sum = 0.0;
    for (x, y) in v1.iter().zip(v2.iter()) {
        let d = y - x;
        sum += d * d;
    }
    sum.sqrt()
}

pub fn magnitude(v: &[f64]) -> f64 {
    let mut sum = 0.0;
    for x in v {
        sum += x * x;
    }
    sum.sqrt()
}
