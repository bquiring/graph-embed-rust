#[inline]
pub fn distance(v1: &[f64], v2: &[f64]) -> f64 {
    assert_eq!(v1.len(), v2.len());
    let mut sum = 0.0;
    for i in 0..v1.len() {
    //for (x, y) in v1.iter().zip(v2.iter()) {
        let x = v1[i];
        let y = v2[i];
        let d = y - x;
        sum += d * d;
    }
    sum.sqrt()
}

#[inline]
pub fn magnitude(v: &[f64]) -> f64 {
    let mut sum = 0.0;
    for x in v {
        sum += x * x;
    }
    sum.sqrt()
}
