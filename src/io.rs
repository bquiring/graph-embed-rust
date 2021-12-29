use nalgebra_sparse::coo::CooMatrix;
use std::{
    fs::File,
    io::{self, prelude::*, BufReader},
    path::Path,
};

pub struct MatrixMarket {
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f32>,
    nrows: usize,
    ncols: usize,
}

impl MatrixMarket {
    // TODO: error on bad input...
    pub fn read(path: &Path) -> io::Result<Self> {
        let f = File::open(path)?;
        let reader = BufReader::new(f);

        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let mut nrows = 0;
        let mut ncols = 0;
        for line in reader.lines() {
            let line = line?;

            // skip header for now
            if line.starts_with('%') {
                continue;
            }

            // read data
            // TODO: sanitize
            let mut split = line.split_whitespace();
            let row: usize = split.next().unwrap().parse().unwrap();
            let col: usize = split.next().unwrap().parse().unwrap();
            let value: f32 = split.next().unwrap().parse().unwrap();
            row_indices.push(row);
            col_indices.push(col);
            values.push(value);

            nrows = nrows.max(row);
            ncols = ncols.max(col);
        }
        Ok(Self {
            row_indices,
            col_indices,
            values,
            nrows,
            ncols,
        })
    }

    pub fn to_coo(&self) -> CooMatrix<f32> {
        CooMatrix::try_from_triplets(
            self.nrows + 1,
            self.ncols + 1,
            self.row_indices.clone(),
            self.col_indices.clone(),
            self.values.clone(),
        )
        .unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // not a real test
    #[test]
    fn ca_netscience() {
        let path = Path::new("res/ca-netscience/ca-netscience.mtx");
        let mm = MatrixMarket::read(&path).unwrap();
        let coo = mm.to_coo();
    }
}
