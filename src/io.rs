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

    pub fn to_sym_coo(&self) -> CooMatrix<f32> {
        assert_eq!(self.nrows, self.ncols);
        assert_eq!(self.row_indices.len(), self.col_indices.len());
        assert_eq!(self.row_indices.len(), self.values.len());
        let mut coo = CooMatrix::new(self.nrows + 1, self.ncols + 1);
        for i in 0..self.nrows {
            coo.push(self.row_indices[i], self.col_indices[i], self.values[i]);
            coo.push(self.col_indices[i], self.row_indices[i], self.values[i]);
        }
        coo
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
