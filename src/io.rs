use nalgebra_sparse::coo::CooMatrix;
use std::{
    fs::File,
    io::{self, prelude::*, BufReader},
    path::Path,
};

pub struct MatrixMarket {
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
    nrows: usize,
    ncols: usize,
}

#[derive(Debug)]
pub enum MmError {
    Parse(ParseError),
    Io(io::Error),
}

#[derive(Debug)]
pub enum ParseError {
    ExpectedRow,
    ExpectedCol,
    ExpectedValue,
}

impl From<io::Error> for MmError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ParseError> for MmError {
    fn from(e: ParseError) -> Self {
        Self::Parse(e)
    }
}

impl MatrixMarket {
    // TODO: error on bad input...
    pub fn read(path: &Path) -> Result<Self, MmError> {
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
            let row: usize = split
                .next()
                .ok_or(ParseError::ExpectedRow)?
                .parse()
                .unwrap();
            let col: usize = split
                .next()
                .ok_or(ParseError::ExpectedCol)?
                .parse()
                .unwrap();
            let value: f64 = split
                .next()
                .ok_or(ParseError::ExpectedValue)?
                .parse()
                .unwrap();
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

    pub fn to_sym_coo(&self) -> CooMatrix<f64> {
        assert_eq!(self.row_indices.len(), self.col_indices.len());
        assert_eq!(self.row_indices.len(), self.values.len());
        let nvertices = self.nrows.max(self.ncols);
        let mut coo = CooMatrix::new(nvertices + 1, nvertices + 1);
        for i in 0..self.row_indices.len() {
            coo.push(self.row_indices[i], self.col_indices[i], self.values[i]);
            coo.push(self.col_indices[i], self.row_indices[i], self.values[i]);
        }
        coo
    }
}
