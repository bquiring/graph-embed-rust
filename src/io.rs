use nalgebra_sparse::coo::CooMatrix;
use std::{
    fs::File,
    io::{self, prelude::*, BufReader},
    path::Path,
};

pub struct MatrixMarket {
    nrows: usize,
    ncols: usize,
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
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
    pub fn read_from_path(path: &Path, zero_indexed : bool) -> Result<Self, MmError> {
        let f = File::open(path)?;
        parse(&mut BufReader::new(f), zero_indexed)
    }

    pub fn read_from_string(s: &str, zero_indexed : bool) -> Result<Self, MmError> {
        parse(&mut s.as_bytes(), zero_indexed)
    }

    pub fn to_sym_coo(&self) -> CooMatrix<f64> {
        assert_eq!(self.row_indices.len(), self.col_indices.len());
        assert_eq!(self.row_indices.len(), self.values.len());
        let nvertices = self.nrows.max(self.ncols);
        let mut coo = CooMatrix::new(nvertices, nvertices);
        for i in 0..self.row_indices.len() {
            coo.push(self.row_indices[i], self.col_indices[i], self.values[i]);
            coo.push(self.col_indices[i], self.row_indices[i], self.values[i]);
        }
        coo
    }
}

fn parse<R: BufRead>(reader: &mut R, zero_indexed : bool) -> Result<MatrixMarket, MmError> {
    let mut nrows = 0;
    let mut ncols = 0;
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    for line in reader.lines() {
        let line = line?;

        // skip header for now
        if line.starts_with('%') {
            continue;
        }

        // read data
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
        let (row, col) = if zero_indexed {(row, col)} else {(row-1, col-1)};

        row_indices.push(row);
        col_indices.push(col);
        values.push(value);
        nrows = nrows.max(row+1);
        ncols = ncols.max(col+1);
    }
    Ok(MatrixMarket {
        nrows,
        ncols,
        row_indices,
        col_indices,
        values,
    })
}
