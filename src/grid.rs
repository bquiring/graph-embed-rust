use rayon::iter::*;
use rayon::slice::{Chunks, ChunksMut, ParallelSlice, ParallelSliceMut};
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
pub struct Grid {
    nrows: usize,
    ncols: usize,
    inner: Vec<f64>,
}

impl Grid {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            inner: vec![0.0; nrows * ncols],
            nrows,
            ncols,
        }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.inner.iter()
    }

    pub fn row_iter(&self, index: usize) -> impl Iterator<Item = &f64> {
        self[index].iter()
    }
}

impl Index<usize> for Grid {
    type Output = [f64];

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.ncols;
        &self.inner[start..start + self.ncols]
    }
}

impl IndexMut<usize> for Grid {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let start = index * self.ncols;
        &mut self.inner[start..start + self.ncols]
    }
}

impl Index<(usize, usize)> for Grid {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.inner[index.0 * self.ncols + index.1]
    }
}

impl IndexMut<(usize, usize)> for Grid {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.inner[index.0 * self.ncols + index.1]
    }
}

impl<'a> IntoParallelIterator for &'a mut Grid {
    type Item = &'a mut [f64];
    type Iter = ChunksMut<'a, f64>;

    fn into_par_iter(self) -> Self::Iter {
        self.inner.par_chunks_mut(self.ncols)
    }
}
