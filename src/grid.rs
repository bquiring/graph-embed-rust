use rayon::{
    iter::IntoParallelIterator,
    slice::{ChunksMut, ParallelSliceMut},
};
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
pub struct Grid<T> {
    nrows: usize,
    ncols: usize,
    inner: Vec<T>,
}

impl<T: Copy + Default> Grid<T> {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            inner: vec![T::default(); nrows * ncols],
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

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.inner.iter()
    }

    pub fn row_iter(&self) -> impl Iterator<Item = &[T]> {
        self.inner.chunks(self.ncols)
    }
}

impl<T> Index<usize> for Grid<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.ncols;
        &self.inner[start..start + self.ncols]
    }
}

impl<T> IndexMut<usize> for Grid<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let start = index * self.ncols;
        &mut self.inner[start..start + self.ncols]
    }
}

impl<T> Index<(usize, usize)> for Grid<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.inner[index.1 * self.ncols + index.0]
    }
}

impl<T> IndexMut<(usize, usize)> for Grid<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.inner[index.1 * self.ncols + index.0]
    }
}

impl<'a, T: Send> IntoParallelIterator for &'a mut Grid<T> {
    type Item = &'a mut [T];
    type Iter = ChunksMut<'a, T>;

    fn into_par_iter(self) -> Self::Iter {
        self.inner.par_chunks_mut(self.ncols)
    }
}
