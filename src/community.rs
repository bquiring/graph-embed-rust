use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use rand::seq::SliceRandom;
use std::{collections::HashMap, ops::Range};

// Uses the Louvain method, https://arxiv.org/abs/0803.0476
// Based on code from https://sites.google.com/site/findcommunities/

#[derive(Clone, Debug)]
pub struct Level {
    node_to_comm: Vec<usize>,
    comm_sizes: Vec<usize>,
}

impl Level {
    fn new(node_to_comm: Vec<Option<usize>>) -> Self {
        let node_to_comm: Vec<_> = node_to_comm.iter().flatten().copied().collect();
        let comm_sizes = if let Some(&max) = node_to_comm.iter().max() {
            let num_comms = max + 1;
            let mut comm_sizes = vec![0; num_comms];
            for comm in 0..num_comms {
                let size = node_to_comm.iter().filter(|&c| *c == comm).count();
                comm_sizes[comm] += size;
            }
            comm_sizes
        } else {
            Vec::new()
        };
        Self {
            node_to_comm,
            comm_sizes,
        }
    }

    pub fn comm_of(&self, node: usize) -> Option<usize> {
        self.node_to_comm.get(node).copied()
    }

    pub fn comm_size(&self, comm: usize) -> Option<usize> {
        self.comm_sizes.get(comm).copied()
    }

    pub fn num_comm(&self) -> usize {
        self.comm_sizes.len()
    }

    pub fn num_vert(&self) -> usize {
        self.node_to_comm.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.node_to_comm.iter().enumerate().map(|(n, &c)| (n, c))
    }
}

pub fn louvain(m: &CsrMatrix<f64>, min_mod: f64) -> Vec<Level> {
    let graph = Graph::from_csr(m.clone());
    let mut c = Community::new(graph, min_mod);
    let mut levels = Vec::new();
    let mut improved;
    loop {
        improved = c.next_level();
        let graph = c.partition_to_graph();
        levels.push(Level::new(c.node_to_comm));
        c = Community::new(graph, min_mod);

        if !improved {
            break;
        }
    }
    levels
}

#[derive(Clone, Debug)]
struct Community {
    graph: Graph,
    size: usize,
    min_mod: f64,
    node_to_comm: Vec<Option<usize>>,
    inside: Vec<f64>,
    total: Vec<f64>,
    neigh_weight: Vec<f64>,
    neigh_pos: Vec<usize>,
    neigh_last: usize,
}

impl Community {
    fn new(graph: Graph, min_mod: f64) -> Self {
        let size = graph.num_nodes();
        let mut node_to_comm = Vec::with_capacity(size);
        let mut inside = Vec::with_capacity(size);
        let mut total = Vec::with_capacity(size);
        for node in 0..size {
            node_to_comm.push(Some(node));
            inside.push(graph.loops(node));
            total.push(graph.wdeg(node));
        }
        Self {
            graph,
            size,
            min_mod,
            node_to_comm,
            inside,
            total,
            neigh_weight: vec![-1.0; size],
            neigh_pos: vec![0; size],
            neigh_last: 0,
        }
    }

    fn remove(&mut self, node: usize, comm: usize, dnode_comm: f64) {
        assert!(node < self.size);
        self.total[comm] -= self.graph.wdeg(node);
        self.inside[comm] -= 2.0 * dnode_comm + self.graph.loops(node) as f64;
        self.node_to_comm[node] = None;
    }

    fn insert(&mut self, node: usize, comm: usize, dnode_comm: f64) {
        assert!(node < self.size);
        self.total[comm] += self.graph.wdeg(node);
        self.inside[comm] += 2.0 * dnode_comm + self.graph.loops(node) as f64;
        self.node_to_comm[node] = Some(comm);
    }

    fn modularity(&self) -> f64 {
        let mut q = 0.0;
        let t = self.graph.weight_sum();
        for (node, wdeg) in self.total.iter().enumerate().filter(|(_, &w)| w > 0.0) {
            let x = wdeg / t;
            q += self.inside[node] / t - x * x;
        }
        q
    }

    #[inline]
    fn modularity_gain(&self, node: usize, comm: usize, dnode_comm: f64, wdeg: f64) -> f64 {
        assert!(node < self.size);
        dnode_comm - self.total[comm] * wdeg / self.graph.weight_sum()
    }

    fn neigh_comm(&mut self, node: usize) {
        for &pos in &self.neigh_pos[..self.neigh_last] {
            self.neigh_weight[pos] = -1.0;
        }

        self.neigh_pos[0] = self.node_to_comm[node].unwrap();
        self.neigh_weight[self.neigh_pos[0]] = 0.0;
        self.neigh_last = 1;

        for &neigh in self.graph.neighbors(node) {
            if let Some(neigh_comm) = self.node_to_comm[neigh] {
                if neigh != node {
                    if (self.neigh_weight[neigh_comm] - -1.0).abs() < f64::EPSILON {
                        self.neigh_weight[neigh_comm] = 0.0;
                        self.neigh_pos[self.neigh_last] = neigh_comm;
                        self.neigh_last += 1;
                    }
                    self.neigh_weight[neigh_comm] += self.total[neigh];
                }
            }
        }
    }

    fn next_level(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let mut rand_order: Vec<_> = (0..self.size).collect();
        rand_order.shuffle(&mut rng);

        let mut improved = false;
        let mut new_mod = self.modularity();
        let mut iter = 0;
        // println!("next level");
        loop {
            // println!("loop iter = {}", iter);
            iter += 1;
            let cur_mod = new_mod;
            let mut num_moves = 0;

            for &node in &rand_order {
                let comm = self.node_to_comm[node].unwrap();
                let wdeg = self.graph.wdeg(node);

                self.neigh_comm(node);
                self.remove(node, comm, self.neigh_weight[comm]);

                let mut best_comm = comm;
                let mut best_links = 0.0;
                let mut best_incr = 0.0;
                for &pos in &self.neigh_pos[..self.neigh_last] {
                    let weight = self.neigh_weight[pos];
                    let incr = self.modularity_gain(node, pos, weight, wdeg);
                    if incr > best_incr {
                        best_comm = pos;
                        best_links = weight;
                        best_incr = incr;
                    }
                }

                self.insert(node, best_comm, best_links);

                if best_comm != comm {
                    num_moves += 1;
                }
            }

            new_mod = self.modularity();
            if num_moves > 0 {
                improved = true;
            }

            if num_moves == 0 || new_mod - cur_mod <= self.min_mod {
                break;
            }
        }
        improved
    }

    fn partition_to_graph(&mut self) -> Graph {
        let mut pop = vec![false; self.size];
        for node in 0..self.size {
            let comm = self.node_to_comm[node].unwrap();
            if let Some(false) = pop.get(comm) {
                pop[comm] = true;
            }
        }

        let mut fin = 0;
        let mut renumber = vec![None; self.size];
        for comm in 0..self.size {
            if let Some(true) = pop.get(comm) {
                renumber[comm] = Some(fin);
                fin += 1;
            }
        }

        let mut comm_nodes = vec![Vec::new(); fin];
        for node in 0..self.size {
            comm_nodes[renumber[self.node_to_comm[node].unwrap()].unwrap()].push(node);
        }

        let mut degs = Vec::with_capacity(comm_nodes.len());
        let mut links = Vec::with_capacity(comm_nodes.len());
        for nodes in comm_nodes.iter() {
            let mut comm_links = HashMap::new();
            for &node in nodes.iter() {
                let neighbors = self.graph.neighbors(node);
                let neighbor_weights = self.graph.neighbor_weights(node);
                for (&neigh, weight) in neighbors.iter().zip(neighbor_weights.iter()) {
                    let neigh_comm = renumber[self.node_to_comm[neigh].unwrap()];
                    *comm_links.entry(neigh_comm).or_insert(0.0) += weight;
                }
            }
            degs.push(comm_links.len() + degs.last().unwrap_or(&0));
            links.push(comm_links);
        }

        for node in 0..self.size {
            self.node_to_comm[node] = Some(renumber[self.node_to_comm[node].unwrap()].unwrap());
        }

        let mut coo = CooMatrix::new(links.len(), links.len());
        for (node, neighbors) in links.iter().enumerate() {
            coo.push(node, node, degs[node] as f64);
            for (&neigh, &neigh_weight) in neighbors {
                coo.push(node, neigh.unwrap(), neigh_weight);
                coo.push(neigh.unwrap(), node, neigh_weight);
            }
        }
        Graph::from_csr(CsrMatrix::from(&coo))
    }
}

#[derive(Clone, Debug)]
struct Graph {
    inner: CsrMatrix<f64>,
    wdeg: Vec<f64>,
    loops: Vec<f64>,
    wsum: f64,
}

impl Graph {
    fn from_csr(m: CsrMatrix<f64>) -> Self {
        assert_eq!(m.nrows(), m.ncols());
        let mut wdeg = Vec::with_capacity(m.nrows());
        let mut loops = vec![0.0; m.nrows()];
        let mut wsum = 0.0;
        for node in 0..m.nrows() {
            let start = m.row_offsets()[node];
            let end = m.row_offsets()[node + 1];
            let mut sum = 1.0;
            for (&nb, &w) in m.col_indices()[start..end]
                .iter()
                .zip(m.values()[start..end].iter())
            {
                sum += w;
                if nb == node {
                    loops[node] += w;
                }
            }
            wsum += sum;
            wdeg.push(sum);
        }
        Self {
            inner: m,
            wdeg,
            loops,
            wsum,
        }
    }

    fn neighbors(&self, node: usize) -> &[usize] {
        &self.inner.col_indices()[self.neighbor_range(node)]
    }

    fn neighbor_weights(&self, node: usize) -> &[f64] {
        &self.inner.values()[self.neighbor_range(node)]
    }

    #[inline]
    fn neighbor_range(&self, node: usize) -> Range<usize> {
        self.inner.row_offsets()[node]..self.inner.row_offsets()[node + 1]
    }

    #[inline]
    fn wdeg(&self, node: usize) -> f64 {
        self.wdeg[node]
    }

    #[inline]
    fn loops(&self, node: usize) -> f64 {
        self.loops[node]
    }

    #[inline]
    fn num_nodes(&self) -> usize {
        self.inner.nrows()
    }

    #[inline]
    fn weight_sum(&self) -> f64 {
        self.wsum
    }
}
