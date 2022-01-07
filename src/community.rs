use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use rand::seq::SliceRandom;
use std::{
    collections::{hash_map::Entry, HashMap},
    ops::Range,
};

// Uses the Louvain method, https://arxiv.org/abs/0803.0476
// Based on code from https://sites.google.com/site/findcommunities/

#[derive(Clone, Debug)]
pub struct Level {
    node_to_comm: HashMap<usize, usize>,
    comm_sizes: HashMap<usize, usize>,
}

impl Level {
    fn new(node_to_comm: HashMap<usize, usize>) -> Self {
        let mut comm_sizes = HashMap::new();
        if let Some(&max) = node_to_comm.values().max() {
            for comm in 0..max {
                let size = node_to_comm.values().filter(|&v| *v == comm).count();
                comm_sizes.insert(comm, size);
            }
        }
        Self {
            node_to_comm,
            comm_sizes,
        }
    }

    pub fn comm_of(&self, node: usize) -> Option<usize> {
        self.node_to_comm.get(&node).copied()
    }

    pub fn comm_size(&self, comm: usize) -> Option<usize> {
        self.comm_sizes.get(&comm).copied()
    }

    pub fn num_comm(&self) -> usize {
        self.comm_sizes.len()
    }

    pub fn nodes(&self, comm: usize) -> Vec<usize> {
        match self.comm_sizes.get(&comm) {
            None | Some(&0) => Vec::new(),
            Some(&size) => {
                let mut nodes = Vec::with_capacity(size);
                for (&node, &c) in &self.node_to_comm {
                    if c == comm {
                        nodes.push(node);
                    }
                }
                nodes.sort_unstable();
                nodes
            }
        }
    }

    pub fn sorted(&self) -> Vec<(usize, usize)> {
        let mut kv: Vec<_> = self.node_to_comm.iter().map(|(&k, &v)| (k, v)).collect();
        kv.sort_by(|k1, k2| k1.0.cmp(&k2.0));
        kv
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
    node_to_comm: HashMap<usize, usize>,
    inside: Vec<f64>,
    wdeg: Vec<f64>,
    neigh_weight: Vec<f64>,
    neigh_pos: Vec<usize>,
    neigh_last: usize,
}

impl Community {
    fn new(graph: Graph, min_mod: f64) -> Self {
        let size = graph.num_nodes();
        let mut node_to_comm = HashMap::with_capacity(size);
        let mut inside = Vec::with_capacity(size);
        let mut wdeg = Vec::with_capacity(size);
        for node in 0..size {
            node_to_comm.insert(node, node);
            inside.push(graph.num_self_loops(node) as f64);
            wdeg.push(graph.wdeg(node));
        }
        Self {
            graph,
            size,
            min_mod,
            node_to_comm,
            inside,
            wdeg,
            neigh_weight: vec![-1.0; size],
            neigh_pos: vec![0; size],
            neigh_last: 0,
        }
    }

    fn remove(&mut self, node: usize, comm: usize, dnode_comm: f64) {
        assert!(node < self.size);
        self.wdeg[comm] -= self.graph.wdeg(node);
        self.inside[comm] -= 2.0 * dnode_comm + self.graph.num_self_loops(node) as f64;
        self.node_to_comm.remove(&node);
    }

    fn insert(&mut self, node: usize, comm: usize, dnode_comm: f64) {
        assert!(node < self.size);
        self.wdeg[comm] += self.graph.wdeg(node);
        self.inside[comm] += 2.0 * dnode_comm + self.graph.num_self_loops(node) as f64;
        self.node_to_comm.insert(node, comm);
    }

    fn modularity(&self) -> f64 {
        let mut q = 0.0;
        let t = self.graph.weight_sum();
        for (node, wdeg) in self.wdeg.iter().enumerate().filter(|(_, &w)| w > 0.0) {
            let x = wdeg / t;
            q += self.inside[node] / t - x * x;
        }
        q
    }

    #[inline]
    fn modularity_gain(&self, node: usize, comm: usize, dnode_comm: f64, wdeg: f64) -> f64 {
        assert!(node < self.size);
        dnode_comm - self.wdeg[comm] * wdeg / self.graph.weight_sum()
    }

    fn neigh_comm(&mut self, node: usize) {
        for &pos in &self.neigh_pos[..self.neigh_last] {
            self.neigh_weight[pos] = -1.0;
        }

        self.neigh_pos[0] = self.node_to_comm[&node];
        self.neigh_weight.insert(self.neigh_pos[0], 0.0);
        self.neigh_last = 1;

        for &neigh in self.graph.neighbors(node) {
            let neigh_comm = self.node_to_comm[&neigh];
            let neigh_weight = self.wdeg[neigh];

            if neigh != node {
                if (self.neigh_weight[neigh_comm] - -1.0).abs() < f64::EPSILON {
                    self.neigh_weight[neigh_comm] = 0.0;
                    self.neigh_pos[self.neigh_last] = neigh_comm;
                    self.neigh_last += 1;
                }
                self.neigh_weight[neigh_comm] += neigh_weight;
            }
        }
    }

    fn next_level(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let mut rand_order: Vec<_> = (0..self.size).collect();
        rand_order.shuffle(&mut rng);

        let mut improved = false;
        let mut new_mod = self.modularity();
        loop {
            let cur_mod = new_mod;
            let mut num_moves = 0;

            for &node in &rand_order {
                let comm = self.node_to_comm[&node];
                let wdeg = self.wdeg[node];

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

    // has to be a better way ... this is pretty sloppy
    fn partition_to_graph(&mut self) -> Graph {
        let mut renumber = HashMap::with_capacity(self.size);
        for node in 0..self.size {
            renumber
                .entry(self.node_to_comm[&node])
                .and_modify(|e| *e += 1)
                .or_insert(0);
        }

        let mut fin = 0;
        for comm in 0..self.size {
            if let Entry::Occupied(mut e) = renumber.entry(comm) {
                e.insert(fin);
                fin += 1;
            }
        }

        let mut comm_nodes = vec![Vec::new(); fin];
        for node in 0..self.size {
            comm_nodes[renumber[&self.node_to_comm[&node]]].push(node);
        }

        let mut degs = Vec::with_capacity(comm_nodes.len());
        let mut links = Vec::with_capacity(comm_nodes.len());
        for nodes in comm_nodes.iter() {
            let mut comm_links = HashMap::new();
            for &node in nodes.iter() {
                let neighbors = self.graph.neighbors(node);
                let neighbor_weights = self.graph.neighbor_weights(node);
                for (neigh, weight) in neighbors.iter().zip(neighbor_weights.iter()) {
                    let neigh_comm = renumber[&self.node_to_comm[neigh]];
                    *comm_links.entry(neigh_comm).or_insert(0.0) += weight;
                }
            }
            degs.push(comm_links.len() + degs.last().unwrap_or(&0));
            links.push(comm_links);
        }

        for node in 0..comm_nodes.len() {
            self.node_to_comm
                .insert(node, renumber[&self.node_to_comm[&node]]);
        }

        let mut coo = CooMatrix::new(links.len(), links.len());
        for (node, neighbors) in links.iter().enumerate() {
            coo.push(node, node, degs[node] as f64);
            for (&neigh, &neigh_weight) in neighbors {
                coo.push(node, neigh, neigh_weight);
                coo.push(neigh, node, neigh_weight);
            }
        }
        Graph::from_csr(CsrMatrix::from(&coo))
    }
}

#[derive(Clone, Debug)]
struct Graph {
    inner: CsrMatrix<f64>,
}

impl Graph {
    fn from_csr(m: CsrMatrix<f64>) -> Self {
        assert_eq!(m.nrows(), m.ncols());
        Self { inner: m }
    }

    fn neighbors(&self, node: usize) -> &[usize] {
        &self.inner.col_indices()[self.neighbor_range(node)]
    }

    fn neighbor_weights(&self, node: usize) -> &[f64] {
        &self.inner.values()[self.neighbor_range(node)]
    }

    fn neighbor_range(&self, node: usize) -> Range<usize> {
        self.inner.row_offsets()[node]..self.inner.row_offsets()[node + 1]
    }

    fn wdeg(&self, node: usize) -> f64 {
        let sum: f64 = self.neighbor_weights(node).iter().sum();
        sum + 1.0
    }

    fn num_self_loops(&self, node: usize) -> usize {
        self.neighbors(node)
            .iter()
            .filter(|&neigh| *neigh == node)
            .count()
    }

    fn num_nodes(&self) -> usize {
        self.inner.nrows()
    }

    fn weight_sum(&self) -> f64 {
        (0..self.num_nodes()).map(|node| self.wdeg(node)).sum()
    }
}
