//! Deterministic fixtures shared by benches and tests.
//!
//! Deterministic LCG instead of `rand` to keep the dependency tree minimal
//! and every run exactly reproducible.

use rustfox_dom::{Dom, NodeId};

pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x2545F4914F6CDD1D) | 1)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        // Use the HIGH bits: low bits of an LCG have short periods
        // (bit k period <= 2^k), which would bias tag/class selection.
        (self.next_u64() >> 32) % n
    }
}

pub const TAGS: &[&str] = &["div", "span", "p", "a", "li", "ul", "section", "header"];
pub const CLASS_COUNT: u64 = 64;

/// Breadth-first synthetic tree with ~`target_nodes` nodes, depth `depth`.
pub fn build_dom(dom: &mut Dom, target_nodes: usize, depth: usize, seed: u64) -> NodeId {
    let mut rng = Lcg::new(seed);

    let tag_for = |rng: &mut Lcg| TAGS[rng.below(TAGS.len() as u64) as usize];
    let classes_for = |rng: &mut Lcg| -> Vec<String> {
        let n = rng.below(4); // 0..=3 classes
                              // Class lists are sets: draw distinct indices (duplicate classes in
                              // class="..." collapse in real DOMs, so the fixture must not emit them).
        let mut used = [false; CLASS_COUNT as usize];
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            loop {
                let c = rng.below(CLASS_COUNT) as usize;
                if !used[c] {
                    used[c] = true;
                    out.push(format!("c{c}"));
                    break;
                }
            }
        }
        out
    };

    let root_classes: Vec<String> = vec!["root".to_string()];
    let root = dom.create_element(
        "html",
        &root_classes.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );

    // (node, level) frontier
    let mut frontier = vec![(root, 0usize)];
    let mut made = 1usize;

    while made < target_nodes {
        let mut next_frontier = Vec::new();
        for &(parent, level) in &frontier {
            if level >= depth {
                continue;
            }
            let kids = 2 + rng.below(3) as usize; // 2..=4
            for _ in 0..kids {
                if made >= target_nodes {
                    break;
                }
                let classes = classes_for(&mut rng);
                let cls_refs: Vec<&str> = classes.iter().map(|s| s.as_str()).collect();
                let child = dom.create_element(tag_for(&mut rng), &cls_refs);
                dom.append(parent, child);
                next_frontier.push((child, level + 1));
                made += 1;
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }
    root
}

/// `count` random complex selectors, 1..=3 compounds, mixed combinators.
pub fn build_selectors(count: usize, seed: u64) -> Vec<String> {
    let mut rng = Lcg::new(seed ^ 0xFFFF);
    let mut out = Vec::with_capacity(count);
    for _i in 0..count {
        let compounds = 1 + rng.below(3) as usize;
        let mut s = String::new();
        for c in 0..compounds {
            if c > 0 {
                s.push_str(if rng.below(2) == 0 { " > " } else { " " });
            }
            // Rightmost compound gets a tag more often (realistic: subject usually typed)
            let want_tag = if c == compounds - 1 {
                rng.below(4) != 0
            } else {
                rng.below(2) == 0
            };
            let mut wrote = false;
            if want_tag {
                s.push_str(TAGS[rng.below(TAGS.len() as u64) as usize]);
                wrote = true;
            }
            if !wrote || rng.below(3) != 0 {
                let ncls = 1 + rng.below(2);
                for _ in 0..ncls {
                    s.push_str(&format!(".c{}", rng.below(CLASS_COUNT)));
                }
            }
        }
        out.push(s);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_sizes() {
        let mut dom = Dom::new();
        build_dom(&mut dom, 10_000, 12, 42);
        assert!(dom.len() >= 9_000 && dom.len() <= 10_500);
        let sels = build_selectors(200, 42);
        assert_eq!(sels.len(), 200);
        assert!(sels
            .iter()
            .all(|s| s.contains('.') || s.chars().next().is_some_and(|c| c.is_ascii_alphabetic())));
    }
}
