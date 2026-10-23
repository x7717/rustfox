//! Incremental restyle parity: `restyle` must be byte-identical to a fresh
//! full pass — same matched sets, same persistent states — for every dirty
//! set, every dirty-set shape, and every chaining depth.
//!
//! The dirty-set shapes tested are the ones that break naive incremental
//! schemes: nested dirty nodes inside one subtree, dirty roots (whole-tree
//! recomputes), dirty leaves, hi-class (>= 128) nodes on the spill path,
//! duplicated/out-of-range ids, and the all/none extremes.

use rustfox_dom::{ClassId, Dom};
use rustfox_style::{
    fast::{FastStyleSet, SoaCache},
    fixture, naive, parse_all,
};

/// LCG (fixture-style) so every run is exactly reproducible.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x2545F4914F6CDD1D) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        (self.next_u64() >> 32) % n
    }
}

/// Build the standard fixture: 2k nodes, 200 selectors, seed 42.
fn fixture_2k() -> (Dom, Vec<rustfox_style::Selector>, FastStyleSet, SoaCache) {
    let mut dom = Dom::new();
    let sources = fixture::build_selectors(200, 777);
    fixture::build_dom(&mut dom, 2_000, 12, 42);
    let selectors = parse_all(&mut dom, &sources);
    let set = FastStyleSet::new(selectors.clone());
    let cache = SoaCache::of(&dom);
    (dom, selectors, set, cache)
}

/// Flip fixture class `c{class}` on node `u` in BOTH the dom and the cache —
/// exactly what a live page's `classList.toggle` does, mirrored into the
/// SoA snapshot via the incremental `set_class` entry point.
fn flip(dom: &mut Dom, cache: &mut SoaCache, u: usize, class: u32, on: bool) {
    let cid = match dom.intern_class_probe(&format!("c{class}")) {
        Some(id) => id,
        None => dom.intern_class(&format!("c{class}")).0,
    };
    cache.set_class(u, cid, on);
    let node = &mut dom.nodes[u];
    if on {
        if !node.classes.contains(&ClassId(cid)) {
            node.classes.push(ClassId(cid));
        }
    } else {
        node.classes.retain(|&c| c.0 != cid);
    }
}

#[test]
fn states_pass_matches_style_pass() {
    let (_dom, _sels, set, cache) = fixture_2k();
    let (states_out, _states) = set.style_pass_states(&cache);
    let full = set.style_pass(&cache);
    assert_eq!(
        states_out.to_nested(),
        full.to_nested(),
        "style_pass_states output must equal style_pass"
    );
}

#[test]
fn random_dirty_sets_match_full_pass_and_chain() {
    let (mut dom, _sels, set, mut cache) = fixture_2k();
    let n = dom.len();
    let (mut prev, mut states) = set.style_pass_states(&cache);
    let mut rng = Lcg::new(0xDEAD_BEEF);

    for round in 0..25 {
        // 1..=80 distinct dirty nodes, biased toward shared subtrees in half
        // the rounds (nested dirty nodes + dirty chains). The window must be
        // wider than k, else the dedupe loop starves.
        let k = 1 + rng.below(80) as usize;
        let mut dirty = Vec::with_capacity(k);
        let mut seen = vec![false; n];
        let windowed = rng.below(2) == 0;
        let base = rng.below(n as u64) as usize;
        let mut tries = 0;
        while dirty.len() < k {
            tries += 1;
            let id = if windowed && tries <= 4 * k {
                (base + rng.below(256) as usize) % n
            } else {
                rng.below(n as u64) as usize
            };
            if !seen[id] {
                seen[id] = true;
                dirty.push(id as u32);
            }
        }
        // Real work: flip a fixture-vocabulary class on every dirty node;
        // alternating rounds flip it back, so masks evolve like a live page.
        let class = rng.below(64) as u32;
        let on = round % 2 == 0;
        for &d in &dirty {
            flip(&mut dom, &mut cache, d as usize, class, on);
        }

        // Ground truth on the SAME updated dom: a fresh full pass.
        let fresh = set.style_pass(&cache);
        let out = set.restyle(&cache, &mut states, &prev, &dirty);
        assert_eq!(
            out.to_nested(),
            fresh.to_nested(),
            "restyle != full pass at round {round}"
        );

        // Chaining: the persistent store must match a from-scratch store
        // bit-for-bit after every round.
        let (_f2, fresh_states) = set.style_pass_states(&cache);
        assert!(
            states.equals(&fresh_states),
            "restyle states != fresh states at round {round}"
        );
        prev = out;
    }
}

#[test]
fn restyle_matches_naive_end_to_end() {
    // The strongest parity: chained incremental restyles vs the naive
    // baseline matcher over the raw DOM (no fast-path involvement).
    let (mut dom, sels, set, mut cache) = fixture_2k();
    let n = dom.len();
    let mut rng = Lcg::new(0x5EED_5EED);
    let (mut prev, mut states) = set.style_pass_states(&cache);
    for round in 0..8 {
        let k = 3 + rng.below(40) as usize;
        let mut dirty = Vec::with_capacity(k);
        let mut seen = vec![false; n];
        while dirty.len() < k {
            let id = rng.below(n as u64) as usize;
            if !seen[id] {
                seen[id] = true;
                dirty.push(id as u32);
            }
        }
        let class = rng.below(64) as u32;
        let on = round % 2 == 0;
        for &d in &dirty {
            flip(&mut dom, &mut cache, d as usize, class, on);
        }
        prev = set.restyle(&cache, &mut states, &prev, &dirty);

        let ground = naive::style_pass(&dom, &sels);
        let got = prev.to_nested();
        assert_eq!(got.len(), ground.len(), "len mismatch at round {round}");
        for (i, (x, y)) in got.iter().zip(ground.iter()).enumerate() {
            assert_eq!(x, y, "node {i} round {round}: naive={x:?} restyle={y:?}");
        }
    }
}

#[test]
fn empty_and_all_dirty_extremes() {
    let (_dom, _sels, set, cache) = fixture_2k();
    let n = _dom.len();
    let (prev, mut states) = set.style_pass_states(&cache);

    // No dirty nodes: pure arena copy — identical output.
    let out = set.restyle(&cache, &mut states, &prev, &[]);
    assert_eq!(out.to_nested(), prev.to_nested());
    assert_eq!(out.indices, prev.indices);

    // All dirty: degenerates to a full pass.
    let all: Vec<u32> = (0..n as u32).collect();
    let out = set.restyle(&cache, &mut states, &prev, &all);
    let fresh = set.style_pass(&cache);
    assert_eq!(out.to_nested(), fresh.to_nested());

    // Duplicate + out-of-range ids must be tolerated (and be no-ops).
    let mut junk: Vec<u32> = vec![0, 0, n as u32 + 100, u32::MAX, 5, 5];
    junk.extend_from_slice(&all[..16]);
    let out2 = set.restyle(&cache, &mut states, &prev, &junk);
    let fresh2 = set.style_pass(&cache);
    assert_eq!(out2.to_nested(), fresh2.to_nested());
}

#[test]
fn dirty_root_recomputes_whole_tree() {
    let (_dom, _sels, set, mut cache) = fixture_2k();
    let n = _dom.len();
    let (prev, mut states) = set.style_pass_states(&cache);

    // The fixture is a BFS-built tree: node 0 owns (almost) the whole tree.
    let subtree = cache.subtree_len(0);
    assert!(subtree > n / 2, "fixture root should own most of the tree");

    cache.set_class(0, 3, true);
    let out = set.restyle(&cache, &mut states, &prev, &[0]);
    let fresh = set.style_pass(&cache);
    assert_eq!(out.to_nested(), fresh.to_nested());
    let (_f, fs) = set.style_pass_states(&cache);
    assert!(states.equals(&fs));
}

#[test]
fn hi_class_spill_node_restyle() {
    // Manual dom with hi classes (>= 128): the spill path must survive
    // incremental updates, including the O(n) CSR rebuild when a hi class
    // toggles.
    let mut dom = Dom::new();
    let root = dom.create_element("div", &["c200"]);
    let mid = dom.create_element("span", &["c130", "c5"]);
    let leaf = dom.create_element("p", &["c9"]);
    dom.append(root, mid);
    dom.append(mid, leaf);
    let sources = vec![
        ".c200 .c5".to_string(),
        "span.c130 > p".to_string(),
        ".c130 p".to_string(),
        "div .c9".to_string(),
        "*".to_string(),
    ];
    let selectors = parse_all(&mut dom, &sources);
    let set = FastStyleSet::new(selectors);
    let mut cache = SoaCache::of(&dom);
    let (prev, mut states) = set.style_pass_states(&cache);

    let mid_u = mid.0 as usize;
    let c130 = dom.intern_class_probe("c130").unwrap();
    let c200 = dom.intern_class_probe("c200").unwrap();

    cache.set_class(mid_u, c130, false);
    let out = set.restyle(&cache, &mut states, &prev, &[mid.0]);
    let fresh = set.style_pass(&cache);
    assert_eq!(out.to_nested(), fresh.to_nested());

    cache.set_class(mid_u, c130, true);
    let out = set.restyle(&cache, &mut states, &prev, &[mid.0]);
    let fresh = set.style_pass(&cache);
    assert_eq!(out.to_nested(), fresh.to_nested());

    // Hi class on the root, dirty = root: whole-tree recompute with spill.
    cache.set_class(0, c200, false);
    let out = set.restyle(&cache, &mut states, &prev, &[0]);
    let fresh = set.style_pass(&cache);
    assert_eq!(out.to_nested(), fresh.to_nested());
}

#[test]
fn state_store_layout_invariants() {
    let (_dom, _sels, set, cache) = fixture_2k();
    let (_prev, states) = set.style_pass_states(&cache);
    assert_eq!(states.wr(), set.wr_words());
    // equals() must be reflexive; a second identical pass reproduces states.
    let (p2, s2) = set.style_pass_states(&cache);
    assert!(!p2.indices.is_empty());
    assert!(states.equals(&s2));
}
