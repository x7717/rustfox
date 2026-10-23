//! v0.6 — bit-parallel selector automaton ("NFA over the tree").
//!
//! The v0.4/v0.5 family matched right-to-left: every candidate selector
//! triggered an ancestor-chain walk, bounded below by
//! `candidates x depth` compound checks. Profiling showed those walks are
//! the irreducible cost of that architecture, so the architecture goes.
//!
//! This matcher is a top-down left-to-right bit-parallel NFA, the same trick
//! bit-parallel regex engines use on strings, lifted to trees:
//!
//! * Every `(selector i, compound j)` pair is one bit in a bank of `u128`
//!   words; pairs of a selector are packed consecutively (`off[i] + j`).
//! * `act[u]` = pairs whose prefix `compounds[0..=j]` matches ending exactly
//!   at node `u`; `acc[u]` = union of `act` over the root->u path.
//! * Transition at `u` (per word, ~W u128 ops total):
//!   - `Child` pair j: needs `act[parent]` bit `j-1`  (multiword `<< 1`)
//!   - `Desc`  pair j: needs `acc[parent]` bit `j-1`
//!   - AND `ok[u]` = bitset of pairs whose compound matches node u.
//! * `ok[u]`: mask-less pairs come from a direct per-tag bitset (no probe);
//!   the rest are indexed by exact class mask — a compound mask must be a
//!   *subset* of the node mask, so only `2^popcount-1` subset probes run.
//!   Bits shifted across selector boundaries die at the transition because
//!   chain-start slots are gated by `first`, not by `child`/`desc`.
//! * Selector i matches node u iff `act[u]` has its subject bit — extraction
//!   walks words in order, so output is ascending by selector, no sort.
//!
//! The DOM is streamed through a `SoaCache` (tag ids, class masks, flattened
//! children). Building it is O(n) once; steady-state restyles — the hot path
//! in a real browser — reuse it.
//!
//! Output is byte-identical to the naive baseline (parity-tested).

use crate::indexed::MatchedStyles;
use crate::parser::{Combinator, Selector};
use rustc_hash::FxHashMap;
use rustfox_dom::Dom;

/// Classes with id < 128 live in masks; higher ids spill to `hi_pairs`.
const MASK_BITS: u32 = 128;

/// One compound that contains a class with id >= 128; cannot live in the
/// mask index, so it is verified per node on the rare hi-class path.
struct HiPair {
    /// pair bit index `off[i] + j`
    p: u32,
    tag: Option<u32>,
    mask_low: u128,
    hi: Vec<u32>,
}

/// Precomputed automaton.
pub struct FastStyleSet {
    /// words per bitset
    w: usize,
    /// max compounds per selector (pair stride)
    k: u32,
    first: Vec<u128>,
    child: Vec<u128>,
    desc: Vec<u128>,
    subj: Vec<u128>,
    /// per tag id: (class-mask -> pairs with that exact mask and (no tag or this tag))
    tag_maps: Vec<FxHashMap<u128, Vec<u128>>>,
    /// mask-less pairs per tag, direct-indexed (the hottest lookup)
    tag_mask0: Vec<Vec<u128>>,
    /// fallbacks for node tags outside the selector set
    tagfree: FxHashMap<u128, Vec<u128>>,
    tagfree_mask0: Vec<u128>,
    hi_pairs: Vec<HiPair>,
    /// OR of all low class masks — restricts subset enumeration
    universe: u128,
}

/// Contiguous SoA snapshot of the DOM, built once per structure change and
/// reused across style passes (steady-state restyle is the browser hot path).
pub struct SoaCache {
    pub(crate) tags: Vec<u32>,
    pub(crate) masks: Vec<u128>,
    pub(crate) hi: Vec<Vec<u32>>,
    pub(crate) child_start: Vec<u32>,
    pub(crate) child_nodes: Vec<u32>,
    pub(crate) roots: Vec<u32>,
}

impl SoaCache {
    pub fn of(dom: &Dom) -> Self {
        let n = dom.len();
        let mut tags = Vec::with_capacity(n);
        let mut masks = vec![0u128; n];
        let mut hi: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut child_start = vec![0u32; n + 1];
        let mut child_nodes: Vec<u32> = Vec::with_capacity(n);
        for (i, node) in dom.nodes.iter().enumerate() {
            tags.push(node.tag.0);
            for c in &node.classes {
                if c.0 < MASK_BITS {
                    masks[i] |= 1u128 << c.0;
                } else {
                    hi[i].push(c.0);
                }
            }
            child_start[i] = child_nodes.len() as u32;
            child_nodes.extend(node.children.iter().map(|c| c.0));
        }
        child_start[n] = child_nodes.len() as u32;
        // roots: nodes never listed as a child, ascending
        let mut has_parent = vec![false; n];
        for &c in &child_nodes {
            has_parent[c as usize] = true;
        }
        let roots = (0..n)
            .filter(|&i| !has_parent[i])
            .map(|i| i as u32)
            .collect();
        Self {
            tags,
            masks,
            hi,
            child_start,
            child_nodes,
            roots,
        }
    }
}

impl FastStyleSet {
    pub fn new(selectors: Vec<Selector>) -> Self {
        let n_sel = selectors.len();
        let k = selectors
            .iter()
            .map(|s| s.compounds.len())
            .max()
            .unwrap_or(1) as u32;
        let w = (n_sel * k as usize).div_ceil(128).max(1);

        let mut set = Self {
            w,
            k,
            first: vec![0u128; w],
            child: vec![0u128; w],
            desc: vec![0u128; w],
            subj: vec![0u128; w],
            tag_maps: Vec::new(),
            tag_mask0: Vec::new(),
            tagfree: FxHashMap::default(),
            tagfree_mask0: vec![0u128; w],
            hi_pairs: Vec::new(),
            universe: 0,
        };

        let max_tag = selectors
            .iter()
            .flat_map(|s| s.compounds.iter())
            .filter_map(|c| c.tag().map(|t| t.0 as usize + 1))
            .max()
            .unwrap_or(0);
        set.tag_maps = (0..max_tag).map(|_| FxHashMap::default()).collect();
        set.tag_mask0 = (0..max_tag).map(|_| vec![0u128; w]).collect();

        for (i, sel) in selectors.iter().enumerate() {
            let len = sel.compounds.len();
            for (j, comp) in sel.compounds.iter().enumerate() {
                let p = i as u32 * k + j as u32;
                let wi = (p / 128) as usize;
                let bit = 1u128 << (p % 128);

                if j == 0 {
                    set.first[wi] |= bit;
                }
                if j > 0 {
                    match sel.combinators[j - 1] {
                        Combinator::Child => set.child[wi] |= bit,
                        Combinator::Descendant => set.desc[wi] |= bit,
                    }
                }
                if j + 1 == len {
                    set.subj[wi] |= bit;
                }

                let mut mask_low = 0u128;
                let mut hi = Vec::new();
                for c in comp.classes() {
                    if c.0 < MASK_BITS {
                        mask_low |= 1u128 << c.0;
                    } else {
                        hi.push(c.0);
                    }
                }
                set.universe |= mask_low;

                if hi.is_empty() {
                    match comp.tag().map(|t| t.0 as usize) {
                        Some(t) => {
                            if mask_low == 0 {
                                set.tag_mask0[t][wi] |= bit;
                            } else {
                                let e = set.tag_maps[t]
                                    .entry(mask_low)
                                    .or_insert_with(|| vec![0u128; w]);
                                e[wi] |= bit;
                            }
                        }
                        None => {
                            if mask_low == 0 {
                                set.tagfree_mask0[wi] |= bit;
                                for m in &mut set.tag_mask0 {
                                    m[wi] |= bit;
                                }
                            } else {
                                let e = set
                                    .tagfree
                                    .entry(mask_low)
                                    .or_insert_with(|| vec![0u128; w]);
                                e[wi] |= bit;
                                for m in &mut set.tag_maps {
                                    let e = m.entry(mask_low).or_insert_with(|| vec![0u128; w]);
                                    e[wi] |= bit;
                                }
                            }
                        }
                    }
                } else {
                    set.hi_pairs.push(HiPair {
                        p,
                        tag: comp.tag().map(|t| t.0),
                        mask_low,
                        hi,
                    });
                }
            }
        }
        set
    }

    /// Cold path: build the SoA snapshot and run one pass.
    pub fn style_pass_dom(&self, dom: &Dom) -> MatchedStyles {
        let cache = SoaCache::of(dom);
        self.style_pass(&cache)
    }

    /// Steady-state pass over a cached SoA snapshot.
    /// Output identical to the naive matcher (parity-tested).
    pub fn style_pass(&self, soa: &SoaCache) -> MatchedStyles {
        let n = soa.tags.len();
        let mut out = MatchedStyles {
            starts: vec![0; n],
            ends: vec![0; n],
            indices: Vec::new(),
        };
        if self.w == 0 || n == 0 {
            return out;
        }

        let mut sc = Scratch::new(self.w);
        let mut level_store: Vec<u128> = Vec::new();

        /// DFS frame: child range + where this node's act/acc live.
        struct Ent {
            next: u32,
            end: u32,
            pos: u32,
        }
        let mut stack: Vec<Ent> = Vec::new();

        for root in &soa.roots {
            let ru = *root as usize;
            let pos = enter_node(self, ru, soa, None, &mut sc, &mut level_store, &mut out);
            stack.push(Ent {
                next: soa.child_start[ru],
                end: soa.child_start[ru + 1],
                pos,
            });

            while !stack.is_empty() {
                let descend = {
                    let top = stack.last().expect("stack checked non-empty");
                    top.next < top.end
                };
                if descend {
                    let (child, ppos) = {
                        let top = stack.last_mut().expect("stack checked non-empty");
                        let c = soa.child_nodes[top.next as usize];
                        top.next += 1;
                        (c, top.pos as usize)
                    };
                    let u = child as usize;
                    let pos = enter_node(
                        self,
                        u,
                        soa,
                        Some(ppos),
                        &mut sc,
                        &mut level_store,
                        &mut out,
                    );
                    stack.push(Ent {
                        next: soa.child_start[u],
                        end: soa.child_start[u + 1],
                        pos,
                    });
                } else {
                    let ent = stack.pop().expect("stack checked non-empty");
                    level_store.truncate(ent.pos as usize);
                }
            }
        }
        out
    }
}

/// Reusable per-pass word buffers.
struct Scratch {
    ok: Vec<u128>,
    act: Vec<u128>,
    accn: Vec<u128>,
    act_sh: Vec<u128>,
    acc_sh: Vec<u128>,
    pact: Vec<u128>,
    pacc: Vec<u128>,
}

impl Scratch {
    fn new(w: usize) -> Self {
        Self {
            ok: vec![0u128; w],
            act: vec![0u128; w],
            accn: vec![0u128; w],
            act_sh: vec![0u128; w],
            acc_sh: vec![0u128; w],
            pact: vec![0u128; w],
            pacc: vec![0u128; w],
        }
    }
}

/// Compute `act`/`acc` for node `u` given the parent's state, run the
/// transition, extract matched selectors, and push the state to
/// `level_store`. Returns the store position of this frame.
#[allow(clippy::needless_range_loop)]
fn enter_node(
    set: &FastStyleSet,
    u: usize,
    soa: &SoaCache,
    parent_pos: Option<usize>,
    sc: &mut Scratch,
    level_store: &mut Vec<u128>,
    out: &mut MatchedStyles,
) -> u32 {
    let w = set.w;
    let start = out.indices.len() as u32;
    out.starts[u] = start;

    // ---- load parent state (or zeros for roots) ----
    match parent_pos {
        Some(pp) => {
            sc.pact.copy_from_slice(&level_store[pp..pp + w]);
            sc.pacc.copy_from_slice(&level_store[pp + w..pp + 2 * w]);
        }
        None => {
            sc.pact.fill(0);
            sc.pacc.fill(0);
        }
    }
    let parent_act: &[u128] = &sc.pact;
    let parent_acc: &[u128] = &sc.pacc;

    // ---- ok[u]: pairs whose compound matches this node ----
    // mask-less pairs come from the direct per-tag bitset
    let tag_u = soa.tags[u] as usize;
    if tag_u < set.tag_mask0.len() {
        sc.ok.copy_from_slice(&set.tag_mask0[tag_u]);
    } else {
        sc.ok.copy_from_slice(&set.tagfree_mask0);
    }
    // pairs with classes: compound mask must be a subset of the node mask
    let nmask = soa.masks[u] & set.universe;
    if nmask != 0 {
        let map: &FxHashMap<u128, Vec<u128>> = if tag_u < set.tag_maps.len() {
            &set.tag_maps[tag_u]
        } else {
            &set.tagfree
        };
        let mut sub = nmask;
        loop {
            if let Some(bs) = map.get(&sub) {
                for (o, b) in sc.ok.iter_mut().zip(bs.iter()) {
                    *o |= *b;
                }
            }
            if sub == 0 {
                break;
            }
            sub = (sub - 1) & nmask;
        }
    }
    if !soa.hi[u].is_empty() {
        let hic = &soa.hi[u];
        for hp in &set.hi_pairs {
            if hp.tag.is_none_or(|t| t as usize == tag_u)
                && soa.masks[u] & hp.mask_low == hp.mask_low
                && hp.hi.iter().all(|c| hic.contains(c))
            {
                sc.ok[(hp.p / 128) as usize] |= 1u128 << (hp.p % 128);
            }
        }
    }

    // ---- shift parent states LEFT by one pair slot ----
    // a prefix match at pair (i, j-1) in the parent becomes available for
    // pair (i, j) in the child; bit 127 carries across words. Bits that
    // spill into a selector's chain-start slot die at the transition,
    // because chain-start slots are gated by `first`, not `child`/`desc`.
    let mut act_carry = 0u128;
    let mut acc_carry = 0u128;
    for i in 0..w {
        sc.act_sh[i] = (parent_act[i] << 1) | act_carry;
        act_carry = parent_act[i] >> 127;
        sc.acc_sh[i] = (parent_acc[i] << 1) | acc_carry;
        acc_carry = parent_acc[i] >> 127;
    }

    // ---- transition ----
    for i in 0..w {
        sc.act[i] = sc.ok[i]
            & ((set.child[i] & sc.act_sh[i]) | (set.desc[i] & sc.acc_sh[i]) | set.first[i]);
        sc.accn[i] = parent_acc[i] | sc.act[i];
    }

    // ---- extract matched selectors, ascending (no sort needed) ----
    for wi in 0..w {
        let mut m = sc.act[wi] & set.subj[wi];
        while m != 0 {
            let b = m.trailing_zeros() as usize;
            let p = (wi * 128 + b) as u32;
            out.indices.push(p / set.k);
            m &= m - 1;
        }
    }
    out.ends[u] = out.indices.len() as u32;

    let pos = level_store.len() as u32;
    level_store.extend_from_slice(&sc.act);
    level_store.extend_from_slice(&sc.accn);
    pos
}
