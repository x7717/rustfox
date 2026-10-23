//! v0.8 — incremental restyle on top of the bit-parallel automaton.
//!
//! History of this matcher: v0.4 indexed (rule buckets), v0.5 SoA + undo-log
//! bloom, v0.6 bit-parallel NFA (6.8–7.5x vs naive), v0.7 AVX2 transition
//! kernel + zero-copy parent reads (7.5–8.0x). v0.8 adds the steady-state
//! hot path a real browser runs millions of times per session:
//!
//! **Incremental restyle** ([`FastStyleSet::restyle`]). The automaton's
//! dataflow is strictly parent → child (a node's `act`/`acc` derive only
//! from its parent's state plus its own `(tag, class-mask)`), so a class
//! change on node `u` can only affect `u`'s subtree. `SoaCache` therefore
//! carries a `parent` array and a DFS `preorder` in which every subtree is
//! a contiguous range; `restyle` walks that array once — clean nodes copy
//! their previous match range (pure memcpy), dirty subtrees re-run the
//! automaton over a contiguous slice with freshly written parent states
//! read straight back. No stack, no recursion. Parity with a fresh full
//! pass is tested on randomized dirty-set sequences (including hi-class
//! spill, nested dirty nodes, dirty roots, and the all/none extremes) —
//! both matched sets AND persistent `StateStore` bits.
//!
//! Earlier v0.7 notes (kept for the optimization archaeology):
//!
//! 1. **Traversal order.** A level-order (BFS) two-bank layout was tried
//!    and regressed +32% @ 10k (working set O(level width) ≈ 600 KB vs
//!    O(depth) ≈ hundreds of bytes L1-resident for DFS) — reverted. The
//!    preorder array used by restyle is only an iteration ORDER, not a
//!    bank layout: parent states live in the per-node `StateStore`, so
//!    the working-set argument does not apply to it.
//! 2. **AVX2 transition kernel.** The hot inner expression
//!    `ok & ((child & act_sh) | (desc & acc_sh) | first)` and
//!    `accn = pacc | act` are pure bitwise ops over `w` u128 words —
//!    executed 2 words (32 bytes) per AVX2 instruction on x86_64, detected
//!    once per pass.
//! 3. **Direct-mapped `(tag, class-mask) → ok` cache** (`style_pass_cached`).
//!    v0.6 tried an FxHashMap variant and it regressed +23% (hash + insert +
//!    heap Vec per entry). This one is a fixed 2048-slot array with a
//!    fingerprint compare and an inline `[u128; 8]` payload — miss cost is a
//!    store, hit cost is a 128-byte copy that replaces ~4 hash probes.
//!    Benched separately; the fixture has ~30% repeat rate, real-world
//!    class vocabularies are far denser.
//! 4. **SoA `hi` classes flattened** from `Vec<Vec<u32>>` to CSR
//!    (`hi_start`/`hi_list`) — no per-node Vec in cache construction.
//!
//! Automaton semantics (unchanged since v0.6): every `(selector i, compound j)`
//! pair is one bit in a bank of u128 words packed by `off[i] + j`;
//! `act[u]` = prefixes ending exactly at `u`, `acc[u]` = union over the
//! root→u path; Child pairs shift `act`, Desc pairs shift `acc`; a match is
//! `act[u] & subj`. Output remains byte-identical to the naive baseline
//! (parity-tested, including the ≥128 class-id spill path).

use crate::indexed::MatchedStyles;
use crate::parser::{Combinator, Selector};
use rustc_hash::FxHashMap;
use rustfox_dom::Dom;
use std::cell::RefCell;

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

// --- direct-mapped ok cache ------------------------------------------------

const OKC_WORDS: usize = 8;
const OKC_SIZE: usize = 2048;
const OKC_EMPTY: u32 = u32::MAX;

#[derive(Clone)]
struct OkEntry {
    tag: u32,
    mask: u128,
    bits: [u128; OKC_WORDS],
}

impl OkEntry {
    const EMPTY: OkEntry = OkEntry {
        tag: OKC_EMPTY,
        mask: 0,
        bits: [0; OKC_WORDS],
    };
}

/// Precomputed automaton.
pub struct FastStyleSet {
    /// words per bitset (real)
    w: usize,
    /// `w` padded to even — AVX2 chunks read 2 words at a time
    wr: usize,
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
    /// shared across passes; keyed by (tag, mask) — both are pass-invariant
    ok_cache: RefCell<Vec<OkEntry>>,
}

/// Contiguous SoA snapshot of the DOM, built once per structure change and
/// reused across style passes (steady-state restyle is the browser hot path).
///
/// v0.8 adds the traversal scaffolding for incremental restyle:
///
/// * `parent[u]` — `u32::MAX` for roots. The automaton's dataflow is strictly
///   parent → child (a node's `act`/`acc` depend only on its parent's), so a
///   class change on node `u` can only alter states inside `u`'s subtree.
/// * `preorder` — DFS preorder over all roots (roots ascending, children
///   ascending — the exact order `style_pass` visits nodes). A subtree is a
///   CONTIGUOUS range `[pre_start[u], subtree_end[u])` of that array, so a
///   dirty subtree is processed by a plain `for` loop over a slice — no
///   stack, no recursion. Parent-before-child ordering also means a dirty
///   region can read its freshly computed parent state directly.
pub struct SoaCache {
    pub(crate) tags: Vec<u32>,
    pub(crate) masks: Vec<u128>,
    /// CSR: hi classes (id >= 128) per node
    pub(crate) hi_start: Vec<u32>,
    pub(crate) hi_list: Vec<u32>,
    pub(crate) child_start: Vec<u32>,
    pub(crate) child_nodes: Vec<u32>,
    pub(crate) roots: Vec<u32>,
    /// `u32::MAX` for roots
    pub(crate) parent: Vec<u32>,
    pub(crate) preorder: Vec<u32>,
    pub(crate) pre_start: Vec<u32>,
    /// exclusive end of `u`'s subtree in `preorder`
    pub(crate) subtree_end: Vec<u32>,
}

impl SoaCache {
    /// Incremental class update — the steady-state restyle entry point.
    ///
    /// Low classes (id < 128) flip one bit, O(1). A hi-class (id >= 128)
    /// change rebuilds the CSR spill arrays in O(n) — rare by construction
    /// (the fixture and most real vocabularies live far below 128 classes).
    /// After updating, pass the node id in the `dirty` list of the next
    /// [`FastStyleSet::restyle`] call.
    pub fn set_class(&mut self, u: usize, class: u32, on: bool) {
        assert!(u < self.tags.len(), "node id out of range");
        if class < MASK_BITS {
            let bit = 1u128 << class;
            if on {
                self.masks[u] |= bit;
            } else {
                self.masks[u] &= !bit;
            }
        } else {
            let mut list: Vec<u32> = Vec::with_capacity(self.hi_list.len());
            let mut start = Vec::with_capacity(self.tags.len() + 1);
            for i in 0..self.tags.len() {
                start.push(list.len() as u32);
                let s = self.hi_start[i] as usize;
                let e = self.hi_start[i + 1] as usize;
                let mut had = false;
                for k in s..e {
                    let c = self.hi_list[k];
                    if c == class {
                        had = true;
                        if !on {
                            continue;
                        }
                    }
                    list.push(c);
                }
                if on && !had && i == u {
                    list.push(class);
                }
            }
            start.push(list.len() as u32);
            self.hi_list = list;
            self.hi_start = start;
        }
    }

    /// Number of nodes in `u`'s subtree (the contiguous preorder slice the
    /// restyle recomputes when `u` is dirty).
    pub fn subtree_len(&self, u: usize) -> usize {
        (self.subtree_end[u] - self.pre_start[u]) as usize
    }
}

/// Persistent per-node automaton states (`act`/`accn`) for incremental
/// restyle — the memory Stylo would call "style data hanging off the
/// element". `2 * n * wr * 16 B` (≈2.5 MB for 10k nodes × 200 selectors),
/// written once per full pass and updated in place inside dirty subtrees
/// by [`FastStyleSet::restyle`].
pub struct StateStore {
    pub(crate) act: Vec<u128>,
    pub(crate) accn: Vec<u128>,
    wr: usize,
}

impl StateStore {
    pub(crate) fn new(n: usize, wr: usize) -> Self {
        StateStore {
            act: vec![0u128; n * wr],
            accn: vec![0u128; n * wr],
            wr,
        }
    }

    pub fn wr(&self) -> usize {
        self.wr
    }

    /// Exact state equality (act and accn) — used by parity tests to prove
    /// that an incremental restyle reproduces a fresh full pass bit-for-bit.
    pub fn equals(&self, other: &StateStore) -> bool {
        self.wr == other.wr && self.act == other.act && self.accn == other.accn
    }
}

impl SoaCache {
    pub fn of(dom: &Dom) -> Self {
        let n = dom.len();
        let mut tags = Vec::with_capacity(n);
        let mut masks = vec![0u128; n];
        let mut hi_start = Vec::with_capacity(n + 1);
        let mut hi_list: Vec<u32> = Vec::new();
        let mut child_start = vec![0u32; n + 1];
        let mut child_nodes: Vec<u32> = Vec::with_capacity(n);
        for (i, node) in dom.nodes.iter().enumerate() {
            tags.push(node.tag.0);
            for c in &node.classes {
                if c.0 < MASK_BITS {
                    masks[i] |= 1u128 << c.0;
                } else {
                    hi_list.push(c.0);
                }
            }
            hi_start.push(hi_list.len() as u32);
            child_start[i] = child_nodes.len() as u32;
            child_nodes.extend(node.children.iter().map(|c| c.0));
        }
        hi_start.push(hi_list.len() as u32);
        child_start[n] = child_nodes.len() as u32;
        // roots: nodes never listed as a child, ascending
        let mut has_parent = vec![false; n];
        for &c in &child_nodes {
            has_parent[c as usize] = true;
        }
        let roots: Vec<u32> = (0..n)
            .filter(|&i| !has_parent[i])
            .map(|i| i as u32)
            .collect();

        // --- restyle scaffolding: parent + preorder + subtree ranges ---
        let mut parent = vec![u32::MAX; n];
        for i in 0..n {
            for k in child_start[i] as usize..child_start[i + 1] as usize {
                parent[child_nodes[k] as usize] = i as u32;
            }
        }
        // DFS preorder, roots ascending, children ascending — byte-for-byte
        // the order style_pass visits nodes (per-root DFS, stack-based).
        let mut preorder: Vec<u32> = Vec::with_capacity(n);
        let mut pre_start = vec![0u32; n];
        let mut stack: Vec<u32> = roots.iter().rev().copied().collect();
        while let Some(u) = stack.pop() {
            let uu = u as usize;
            pre_start[uu] = preorder.len() as u32;
            preorder.push(u);
            for k in (child_start[uu] as usize..child_start[uu + 1] as usize).rev() {
                stack.push(child_nodes[k]);
            }
        }
        // Subtree sizes by reverse accumulation (children appear after their
        // parent in preorder), then `subtree_end[u] = pre_start[u] + size[u]`:
        // a subtree is one contiguous preorder slice.
        let mut size = vec![1u32; n];
        for i in (0..n).rev() {
            let u = preorder[i] as usize;
            let par = parent[u];
            if par != u32::MAX {
                size[par as usize] += size[u];
            }
        }
        let mut subtree_end = vec![0u32; n];
        for i in 0..n {
            subtree_end[i] = pre_start[i] + size[i];
        }

        Self {
            tags,
            masks,
            hi_start,
            hi_list,
            child_start,
            child_nodes,
            roots,
            parent,
            preorder,
            pre_start,
            subtree_end,
        }
    }
}

impl FastStyleSet {
    /// Padded words per bitset row (exposed for test/bench invariants).
    pub fn wr_words(&self) -> usize {
        self.wr
    }

    pub fn new(selectors: Vec<Selector>) -> Self {
        let n_sel = selectors.len();
        let k = selectors
            .iter()
            .map(|s| s.compounds.len())
            .max()
            .unwrap_or(1) as u32;
        let w = (n_sel * k as usize).div_ceil(128).max(1);
        let wr = (w + 1) & !1; // even: AVX2 processes 2 words per op

        let mut set = Self {
            w,
            wr,
            k,
            first: vec![0u128; wr],
            child: vec![0u128; wr],
            desc: vec![0u128; wr],
            subj: vec![0u128; wr],
            tag_maps: Vec::new(),
            tag_mask0: Vec::new(),
            tagfree: FxHashMap::default(),
            tagfree_mask0: vec![0u128; wr],
            hi_pairs: Vec::new(),
            universe: 0,
            ok_cache: RefCell::new(vec![OkEntry::EMPTY; OKC_SIZE]),
        };

        let max_tag = selectors
            .iter()
            .flat_map(|s| s.compounds.iter())
            .filter_map(|c| c.tag().map(|t| t.0 as usize + 1))
            .max()
            .unwrap_or(0);
        set.tag_maps = (0..max_tag).map(|_| FxHashMap::default()).collect();
        set.tag_mask0 = (0..max_tag).map(|_| vec![0u128; wr]).collect();

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
                                    .or_insert_with(|| vec![0u128; wr]);
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
                                    .or_insert_with(|| vec![0u128; wr]);
                                e[wi] |= bit;
                                for m in &mut set.tag_maps {
                                    let e = m.entry(mask_low).or_insert_with(|| vec![0u128; wr]);
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
        self.style_pass_impl(soa, false)
    }

    /// Same pass with the direct-mapped `(tag, mask) → ok` cache enabled.
    /// Wins when (tag, class-mask) combinations repeat across nodes —
    /// real-world class vocabularies repeat heavily; the synthetic fixture
    /// repeats ~30%. Benched separately as `fast_okc`.
    pub fn style_pass_cached(&self, soa: &SoaCache) -> MatchedStyles {
        self.style_pass_impl(soa, true)
    }

    /// Full pass that also persists per-node states into a [`StateStore`] —
    /// the base snapshot [`Self::restyle`] updates incrementally.
    ///
    /// Same traversal order and same output as [`Self::style_pass`]
    /// (parity-tested), but the DFS stack is replaced by the precomputed
    /// preorder array and parent states are read from the store directly.
    pub fn style_pass_states(&self, soa: &SoaCache) -> (MatchedStyles, StateStore) {
        let n = soa.tags.len();
        let mut out = MatchedStyles {
            starts: vec![0; n],
            ends: vec![0; n],
            indices: Vec::new(),
        };
        let mut store = StateStore::new(n, self.wr);
        if self.w == 0 || n == 0 {
            return (out, store);
        }

        #[cfg(target_arch = "x86_64")]
        let avx = is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let avx = false;

        let wr = self.wr;
        let mut sc = Scratch::new(wr);
        let zeros: Vec<u128> = vec![0u128; wr];

        // Preorder walk: parent states are read straight from the store
        // (already final — preorder guarantees parent-before-child), the
        // node's own state is written back at its slot. No stack, no
        // truncate/extend bookkeeping — one branchless loop over the order.
        for &p in &soa.preorder {
            let u = p as usize;
            let par = soa.parent[u];
            let (pact, pacc): (&[u128], &[u128]) = if par == u32::MAX {
                (&zeros, &zeros)
            } else {
                let pb = par as usize * wr;
                (&store.act[pb..pb + wr], &store.accn[pb..pb + wr])
            };
            self.run_node(u, soa, pact, pacc, &mut sc, false, avx, &mut out);
            let b = u * wr;
            store.act[b..b + wr].copy_from_slice(&sc.act);
            store.accn[b..b + wr].copy_from_slice(&sc.accn);
        }
        (out, store)
    }

    /// **Incremental restyle** — recompute only what a class change can
    /// possibly affect.
    ///
    /// Contract (Stylo-style):
    ///
    /// * `soa` already reflects the NEW classes (use [`SoaCache::set_class`]);
    /// * `dirty` lists the node ids whose classes changed since the last
    ///   pass (duplicates are fine, out-of-range ids are ignored);
    /// * `states` is the persistent store from the previous
    ///   [`Self::style_pass_states`] / [`Self::restyle`] call — updated in
    ///   place, so the next restyle chains directly;
    /// * `prev` is the previous pass's [`MatchedStyles`] — used ONLY for
    ///   clean nodes' index ranges (their states and matches are unchanged:
    ///   the automaton's dataflow is strictly parent → child).
    ///
    /// Output is byte-identical to a fresh full pass on the updated `soa`
    /// (parity-tested on randomized dirty sets, including hi-class spill,
    /// nested dirty nodes, dirty roots, and the all/none extremes).
    ///
    /// Cost model: O(n) cheap arena rebuild (copying clean ranges — a few
    /// ns/node of memcpy) + full automaton work ONLY inside dirty subtrees.
    /// Structural changes (create/remove nodes) invalidate the SoA snapshot
    /// and require a rebuild + full pass.
    pub fn restyle(
        &self,
        soa: &SoaCache,
        states: &mut StateStore,
        prev: &MatchedStyles,
        dirty: &[u32],
    ) -> MatchedStyles {
        let n = soa.tags.len();
        let mut out = MatchedStyles {
            starts: vec![0; n],
            ends: vec![0; n],
            indices: Vec::with_capacity(prev.indices.len()),
        };
        if self.w == 0 || n == 0 {
            return out;
        }

        #[cfg(target_arch = "x86_64")]
        let avx = is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let avx = false;

        let wr = self.wr;
        let mut sc = Scratch::new(wr);
        let zeros: Vec<u128> = vec![0u128; wr];

        // 1. flag dirty nodes (dedupe for free)
        let mut dirty_f = vec![false; n];
        for &d in dirty {
            if (d as usize) < n {
                dirty_f[d as usize] = true;
            }
        }
        // 2. expand to subtree closure: preorder guarantees the parent is
        //    visited before the child, so one linear pass suffices.
        let mut expanded = vec![false; n];
        for &p in &soa.preorder {
            let u = p as usize;
            let par = soa.parent[u];
            expanded[u] = dirty_f[u] || (par != u32::MAX && expanded[par as usize]);
        }

        // 3. rebuild the flat arena in preorder order: clean nodes copy
        //    their previous index range (pure memcpy of u32s), dirty
        //    subtrees run the automaton over one contiguous preorder slice.
        let order = &soa.preorder;
        let mut p = 0usize;
        while p < n {
            let u = order[p] as usize;
            if !expanded[u] {
                let s = prev.starts[u] as usize;
                let e = prev.ends[u] as usize;
                out.starts[u] = out.indices.len() as u32;
                out.indices.extend_from_slice(&prev.indices[s..e]);
                out.ends[u] = out.indices.len() as u32;
                p += 1;
            } else {
                // dirty root: its whole subtree is [p, subtree_end[u]) —
                // a contiguous slice, processed parent-before-child, so
                // freshly written parent states are read back directly.
                let end = soa.subtree_end[u] as usize;
                for &q in &order[p..end] {
                    let v = q as usize;
                    let par = soa.parent[v];
                    let (pact, pacc): (&[u128], &[u128]) = if par == u32::MAX {
                        (&zeros, &zeros)
                    } else {
                        let pb = par as usize * wr;
                        (&states.act[pb..pb + wr], &states.accn[pb..pb + wr])
                    };
                    self.run_node(v, soa, pact, pacc, &mut sc, false, avx, &mut out);
                    let b = v * wr;
                    states.act[b..b + wr].copy_from_slice(&sc.act);
                    states.accn[b..b + wr].copy_from_slice(&sc.accn);
                }
                p = end;
            }
        }
        out
    }

    fn style_pass_impl(&self, soa: &SoaCache, use_cache: bool) -> MatchedStyles {
        let n = soa.tags.len();
        let mut out = MatchedStyles {
            starts: vec![0; n],
            ends: vec![0; n],
            indices: Vec::new(),
        };
        if self.w == 0 || n == 0 {
            return out;
        }

        #[cfg(target_arch = "x86_64")]
        let avx = is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let avx = false;

        let wr = self.wr;

        let mut sc = Scratch::new(wr);
        // immutable zeros for root nodes (no parent state to read)
        let zeros: Vec<u128> = vec![0u128; wr];

        // DFS with a frame store: `level_store` holds (act, accn) for every
        // node on the current root->leaf path, `stack` walks the child ranges.
        // Working set is O(depth * wr) — a few hundred bytes, L1-resident.
        //
        // v0.7 experiment log: a level-order (BFS) two-bank layout was tried
        // first (parents read sequentially from the previous level, children
        // streamed into the next) — it regressed +32% @ 10k because the
        // working set became O(max level width) ≈ 600 KB of L2/L3 traffic per
        // level, where DFS keeps the entire active state in L1. Reverted;
        // the real wins (zero-copy parent reads + AVX2 transition + ok-cache)
        // are orthogonal to the traversal order and kept.
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
            let pos = self.enter_node(
                ru,
                soa,
                None,
                &mut level_store,
                &mut sc,
                &mut out,
                use_cache,
                avx,
                &zeros,
            );
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
                    let pos = self.enter_node(
                        u,
                        soa,
                        Some(ppos),
                        &mut level_store,
                        &mut sc,
                        &mut out,
                        use_cache,
                        avx,
                        &zeros,
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

    /// Compute `ok[u]`, run the transition, extract matched selectors, and
    /// push the node's act/accn onto `level_store`. The parent's state is
    /// read DIRECTLY from the store (v0.6 copied it into scratch first —
    /// 2×wr×16 bytes of pure memcpy per node, gone in v0.7).
    /// Returns the store position of this frame.
    #[allow(clippy::too_many_arguments)]
    fn enter_node(
        &self,
        u: usize,
        soa: &SoaCache,
        parent_pos: Option<usize>,
        level_store: &mut Vec<u128>,
        sc: &mut Scratch,
        out: &mut MatchedStyles,
        use_cache: bool,
        avx: bool,
        zeros: &[u128],
    ) -> u32 {
        let wr = self.wr;
        let store_pos = level_store.len() as u32;
        let (pact, pacc): (&[u128], &[u128]) = match parent_pos {
            Some(pp) => (
                &level_store[pp..pp + wr],
                &level_store[pp + wr..pp + 2 * wr],
            ),
            None => (zeros, zeros),
        };
        self.run_node(u, soa, pact, pacc, sc, use_cache, avx, out);
        level_store.extend_from_slice(&sc.act);
        level_store.extend_from_slice(&sc.accn);
        store_pos
    }

    /// The per-node automaton step shared by all pass flavors (v0.8
    /// refactor): `ok[u]` from the tables, parent-state shift, transition,
    /// match extraction into `out`. Parent state arrives as borrowed
    /// slices — from `level_store`, the persistent [`StateStore`], or the
    /// zero vector for roots. Writing the node's own state back is the
    /// caller's job (append vs slot write differ per pass flavor).
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn run_node(
        &self,
        u: usize,
        soa: &SoaCache,
        pact: &[u128],
        pacc: &[u128],
        sc: &mut Scratch,
        use_cache: bool,
        avx: bool,
        out: &mut MatchedStyles,
    ) {
        let wr = self.wr;
        let w = self.w;
        out.starts[u] = out.indices.len() as u32;

        // ---- ok[u]: pairs whose compound matches this node ----
        let tag_u = soa.tags[u] as usize;
        let nmask = soa.masks[u] & self.universe;
        let node_has_hi = soa.hi_start[u] != soa.hi_start[u + 1];
        let cacheable = use_cache && !node_has_hi && wr <= OKC_WORDS;
        let mut ok_from_cache = false;
        if cacheable {
            ok_from_cache = self.ok_cache_copy_to(tag_u as u32, nmask, sc.ok.as_mut_slice());
        }
        if !ok_from_cache {
            self.compute_ok(u, soa, tag_u, nmask, sc.ok.as_mut_slice());
            if cacheable {
                self.ok_cache_store(tag_u as u32, nmask, &sc.ok);
            }
        }

        // ---- shift parent states LEFT by one pair slot ----
        // a prefix match at pair (i, j-1) in the parent becomes available for
        // pair (i, j) in the child; bit 127 carries across words. Bits that
        // spill into a selector's chain-start slot die at the transition,
        // because chain-start slots are gated by `first`, not `child`/`desc`.
        // The loop runs over `wr` (padded) words: padding words are zero and
        // stay zero — the transition masks them with the zero-padded
        // `child`/`desc`/`first` banks.
        let mut act_carry = 0u128;
        let mut acc_carry = 0u128;
        for (((a, c), ash), accsh) in pact
            .iter()
            .zip(pacc.iter())
            .zip(sc.act_sh.iter_mut())
            .zip(sc.acc_sh.iter_mut())
        {
            *ash = (*a << 1) | act_carry;
            act_carry = *a >> 127;
            *accsh = (*c << 1) | acc_carry;
            acc_carry = *c >> 127;
        }

        // ---- transition ----
        if avx {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: avx2 detected once per pass; all slices are `wr` (even)
            // words long, so 32-byte chunk loads stay in bounds.
            unsafe {
                transition_avx2(
                    &sc.ok,
                    &self.child,
                    &self.desc,
                    &self.first,
                    &sc.act_sh,
                    &sc.acc_sh,
                    pacc,
                    &mut sc.act,
                    &mut sc.accn,
                    wr / 2,
                );
            }
        } else {
            for i in 0..wr {
                sc.act[i] = sc.ok[i]
                    & ((self.child[i] & sc.act_sh[i])
                        | (self.desc[i] & sc.acc_sh[i])
                        | self.first[i]);
                sc.accn[i] = pacc[i] | sc.act[i];
            }
        }

        // ---- extract matched selectors, ascending (no sort needed) ----
        for wi in 0..w {
            let mut m = sc.act[wi] & self.subj[wi];
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                let p = (wi * 128 + b) as u32;
                out.indices.push(p / self.k);
                m &= m - 1;
            }
        }
        out.ends[u] = out.indices.len() as u32;
    }

    /// `ok[u]` from the automaton tables (mask-less direct index, then the
    /// subset enumeration over the class-mask map, then hi-class spill).
    #[allow(clippy::needless_range_loop)]
    fn compute_ok(&self, u: usize, soa: &SoaCache, tag_u: usize, nmask: u128, ok: &mut [u128]) {
        if tag_u < self.tag_mask0.len() {
            ok.copy_from_slice(&self.tag_mask0[tag_u]);
        } else {
            ok.copy_from_slice(&self.tagfree_mask0);
        }
        // pairs with classes: compound mask must be a subset of the node mask
        if nmask != 0 {
            let map: &FxHashMap<u128, Vec<u128>> = if tag_u < self.tag_maps.len() {
                &self.tag_maps[tag_u]
            } else {
                &self.tagfree
            };
            let mut sub = nmask;
            loop {
                if let Some(bs) = map.get(&sub) {
                    for (o, b) in ok.iter_mut().zip(bs.iter()) {
                        *o |= *b;
                    }
                }
                if sub == 0 {
                    break;
                }
                sub = (sub - 1) & nmask;
            }
        }
        if soa.hi_start[u] != soa.hi_start[u + 1] {
            let hic = &soa.hi_list[soa.hi_start[u] as usize..soa.hi_start[u + 1] as usize];
            for hp in &self.hi_pairs {
                if hp.tag.is_none_or(|t| t as usize == tag_u)
                    && soa.masks[u] & hp.mask_low == hp.mask_low
                    && hp.hi.iter().all(|c| hic.contains(c))
                {
                    ok[(hp.p / 128) as usize] |= 1u128 << (hp.p % 128);
                }
            }
        }
    }

    #[inline]
    fn okc_index(tag: u32, mask: u128) -> usize {
        let h = (tag as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (mask as u64)
            ^ ((mask >> 64) as u64).rotate_left(29);
        (h as usize) & (OKC_SIZE - 1)
    }

    /// On fingerprint hit, copy the cached bits into `dst`; returns hit.
    #[inline]
    fn ok_cache_copy_to(&self, tag: u32, mask: u128, dst: &mut [u128]) -> bool {
        let cache = self.ok_cache.borrow();
        let e = &cache[Self::okc_index(tag, mask)];
        if e.tag == tag && e.mask == mask {
            dst.copy_from_slice(&e.bits[..dst.len()]);
            true
        } else {
            false
        }
    }

    #[inline]
    fn ok_cache_store(&self, tag: u32, mask: u128, src: &[u128]) {
        let mut cache = self.ok_cache.borrow_mut();
        let e = &mut cache[Self::okc_index(tag, mask)];
        e.tag = tag;
        e.mask = mask;
        e.bits = [0u128; OKC_WORDS];
        e.bits[..src.len()].copy_from_slice(src);
    }
}

/// Reusable per-pass word buffers.
struct Scratch {
    ok: Vec<u128>,
    act_sh: Vec<u128>,
    acc_sh: Vec<u128>,
    act: Vec<u128>,
    accn: Vec<u128>,
}

impl Scratch {
    fn new(wr: usize) -> Self {
        Self {
            ok: vec![0u128; wr],
            act_sh: vec![0u128; wr],
            acc_sh: vec![0u128; wr],
            act: vec![0u128; wr],
            accn: vec![0u128; wr],
        }
    }
}

/// AVX2 transition: 2 u128 words (32 bytes) per instruction.
///
/// `act = ok & ((child & act_sh) | (desc & acc_sh) | first)`
/// `accn = pacc | act`
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn transition_avx2(
    ok: &[u128],
    child: &[u128],
    desc: &[u128],
    first: &[u128],
    act_sh: &[u128],
    acc_sh: &[u128],
    pacc: &[u128],
    act_out: &mut [u128],
    accn_out: &mut [u128],
    chunks: usize,
) {
    use std::arch::x86_64::*;
    for c in 0..chunks {
        let i = c * 2;
        let v_ok = _mm256_loadu_si256(ok.as_ptr().add(i) as *const __m256i);
        let v_child = _mm256_loadu_si256(child.as_ptr().add(i) as *const __m256i);
        let v_desc = _mm256_loadu_si256(desc.as_ptr().add(i) as *const __m256i);
        let v_first = _mm256_loadu_si256(first.as_ptr().add(i) as *const __m256i);
        let v_act_sh = _mm256_loadu_si256(act_sh.as_ptr().add(i) as *const __m256i);
        let v_acc_sh = _mm256_loadu_si256(acc_sh.as_ptr().add(i) as *const __m256i);
        let v_pacc = _mm256_loadu_si256(pacc.as_ptr().add(i) as *const __m256i);
        let t = _mm256_or_si256(
            _mm256_and_si256(v_child, v_act_sh),
            _mm256_and_si256(v_desc, v_acc_sh),
        );
        let t = _mm256_or_si256(t, v_first);
        let act = _mm256_and_si256(v_ok, t);
        let accn = _mm256_or_si256(v_pacc, act);
        _mm256_storeu_si256(act_out.as_mut_ptr().add(i) as *mut __m256i, act);
        _mm256_storeu_si256(accn_out.as_mut_ptr().add(i) as *mut __m256i, accn);
    }
}
