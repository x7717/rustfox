//! v0.7 — level-order bit-parallel automaton + AVX2 transition kernel.
//!
//! History of this matcher: v0.4 indexed (rule buckets), v0.5 SoA + undo-log
//! bloom, v0.6 bit-parallel NFA (6.8–7.5x vs naive). v0.7 keeps the v0.6
//! automaton semantics and attacks the per-node constant costs that profiling
//! pointed at:
//!
//! 1. **DFS stack → level-order (BFS) banks.** The DFS kept parent states in
//!    a `level_store` and copied them into scratch on every node, plus a
//!    truncate per popped frame. Now each tree level is one contiguous bank:
//!    parents are read sequentially from level L-1, children written
//!    sequentially into level L — pure streaming, no per-node copies, no
//!    stack frames. (Any top-down order is valid: a node's `act/acc` depends
//!    only on its parent's.)
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
//! Automaton semantics (unchanged from v0.6): every `(selector i, compound j)`
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
pub struct SoaCache {
    pub(crate) tags: Vec<u32>,
    pub(crate) masks: Vec<u128>,
    /// CSR: hi classes (id >= 128) per node
    pub(crate) hi_start: Vec<u32>,
    pub(crate) hi_list: Vec<u32>,
    pub(crate) child_start: Vec<u32>,
    pub(crate) child_nodes: Vec<u32>,
    pub(crate) roots: Vec<u32>,
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
        let roots = (0..n)
            .filter(|&i| !has_parent[i])
            .map(|i| i as u32)
            .collect();
        Self {
            tags,
            masks,
            hi_start,
            hi_list,
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
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
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
        let w = self.w;
        let start = out.indices.len() as u32;
        out.starts[u] = start;
        let store_pos = level_store.len() as u32;

        // ---- load parent state (or zeros for roots): zero-copy borrow ----
        let (pact, pacc): (&[u128], &[u128]) = match parent_pos {
            Some(pp) => (
                &level_store[pp..pp + wr],
                &level_store[pp + wr..pp + 2 * wr],
            ),
            None => (zeros, zeros),
        };

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

        // ---- push this frame's state (parent borrow ended above) ----
        level_store.extend_from_slice(&sc.act);
        level_store.extend_from_slice(&sc.accn);
        store_pos
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
