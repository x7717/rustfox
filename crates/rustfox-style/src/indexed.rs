//! Subject-indexed style set — rule bucketing, the production technique.
//!
//! Profiling the naive/bloom passes shows the dominant cost is probing all
//! N selectors against every node, where almost always the *subject*
//! (rightmost) compound fails. Buckets pre-group selectors by the subject's
//! tag and first class, so each node only full-matches a handful of
//! candidates. Combined with the ancestor Bloom gate for descendant walks.
//!
//! v0.4 refinements:
//! - flat output (`MatchedStyles`): zero per-node heap allocations in the pass
//! - no per-node temp class vectors
//! - tiny per-node sort of the candidate hits keeps the ascending-order
//!   invariant (naive parity), ~10 elements

use crate::bloom::AncestorBloom;
use crate::naive::matches_compound;
use crate::parser::{Combinator, Selector};
use rustc_hash::FxHashMap;
use rustfox_dom::{Dom, NodeId};

/// Flat matched-styles: `indices[starts[i]..ends[i]]` are the selector
/// indices matched at node `i`, in ascending order.
#[derive(Debug, Default, Clone)]
pub struct MatchedStyles {
    pub starts: Vec<u32>,
    pub ends: Vec<u32>,
    pub indices: Vec<u32>,
}

impl MatchedStyles {
    pub fn for_node(&self, node: usize) -> &[u32] {
        &self.indices[self.starts[node] as usize..self.ends[node] as usize]
    }

    /// Convert to the nested form used by the naive/optimized baselines
    /// (used by parity tests).
    pub fn to_nested(&self) -> Vec<Vec<u32>> {
        (0..self.starts.len())
            .map(|i| self.for_node(i).to_vec())
            .collect()
    }
}

#[derive(Debug)]
pub struct StyleSet {
    pub selectors: Vec<Selector>,
    /// (subject tag, subject first class) -> selector indices.
    /// A subject with a tag goes under `Some(tag)`; a subject without a tag
    /// under `None`. Classes analogous (first class only; the rest are
    /// re-checked per candidate).
    buckets: FxHashMap<(Option<u32>, Option<u32>), Vec<u32>>,
}

impl StyleSet {
    pub fn new(selectors: Vec<Selector>) -> Self {
        let mut buckets: FxHashMap<(Option<u32>, Option<u32>), Vec<u32>> = FxHashMap::default();
        for (i, sel) in selectors.iter().enumerate() {
            let subj = sel.compounds.last().expect("selector has >=1 compound");
            let tag = subj.tag().map(|t| t.0);
            let first_class = subj.classes().next().map(|c| c.0);
            buckets
                .entry((tag, first_class))
                .or_default()
                .push(i as u32);
        }
        Self { selectors, buckets }
    }

    /// Full top-down pass with subject bucketing + bloom-gated descendant
    /// walks. Zero per-node allocations. `starts`/`ends` are indexed by `NodeId`.
    pub fn style_pass(&self, dom: &Dom) -> MatchedStyles {
        let mut out = MatchedStyles {
            starts: vec![0; dom.len()],
            ends: vec![0; dom.len()],
            indices: Vec::new(),
        };
        let mut bloom = AncestorBloom::new();

        fn visit(
            dom: &Dom,
            node: NodeId,
            set: &StyleSet,
            out: &mut MatchedStyles,
            bloom: &mut AncestorBloom,
        ) {
            let n = dom.node(node);
            let frame = bloom.push_node(n.tag.0, n.classes.iter().map(|c| c.0));

            let slot_start = out.indices.len();
            out.starts[node.0 as usize] = slot_start as u32;

            fn try_candidate(
                dom: &Dom,
                node: NodeId,
                set: &StyleSet,
                si: u32,
                bloom: &AncestorBloom,
                out: &mut MatchedStyles,
            ) {
                let sel = &set.selectors[si as usize];
                let subj = sel.compounds.last().unwrap();

                // Bucket guarantees the tag constraint; verify the remaining
                // subject classes against this node before any tree walk.
                for c in subj.classes() {
                    if !dom.node(node).classes.contains(&c) {
                        return;
                    }
                }
                if match_subject(dom, node, sel, bloom) {
                    out.indices.push(si);
                }
            }

            // Tag-specific buckets first, then tag-agnostic ones.
            for tag_key in [Some(n.tag.0), None] {
                if let Some(list) = set.buckets.get(&(tag_key, None)) {
                    for &si in list {
                        try_candidate(dom, node, set, si, bloom, out);
                    }
                }
                for &c in &n.classes {
                    if let Some(list) = set.buckets.get(&(tag_key, Some(c.0))) {
                        for &si in list {
                            try_candidate(dom, node, set, si, bloom, out);
                        }
                    }
                }
            }

            // Keep the ascending-order invariant (naive parity), tiny slice.
            out.indices[slot_start..].sort_unstable();
            out.ends[node.0 as usize] = out.indices.len() as u32;

            for &child in dom.children(node) {
                visit(dom, child, set, out, bloom);
            }
            bloom.pop_node(frame);
        }

        for root in dom.roots() {
            visit(dom, root, self, &mut out, &mut bloom);
        }
        out
    }
}

/// Match compounds[0..last] (subject already verified) walking ancestors.
fn match_subject(dom: &Dom, node: NodeId, sel: &Selector, bloom: &AncestorBloom) -> bool {
    fn walk(dom: &Dom, node: NodeId, sel: &Selector, idx: usize, bloom: &AncestorBloom) -> bool {
        if !matches_compound(dom, node, &sel.compounds[idx]) {
            return false;
        }
        if idx == 0 {
            return true;
        }
        let prev = &sel.compounds[idx - 1];
        if !bloom.may_match_compound(prev) {
            return false;
        }
        match sel.combinators[idx - 1] {
            Combinator::Child => dom
                .parent(node)
                .is_some_and(|p| walk(dom, p, sel, idx - 1, bloom)),
            Combinator::Descendant => {
                let mut anc = dom.parent(node);
                while let Some(a) = anc {
                    if walk(dom, a, sel, idx - 1, bloom) {
                        return true;
                    }
                    anc = dom.parent(a);
                }
                false
            }
        }
    }

    let last = sel.compounds.len() - 1;
    if last == 0 {
        return true;
    }
    // Subject compound matches by construction (bucket + class check);
    // start the chain at the subject's parent.
    let prev = &sel.compounds[last - 1];
    if !bloom.may_match_compound(prev) {
        return false;
    }
    match sel.combinators[last - 1] {
        Combinator::Child => dom
            .parent(node)
            .is_some_and(|p| walk(dom, p, sel, last - 1, bloom)),
        Combinator::Descendant => {
            let mut anc = dom.parent(node);
            while let Some(a) = anc {
                if walk(dom, a, sel, last - 1, bloom) {
                    return true;
                }
                anc = dom.parent(a);
            }
            false
        }
    }
}
