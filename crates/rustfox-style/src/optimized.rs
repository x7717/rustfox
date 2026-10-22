//! Optimized matcher — the current best state of the optimization loop.
//!
//! Structure is deliberately identical to `naive` so the benchmark delta is
//! attributable to the optimizations, not to different traversal order:
//!
//! 1. Right-to-left matching (subject compound first).
//! 2. **Ancestor Bloom filter gate** (see `bloom`): before ANY ancestor work
//!    for a compound, a single probe rejects the whole subtree walk when the
//!    required tag/class is definitely absent. Maintained incrementally
//!    during one top-down style pass, exactly like Stylo's per-frame filter.
//! 3. Interned ids everywhere: tag/class comparisons are integer equality.
//! 4. Cheap-first compound evaluation with early exit.

use crate::bloom::AncestorBloom;
use crate::naive::matches_compound;
use crate::parser::{Combinator, Selector};
use rustfox_dom::{Dom, NodeId};

fn match_from(dom: &Dom, node: NodeId, sel: &Selector, idx: usize, bloom: &AncestorBloom) -> bool {
    if !matches_compound(dom, node, &sel.compounds[idx]) {
        return false;
    }
    if idx == 0 {
        return true;
    }

    // Bloom gate: compounds[idx-1] must hold in some ancestor of the subject.
    // All nodes reached below are ancestors of the subject, so a negative
    // probe is authoritative for every remaining level.
    let prev = &sel.compounds[idx - 1];
    if !bloom.may_match_compound(prev) {
        return false;
    }

    match sel.combinators[idx - 1] {
        Combinator::Child => dom
            .parent(node)
            .is_some_and(|p| match_from(dom, p, sel, idx - 1, bloom)),
        Combinator::Descendant => {
            let mut anc = dom.parent(node);
            while let Some(a) = anc {
                if match_from(dom, a, sel, idx - 1, bloom) {
                    return true;
                }
                anc = dom.parent(a);
            }
            false
        }
    }
}

/// Optimized single-selector match for one node.
/// `bloom` must contain exactly the ancestors of `node`.
pub fn matches_selector(dom: &Dom, node: NodeId, sel: &Selector, bloom: &AncestorBloom) -> bool {
    match_from(dom, node, sel, sel.compounds.len() - 1, bloom)
}

/// One full top-down style pass: for every node, the sorted list of matching
/// selector indices. Maintains the ancestor bloom filter incrementally.
pub fn style_pass(dom: &Dom, selectors: &[Selector]) -> Vec<Vec<u32>> {
    let mut out = vec![Vec::new(); dom.len()];
    let mut bloom = AncestorBloom::new();

    fn visit(
        dom: &Dom,
        node: NodeId,
        selectors: &[Selector],
        out: &mut Vec<Vec<u32>>,
        bloom: &mut AncestorBloom,
    ) {
        let n = dom.node(node);
        let frame = bloom.push_node(n.tag.0, n.classes.iter().map(|c| c.0));

        let slot = &mut out[node.0 as usize];
        for (si, sel) in selectors.iter().enumerate() {
            if crate::optimized::matches_selector(dom, node, sel, bloom) {
                slot.push(si as u32);
            }
        }

        for &child in dom.children(node) {
            visit(dom, child, selectors, out, bloom);
        }
        bloom.pop_node(frame);
    }

    for root in dom.roots() {
        visit(dom, root, selectors, &mut out, &mut bloom);
    }
    out
}
