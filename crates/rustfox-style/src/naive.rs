//! Baseline (naive) matcher — the "before" state of the optimization loop.
//!
//! Left as a public, benchmarked implementation so CI can measure the
//! optimized matcher against it on every run: regressions and wins are
//! always visible relative to the honest baseline.

use crate::parser::{Combinator, Compound, Selector};
use rustfox_dom::{Dom, NodeId};

#[inline]
pub fn matches_compound(dom: &Dom, node: NodeId, compound: &Compound) -> bool {
    let n = dom.node(node);
    for simple in &compound.simples {
        let ok = match simple {
            crate::parser::Simple::Tag(t) => n.tag == *t,
            crate::parser::Simple::Class(c) => n.classes.contains(c),
            crate::parser::Simple::Universal => true,
        };
        if !ok {
            return false;
        }
    }
    true
}

/// Recursive right-to-left matching without any fast rejection.
fn match_from(dom: &Dom, node: NodeId, sel: &Selector, idx: usize) -> bool {
    if !matches_compound(dom, node, &sel.compounds[idx]) {
        return false;
    }
    if idx == 0 {
        return true;
    }
    match sel.combinators[idx - 1] {
        Combinator::Child => dom
            .parent(node)
            .is_some_and(|p| match_from(dom, p, sel, idx - 1)),
        Combinator::Descendant => {
            let mut anc = dom.parent(node);
            while let Some(a) = anc {
                if match_from(dom, a, sel, idx - 1) {
                    return true;
                }
                anc = dom.parent(a);
            }
            false
        }
    }
}

pub fn matches_selector(dom: &Dom, node: NodeId, sel: &Selector) -> bool {
    match_from(dom, node, sel, sel.compounds.len() - 1)
}

/// Match every selector against every node. `out[node]` = sorted selector indices.
pub fn style_pass(dom: &Dom, selectors: &[Selector]) -> Vec<Vec<u32>> {
    let mut out = vec![Vec::new(); dom.len()];
    for (i, slot) in out.iter_mut().enumerate() {
        let node = NodeId(i as u32);
        for (si, sel) in selectors.iter().enumerate() {
            if matches_selector(dom, node, sel) {
                slot.push(si as u32);
            }
        }
    }
    out
}
