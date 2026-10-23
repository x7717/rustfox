//! Ancestor bloom filter — the classic Stylo optimization.
//!
//! While traversing the DOM top-down we maintain a Bloom filter containing
//! the tag ids and class ids of every ancestor on the current stack.
//! Before walking the ancestor chain to satisfy an ancestor-position
//! compound we consult the filter: if any id required by that compound is
//! *definitely absent* (its bit is 0), the whole ancestor walk is skipped.
//!
//! Bloom filters never produce false negatives, so correctness is preserved;
//! false positives merely cost a walk that would have happened anyway.

use crate::parser::Compound;

/// 32 words × 64 bits = 2048 bits.
pub const WORDS: usize = 32;
const BITS: u64 = (WORDS * 64) as u64;

/// Full Copy snapshot of the filter state (256 bytes, no heap).
pub type BloomSnapshot = [u64; WORDS];

#[inline]
fn bit_for(id: u32) -> (usize, u64) {
    // Fibonacci hashing on the id, then fold into the bit space.
    let h = (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let bit = ((h >> 13) % BITS) as usize;
    (bit / 64, 1u64 << (bit % 64))
}

#[derive(Debug, Default)]
pub struct AncestorBloom {
    words: [u64; WORDS],
}

/// Saved word states for one node, so `pop` can restore exactly.
#[derive(Debug, Default)]
pub struct Frame {
    saved: Vec<(usize, u64)>,
}

impl AncestorBloom {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn set(&mut self, id: u32, frame: &mut Frame) {
        let (wi, mask) = bit_for(id);
        if self.words[wi] & mask == 0 {
            frame.saved.push((wi, self.words[wi]));
            self.words[wi] |= mask;
        }
    }

    /// Insert one id directly (v0.5 snapshot API; no per-node `Frame` alloc).
    #[inline]
    pub fn insert(&mut self, id: u32) {
        let (wi, mask) = bit_for(id);
        self.words[wi] |= mask;
    }

    /// Copy the whole filter state out (256 bytes on the stack).
    #[inline]
    pub fn snapshot(&self) -> BloomSnapshot {
        self.words
    }

    /// Restore a previously taken snapshot.
    #[inline]
    pub fn restore(&mut self, snap: BloomSnapshot) {
        self.words = snap;
    }

    /// Gate check against precomputed compound data: tag + class bitmask
    /// (ids < 128) + spilled high-id classes. Never rejects a matchable
    /// compound (no false negatives).
    pub fn may_match_info(&self, tag: Option<u32>, mask: u128, hi: &[u32]) -> bool {
        if let Some(t) = tag {
            if !self.contains(t) {
                return false;
            }
        }
        let mut m = mask;
        while m != 0 {
            if !self.contains(m.trailing_zeros()) {
                return false;
            }
            m &= m - 1;
        }
        hi.iter().all(|&c| self.contains(c))
    }

    /// Push one node: insert its tag and class bits. Returns a frame for `pop`.
    pub fn push_node(&mut self, tag: u32, classes: impl IntoIterator<Item = u32>) -> Frame {
        let mut frame = Frame::default();
        self.set(tag, &mut frame);
        for c in classes {
            self.set(c, &mut frame);
        }
        frame
    }

    /// Restore the state from before the matching `push_node`.
    pub fn pop_node(&mut self, frame: Frame) {
        for (wi, old) in frame.saved {
            self.words[wi] = old;
        }
    }

    #[inline]
    fn contains(&self, id: u32) -> bool {
        let (wi, mask) = bit_for(id);
        self.words[wi] & mask != 0
    }

    /// Low-level check: could any ancestor hold this tag / these classes?
    /// Zero-allocation: takes an iterator of class ids.
    pub fn may_match_ids(&self, tag: Option<u32>, classes: impl IntoIterator<Item = u32>) -> bool {
        if let Some(t) = tag {
            if !self.contains(t) {
                return false;
            }
        }
        for c in classes {
            if !self.contains(c) {
                return false;
            }
        }
        true
    }

    /// Could any ancestor match this compound?
    /// If a required tag/class is definitely absent from all ancestors,
    /// the compound cannot match and the caller skips the walk.
    pub fn may_match_compound(&self, compound: &Compound) -> bool {
        self.may_match_ids(compound.tag().map(|t| t.0), compound.classes().map(|c| c.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_selector;
    use rustfox_dom::Dom;

    #[test]
    fn no_false_negatives() {
        let mut bloom = AncestorBloom::new();
        let frame = bloom.push_node(7, [10u32, 20, 30]);
        assert!(bloom.may_match_ids(Some(7), [20]));
        assert!(bloom.may_match_ids(Some(30), []));
        assert!(bloom.may_match_ids(None, [10, 20, 30]));
        // definitely-absent ids must be rejected
        assert!(!bloom.may_match_ids(Some(8), []));
        assert!(!bloom.may_match_ids(None, [21]));
        bloom.pop_node(frame);
        assert!(!bloom.may_match_ids(Some(7), []));
    }

    #[test]
    fn compound_gate_never_rejects_matchable() {
        let mut dom = Dom::new();
        let _ = dom.intern_tag("div");
        let c = dom.intern_class("c1");
        let mut bloom = AncestorBloom::new();
        let frame = bloom.push_node(dom.intern_tag("div").0, [c.0]);
        let sel = parse_selector("div.c1", &mut dom).unwrap();
        assert!(bloom.may_match_compound(&sel.compounds[0]));
        bloom.pop_node(frame);
    }
}
