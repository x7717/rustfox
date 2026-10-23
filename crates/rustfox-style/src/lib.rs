//! rustfox-style: CSS selector parsing + matching hot path.
//!
//! Two implementations live side by side and are benchmarked against each
//! other on every CI run:
//! - `naive`:     honest baseline, no fast rejection
//! - `optimized`: right-to-left matching + ancestor Bloom filter + interned ids
//!
//! `tests::parity` guarantees both produce identical match sets on the
//! synthetic fixture, so the benchmark delta is pure speed, not semantics.

pub mod bloom;
pub mod fast;
pub mod fixture;
pub mod indexed;
pub mod naive;
pub mod optimized;
pub mod parser;

pub use fast::FastStyleSet;
pub use indexed::StyleSet;
pub use parser::{parse_selector, Combinator, Compound, ParseError, Selector, Simple};
use rustfox_dom::Dom;

/// Parse a list of selector strings against `dom` (interns names).
pub fn parse_all(dom: &mut Dom, sources: &[String]) -> Vec<Selector> {
    sources
        .iter()
        .map(|s| parse_selector(s, dom).expect("fixture selector must parse"))
        .collect()
}

#[cfg(test)]
mod parity {
    use super::*;

    #[test]
    fn optimized_matches_naive_on_fixture() {
        let mut dom = Dom::new();
        fixture::build_dom(&mut dom, 2_000, 10, 1234);
        let sources = fixture::build_selectors(120, 777);
        let selectors = parse_all(&mut dom, &sources);

        let a = naive::style_pass(&dom, &selectors);
        let b = optimized::style_pass(&dom, &selectors);
        assert_eq!(a.len(), dom.len());
        assert_eq!(b.len(), dom.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x, y, "mismatch at node {i}: naive={x:?} optimized={y:?}");
        }
    }

    #[test]
    fn indexed_matches_naive_on_fixture() {
        let mut dom = Dom::new();
        fixture::build_dom(&mut dom, 2_000, 10, 1234);
        let sources = fixture::build_selectors(120, 777);
        let selectors = parse_all(&mut dom, &sources);

        let a = naive::style_pass(&dom, &selectors);
        let b = indexed::StyleSet::new(selectors)
            .style_pass(&dom)
            .to_nested();
        assert_eq!(a.len(), dom.len());
        assert_eq!(b.len(), dom.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x, y, "mismatch at node {i}: naive={x:?} indexed={y:?}");
        }
    }

    #[test]
    fn fast_matches_naive_on_fixture() {
        let mut dom = Dom::new();
        fixture::build_dom(&mut dom, 2_000, 10, 1234);
        let sources = fixture::build_selectors(200, 777);
        let selectors = parse_all(&mut dom, &sources);

        let a = naive::style_pass(&dom, &selectors);
        let b = fast::FastStyleSet::new(selectors)
            .style_pass_dom(&dom)
            .to_nested();
        assert_eq!(a.len(), dom.len());
        assert_eq!(b.len(), dom.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x, y, "mismatch at node {i}: naive={x:?} fast={y:?}");
        }
    }

    #[test]
    fn fast_cached_matches_naive_on_fixture() {
        // v0.7: level-order banks + AVX2 transition + ok-cache must stay
        // byte-identical to the naive baseline.
        let mut dom = Dom::new();
        fixture::build_dom(&mut dom, 2_000, 10, 1234);
        let sources = fixture::build_selectors(200, 777);
        let selectors = parse_all(&mut dom, &sources);

        let a = naive::style_pass(&dom, &selectors);
        let set = fast::FastStyleSet::new(selectors);
        let b = set.style_pass_cached(&fast::SoaCache::of(&dom)).to_nested();
        assert_eq!(a.len(), dom.len());
        assert_eq!(b.len(), dom.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x, y, "mismatch at node {i}: naive={x:?} fast_cached={y:?}");
        }
    }

    #[test]
    fn fast_handles_high_class_ids_and_tricky_subjects() {
        // Classes with id >= 128 exercise the spill path; universal and
        // classless subjects exercise the `by_class`/`universal` buckets.
        let mut dom = Dom::new();
        let root = dom.create_element("div", &["c200"]);
        let mid = dom.create_element("span", &["c130", "c5"]);
        let leaf = dom.create_element("p", &[]);
        let star = dom.create_element("li", &["c9"]);
        dom.append(root, mid);
        dom.append(mid, leaf);
        dom.append(mid, star);

        let sources = vec![
            "*".to_string(),
            "* > p".to_string(),
            ".c130 p".to_string(),
            "div .c130 > p".to_string(),
            ".c200 .c5".to_string(),
            "span.c130.c5 > *".to_string(),
            ".c200 > span".to_string(),
            "li".to_string(),
            "div > span .c9".to_string(),
        ];
        let selectors = parse_all(&mut dom, &sources);

        let a = naive::style_pass(&dom, &selectors);
        let b = fast::FastStyleSet::new(selectors)
            .style_pass_dom(&dom)
            .to_nested();
        assert_eq!(a, b);
        let total: usize = a.iter().map(|v| v.len()).sum();
        assert!(total > 0);
    }

    #[test]
    fn nonempty_results() {
        let mut dom = Dom::new();
        fixture::build_dom(&mut dom, 500, 8, 5);
        let sources = vec!["div".to_string(), ".c1 > .c2".to_string()];
        let selectors = parse_all(&mut dom, &sources);
        let a = naive::style_pass(&dom, &selectors);
        let b = optimized::style_pass(&dom, &selectors);
        assert_eq!(a, b);
        let total: usize = a.iter().map(|v| v.len()).sum();
        assert!(total > 0, "fixture should match something");
    }
}
