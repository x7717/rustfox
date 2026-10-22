//! End-to-end correctness: optimized == naive, always.

use rustfox_dom::Dom;
use rustfox_style::{fixture, naive, optimized, parse_all};

#[test]
fn parity_small_and_large() {
    for (nodes, sels, seed) in [
        (300usize, 40usize, 1u64),
        (2_000, 120, 1234),
        (5_000, 200, 99),
    ] {
        let mut dom = Dom::new();
        fixture::build_dom(&mut dom, nodes, 12, seed);
        let sources = fixture::build_selectors(sels, seed ^ 0xBEEF);
        let selectors = parse_all(&mut dom, &sources);

        let a = naive::style_pass(&dom, &selectors);
        let b = optimized::style_pass(&dom, &selectors);
        assert_eq!(
            a, b,
            "parity broken for nodes={nodes} sels={sels} seed={seed}"
        );
    }
}

#[test]
fn bloom_gate_preserves_semantics() {
    // Selector whose ancestor compound references ids that exist in the DOM
    // but never among the ancestors of the subject — bloom must not change
    // the result, only the speed.
    let mut dom = Dom::new();
    fixture::build_dom(&mut dom, 1_000, 10, 7);
    let sources = vec![
        "html div div .c1".to_string(),
        "html > header > ul > li > span".to_string(),
        ".root * .c63".to_string(),
        "section.header div.c5 p".to_string(),
    ];
    let selectors = parse_all(&mut dom, &sources);
    assert_eq!(
        naive::style_pass(&dom, &selectors),
        optimized::style_pass(&dom, &selectors)
    );
}
