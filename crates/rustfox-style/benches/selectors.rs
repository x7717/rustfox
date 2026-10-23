//! Criterion benchmarks: the optimization loop's measurement instrument.
//!
//! `naive` vs `optimized` run side by side on identical fixtures, so every
//! CI run quantifies the current win over the honest baseline.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rustfox_dom::Dom;
use rustfox_style::{
    fast::{FastStyleSet, SoaCache},
    fixture, naive, optimized, parse_all, StyleSet,
};

fn build_fixture(dom: &mut Dom, nodes: usize) -> Vec<rustfox_style::Selector> {
    let sources = fixture::build_selectors(200, 42);
    fixture::build_dom(dom, nodes, 12, 42);
    parse_all(dom, &sources)
}

fn bench_style_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("style_pass");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_millis(400));

    for size in [2_000usize, 10_000] {
        let mut dom = Dom::new();
        let selectors = build_fixture(&mut dom, size);

        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &(&dom, &selectors),
            |b, (dom, sels)| b.iter(|| naive::style_pass(dom, sels)),
        );
        group.bench_with_input(
            BenchmarkId::new("optimized", size),
            &(&dom, &selectors),
            |b, (dom, sels)| b.iter(|| optimized::style_pass(dom, sels)),
        );
        group.bench_with_input(
            BenchmarkId::new("indexed", size),
            &(&dom, &selectors),
            |b, (dom, sels)| {
                let set = StyleSet::new(sels.to_vec());
                b.iter(|| set.style_pass(dom))
            },
        );
        group.bench_with_input(
            BenchmarkId::new("fast", size),
            &(&dom, &selectors),
            |b, (dom, sels)| {
                let set = FastStyleSet::new(sels.to_vec());
                // steady-state restyle: SoA snapshot built once, reused —
                // the browser hot path (style recalc on an unchanged tree)
                let cache = SoaCache::of(dom);
                b.iter(|| set.style_pass(&cache))
            },
        );
        group.bench_with_input(
            BenchmarkId::new("fast_okc", size),
            &(&dom, &selectors),
            |b, (dom, sels)| {
                let set = FastStyleSet::new(sels.to_vec());
                let cache = SoaCache::of(dom);
                // v0.7 experiment: direct-mapped (tag, mask) -> ok cache
                b.iter(|| set.style_pass_cached(&cache))
            },
        );
    }
    group.finish();
}

fn bench_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_millis(300));

    let sources = fixture::build_selectors(200, 42);
    group.bench_function("200_selectors", |b| {
        b.iter_batched(
            Dom::new,
            |mut dom| parse_all(&mut dom, &sources),
            criterion::BatchSize::SmallInput,
        )
    });
    group.finish();
}

/// LCG for deterministic dirty sets (fixture-style, no rand dependency).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

/// v0.8: incremental restyle — the steady-state hot path (class toggles on
/// a live page) vs the full recalc it replaces. Each iteration flips a
/// fixture-vocabulary class on every dirty node (real mask churn) and runs
/// the restyle; the dirty sets are LCG-fixed per fraction so every sample
/// measures identical work.
fn bench_restyle(c: &mut Criterion) {
    let mut group = c.benchmark_group("restyle");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_millis(500));

    let size = 10_000usize;
    let mut dom = Dom::new();
    let selectors = build_fixture(&mut dom, size);
    let set = FastStyleSet::new(selectors);
    let mut cache = SoaCache::of(&dom);
    let n = dom.len();

    // Baseline for the ratios: one full recalc.
    group.bench_function("full_10000", |b| b.iter(|| set.style_pass(&cache)));

    // Setup parity: restyle output == fresh full pass after a real toggle
    // round (asserted once here; the bench loop repeats the same work).
    let (mut prev, mut states) = set.style_pass_states(&cache);

    for &pct in &[0.1f64, 1.0, 5.0, 10.0, 25.0] {
        let k = (((n as f64) * pct / 100.0).round() as usize).max(1);
        let mut rng = Lcg(0x9E37_79B9_7F4A_7C15 ^ (k as u64));
        let mut dirty = Vec::with_capacity(k);
        let mut seen = vec![false; n];
        while dirty.len() < k {
            let id = (rng.next() >> 32) as usize % n;
            if !seen[id] {
                seen[id] = true;
                dirty.push(id as u32);
            }
        }
        // Fixture class vocabulary is c0..c63; toggle bit (k % 64) on the
        // dirty nodes once, then assert parity of the incremental path.
        // `prev` CHAINS across fractions (each round's output feeds the next),
        // exactly like a live page — a stale prev would flag phantom
        // mismatches on nodes dirtied by earlier fractions.
        let class = (k % 64) as u32;
        for &d in &dirty {
            cache.set_class(d as usize, class, true);
        }
        let out = set.restyle(&cache, &mut states, &prev, &dirty);
        let fresh = set.style_pass(&cache);
        // Compact parity report: first mismatching node only (to_nested()
        // would format 10k nested vecs into the panic message).
        let (a, b) = (out.to_nested(), fresh.to_nested());
        assert_eq!(a.len(), b.len(), "restyle/fresh len mismatch at {pct}%");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x, y, "restyle != full pass at {pct}%: node {i}");
        }
        prev = out;

        group.bench_function(format!("dirty_{pct}pct_10000"), |b| {
            b.iter(|| set.restyle(&cache, &mut states, &prev, &dirty))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_style_pass, bench_parse, bench_restyle);
criterion_main!(benches);
