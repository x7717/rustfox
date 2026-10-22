//! Criterion benchmarks: the optimization loop's measurement instrument.
//!
//! `naive` vs `optimized` run side by side on identical fixtures, so every
//! CI run quantifies the current win over the honest baseline.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rustfox_dom::Dom;
use rustfox_style::{fixture, naive, optimized, parse_all, StyleSet};

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

criterion_group!(benches, bench_style_pass, bench_parse);
criterion_main!(benches);
