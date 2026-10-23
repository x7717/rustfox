//! Criterion benchmarks for the scanning primitives.
//!
//! Naive byte-loop baseline vs the runtime-dispatched SIMD path, on
//! haystacks chosen to expose both the "needle absent" (full scan) and
//! "needle at boundary" cases that dominate real tokenizer/parser work.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rustfox_simd::{find_any2, find_byte};
use std::hint::black_box;

/// Deterministic pseudo-random haystack (LCG, same pattern as rustfox-style
/// fixtures) with a guaranteed-unique needle placement.
fn haystack(len: usize, needle_pos: Option<usize>) -> (Vec<u8>, u8) {
    let mut x: u32 = 0x1234_5678;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let mut b = (x >> 24) as u8;
        if b == b'<' {
            b = b'x';
        }
        v.push(b);
    }
    if let Some(p) = needle_pos {
        v[p] = b'<';
    }
    (v, b'<')
}

fn naive_find(hay: &[u8], n: u8) -> Option<usize> {
    hay.iter().position(|&b| b == n)
}

fn naive_any2(hay: &[u8], a: u8, b: u8) -> Option<usize> {
    hay.iter().position(|&c| c == a || c == b)
}

fn bench_find(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/find_byte");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_millis(300));

    for (name, len, pos) in [("4k_absent", 4096, None), ("4k_last", 4096, Some(4095))] {
        let (hay, n) = haystack(len, pos);
        group.throughput(Throughput::Bytes(len as u64));
        group.bench_function(format!("simd/{name}"), |b| {
            b.iter(|| black_box(find_byte(black_box(&hay), n)))
        });
        group.bench_function(format!("naive/{name}"), |b| {
            b.iter(|| black_box(naive_find(black_box(&hay), n)))
        });
    }
    group.finish();
}

fn bench_any2(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd/find_any2");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_millis(300));

    let (hay, _) = haystack(4096, None);
    group.throughput(Throughput::Bytes(hay.len() as u64));
    group.bench_function("simd/4k_absent", |b| {
        b.iter(|| black_box(find_any2(black_box(&hay), b'?', b'#')))
    });
    group.bench_function("naive/4k_absent", |b| {
        b.iter(|| black_box(naive_any2(black_box(&hay), b'?', b'#')))
    });
    group.finish();
}

criterion_group!(benches, bench_find, bench_any2);
criterion_main!(benches);
