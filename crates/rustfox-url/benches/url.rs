//! Criterion benchmark: zero-copy URL parsing vs an honest naive baseline.
//!
//! `rf` — `rustfox_url::Url::parse` (SIMD splits, zero-copy, lazy host
//! lowercase) + extract `host().len() + path().len()`.
//! `naive` — a straightforward byte-by-byte parser with the *same component
//! semantics*, written inline below with plain loops and `str` splits only.
//!
//! Both extract identical (host, path) slices for every fixture URL; the
//! setup asserts byte equality so the comparison stays honest. If a fixture
//! URL were to hit a semantic corner the naive mirror does not implement,
//! the parity check below skips it (currently: none — all four fixture
//! families are within the shared semantics).

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rustfox_url::Url;
use std::time::Duration;

/// Deterministic LCG (mirrors rustfox-style fixture.rs, no `rand` dep).
/// High bits only: low bits of an LCG have short periods.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x2545F4914F6CDD1D) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        (self.next_u64() >> 32) % n
    }
}

/// 2000 mixed URLs: long-domain https, http with userinfo+port+query+
/// fragment+UTF-8, ftp, file — the shapes nsStandardURL sees in the wild.
fn build_urls() -> Vec<String> {
    let mut rng = Lcg::new(0xBEEF_CAFE);
    let mut out = Vec::with_capacity(2000);
    for _i in 0..2000u64 {
        let n = rng.below(100_000);
        let m = rng.below(64);
        let k = rng.below(10_000);
        let url = match rng.below(4) {
            0 => format!(
                "https://www.subdomain-{n}.mirror-{m}.example-site.co.uk/docs/view/{k}/page-{n}.html?ref=search&q=term-{k}&lang=en#section-{m}"
            ),
            1 => format!(
                "http://admin:s3cret-{k}@svc-{n}.internal.example:808{m}/api/v2/записи/{n}?фильтр=активные&лимит={k}&страница={m}#итог-{n}"
            ),
            2 => format!("ftp://files.example.com/pub/mirror-{m}/archive-{n}.tar.gz"),
            _ => format!("file:///home/dev/project-{n}/src/lib/module-{k}.rs"),
        };
        out.push(url);
    }
    out
}

/// Honest naive baseline: byte-by-byte parsing, same semantics as
/// `rustfox_url::Url` for (host, path). Returns `Err(())` on any parse
/// failure (mirrored loosely — the parity check only compares successes).
fn naive_parse(src: &str) -> Result<(&str, &str), ()> {
    let b = src.as_bytes();
    // scheme: ALPHA *(ALPHA/DIGIT/+/-/.) ':'
    if b.is_empty() || !b[0].is_ascii_alphabetic() {
        return Err(());
    }
    let mut i = 1;
    while i < b.len() && b[i] != b':' {
        if !(b[i].is_ascii_alphanumeric() || matches!(b[i], b'+' | b'-' | b'.')) {
            return Err(());
        }
        i += 1;
    }
    if i >= b.len() {
        return Err(()); // no ':'
    }
    let after = i + 1;
    let file = src[..i].eq_ignore_ascii_case("file");

    let hierarchical = b[after..].starts_with(b"//");
    let mut pos = if hierarchical { after + 2 } else { after };

    let mut host = "";
    let mut path_start = pos;

    if hierarchical {
        // authority ends at the first '/', '?' or '#'
        let mut auth_end = b.len();
        for (j, &c) in b[pos..].iter().enumerate() {
            if matches!(c, b'/' | b'?' | b'#') {
                auth_end = pos + j;
                break;
            }
        }
        // userinfo: the LAST '@' wins
        let mut hp = (pos, auth_end);
        for j in (pos..auth_end).rev() {
            if b[j] == b'@' {
                hp = (j + 1, auth_end);
                break;
            }
        }
        // host[:port]
        if hp.0 >= hp.1 {
            if !file {
                return Err(());
            }
        } else if b[hp.0] == b'[' {
            let close = (hp.0..hp.1).find(|&j| b[j] == b']').ok_or(())?;
            host = &src[hp.0..=close];
            if close + 1 < hp.1 {
                if b[close + 1] != b':' {
                    return Err(());
                }
                for &d in &b[close + 2..hp.1] {
                    if !d.is_ascii_digit() {
                        return Err(());
                    }
                }
            }
        } else {
            let mut colon = None;
            for j in (hp.0..hp.1).rev() {
                if b[j] == b':' {
                    colon = Some(j);
                    break;
                }
            }
            match colon {
                Some(c) => {
                    host = &src[hp.0..c];
                    for &d in &b[c + 1..hp.1] {
                        if !d.is_ascii_digit() {
                            return Err(());
                        }
                    }
                }
                None => host = &src[hp.0..hp.1],
            }
            if host.is_empty() && !file {
                return Err(());
            }
        }
        pos = auth_end;
        path_start = pos;
    }

    // path ends at '?' or '#'
    let mut path_end = b.len();
    for (j, &c) in b[path_start..].iter().enumerate() {
        if c == b'?' || c == b'#' {
            path_end = path_start + j;
            break;
        }
    }
    let path = if hierarchical && path_start >= path_end {
        "/"
    } else {
        &src[path_start..path_end]
    };
    Ok((host, path))
}

fn bench_url(c: &mut Criterion) {
    let urls = build_urls();
    let total: u64 = urls.iter().map(|u| u.len() as u64).sum();

    // Setup parity assertion: rf and naive must extract identical
    // host/path for every URL where both succeed. A URL where only one
    // side succeeds would be skipped and documented here; currently
    // unreachable — all fixture families share semantics.
    let mut compared = 0usize;
    for u in &urls {
        if let (Ok(rf), Ok((nh, np))) = (Url::parse(u), naive_parse(u)) {
            assert_eq!(rf.host(), nh, "host mismatch on {u}");
            assert_eq!(rf.path(), np, "path mismatch on {u}");
            compared += 1;
        }
    }
    assert_eq!(
        compared,
        urls.len(),
        "fixture must be fully covered by both parsers"
    );

    let mut group = c.benchmark_group("url/parse");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.throughput(Throughput::Bytes(total));

    group.bench_function("rf", |b| {
        b.iter(|| {
            for u in &urls {
                let url = Url::parse(u).unwrap();
                black_box(url.host().len() + url.path().len());
            }
        })
    });

    group.bench_function("naive", |b| {
        b.iter(|| {
            for u in &urls {
                let (h, p) = naive_parse(u).unwrap();
                black_box(h.len() + p.len());
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_url);
criterion_main!(benches);
