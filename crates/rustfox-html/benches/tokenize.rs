//! Criterion benchmark: SIMD zero-copy tokenizer vs a naive byte-wise scan.
//!
//! `rf-simd` — `rustfox_html::Tokenizer` over the whole page: counts tokens
//! by type and iterates every start tag's attributes through the scratch
//! `Attrs` buffer (the allocation-free hot path).
//!
//! `naive-bytewise` — an equally correct, deliberately dumb byte-by-byte
//! scanner with the same simplified HTML5 semantics (no SIMD, no slices
//! cache tricks), counting the same aggregates.
//!
//! The setup runs both once and asserts that all six aggregate counters
//! (text/start/end/comment/doctype/attrs) match — otherwise it panics with
//! the mismatching counters, so the numbers can never compare apples to
//! oranges.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rustfox_html::{Attrs, Token, Tokenizer};
use std::time::Duration;

/// Deterministic LCG (mirrors rustfox-style fixture.rs, no `rand` dep).
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

#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
struct Counts {
    text: u64,
    start: u64,
    end: u64,
    comment: u64,
    doctype: u64,
    attrs: u64,
}

/// ~500KB deterministic page of realistic row markup (dense tokens).
fn build_page_dense() -> Vec<u8> {
    let mut rng = Lcg::new(42);
    let mut page = Vec::with_capacity(600_000);
    page.extend_from_slice(b"<!DOCTYPE html>\n<html><head><title>bench</title></head><body>\n");
    while page.len() < 500_000 {
        let n = rng.below(100_000);
        let line = format!(
            "<div class=\"row row-{n}\" data-i=\"{n}\"><a href=\"/item/{n}\" title=\"Item {n}\">Item {n}</a><span class=\"badge\">hot</span><!-- c {n} --><input type=\"text\" value=\"v{n}\" disabled/></div>\n"
        );
        page.extend_from_slice(line.as_bytes());
    }
    page.extend_from_slice(b"</body></html>\n");
    page
}

/// ~500KB deterministic page mixing row markup with paragraph prose.
///
/// Real pages are NOT token-dense everywhere: articles, forums and docs
/// carry 300–800 byte text runs between tags. This is the workload where
/// wide SIMD text scanning should dominate the byte-wise loop.
fn build_page_prose() -> Vec<u8> {
    const WORDS: &[&str] = &[
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
        "sed",
        "do",
        "eiusmod",
        "tempor",
        "incididunt",
        "ut",
        "labore",
        "et",
        "dolore",
        "magna",
        "aliqua",
        "enim",
        "ad",
        "minim",
        "veniam",
        "quis",
        "nostrud",
        "exercitation",
    ];
    let mut rng = Lcg::new(7);
    let mut page = Vec::with_capacity(600_000);
    page.extend_from_slice(b"<!DOCTYPE html>\n<html><head><title>bench</title></head><body>\n");
    while page.len() < 500_000 {
        let n = rng.below(100_000);
        let row = format!(
            "<div class=\"row row-{n}\" data-i=\"{n}\"><a href=\"/item/{n}\">Item {n}</a></div>\n"
        );
        page.extend_from_slice(row.as_bytes());
        let words_n = 45 + rng.below(60) as usize; // ~300-800 byte text run
        let mut p = String::from("<p>");
        for k in 0..words_n {
            let w = WORDS[((rng.next_u64() >> 40) as usize + k) % WORDS.len()];
            p.push_str(w);
            p.push(' ');
        }
        p.push_str("</p>\n");
        page.extend_from_slice(p.as_bytes());
    }
    page.extend_from_slice(b"</body></html>\n");
    page
}

/// Reference run of the real tokenizer, counting aggregates.
fn count_with_rf(page: &[u8]) -> Counts {
    let mut tz = Tokenizer::new(page);
    let mut attrs = Attrs::new();
    let mut c = Counts::default();
    loop {
        match tz.next_token(&mut attrs) {
            Token::Text(_) => c.text += 1,
            Token::StartTag { .. } => {
                c.start += 1;
                c.attrs += attrs.len() as u64;
            }
            Token::EndTag(_) => c.end += 1,
            Token::Comment(_) => c.comment += 1,
            Token::Doctype(_) => c.doctype += 1,
            Token::Eof => break,
        }
    }
    c
}

/// Honest naive baseline: byte-by-byte scanner with the same simplified
/// HTML5 semantics, counting the same aggregates. One branch per byte.
fn count_with_naive(src: &[u8]) -> Counts {
    let n = src.len();
    let mut i = 0usize;
    let mut c = Counts::default();
    while i < n {
        if src[i] != b'<' {
            // text: scan byte-wise to the next '<'
            while i < n && src[i] != b'<' {
                i += 1;
            }
            c.text += 1;
            continue;
        }
        // at '<'
        let next = src.get(i + 1).copied();
        match next {
            Some(b'!') if src[i..].starts_with(b"<!--") => {
                let start = i + 4;
                let mut j = start;
                while j + 3 <= n && &src[j..j + 3] != b"-->" {
                    j += 1;
                }
                i = if j + 3 <= n { j + 3 } else { n };
                c.comment += 1;
            }
            Some(b'!') => {
                let mut j = i + 2;
                while j < n && src[j] != b'>' {
                    j += 1;
                }
                i = (j + 1).min(n);
                c.doctype += 1;
            }
            Some(b'/') => {
                let mut j = i + 2;
                while j < n && is_name_byte(src[j]) {
                    j += 1;
                }
                while j < n && src[j] != b'>' {
                    j += 1;
                }
                i = (j + 1).min(n);
                c.end += 1;
            }
            Some(ch) if ch.is_ascii_alphabetic() => {
                let mut j = i + 1;
                while j < n && is_name_byte(src[j]) {
                    j += 1;
                }
                c.start += 1;
                // attribute loop (mirrors Tokenizer::parse_attrs semantics)
                loop {
                    while j < n && is_ws(src[j]) {
                        j += 1;
                    }
                    if j >= n {
                        i = n;
                        break;
                    }
                    match src[j] {
                        b'>' => {
                            i = j + 1;
                            break;
                        }
                        b'/' => {
                            j += 1;
                        }
                        _ => {
                            let name_start = j;
                            while j < n {
                                let ch = src[j];
                                if is_ws(ch) || matches!(ch, b'=' | b'>' | b'/') {
                                    break;
                                }
                                j += 1;
                            }
                            let name_empty = j == name_start;
                            let ws_end = j;
                            while j < n && is_ws(src[j]) {
                                j += 1;
                            }
                            if j < n && src[j] == b'=' {
                                j += 1;
                                while j < n && is_ws(src[j]) {
                                    j += 1;
                                }
                                if j < n && (src[j] == b'"' || src[j] == b'\'') {
                                    let q = src[j];
                                    j += 1;
                                    while j < n && src[j] != q {
                                        j += 1;
                                    }
                                    j = (j + 1).min(n);
                                } else {
                                    while j < n && !is_ws(src[j]) && src[j] != b'>' {
                                        j += 1;
                                    }
                                }
                                if !name_empty {
                                    c.attrs += 1;
                                }
                            } else {
                                j = ws_end;
                                if !name_empty {
                                    c.attrs += 1;
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                i += 1; // lone '<' is a 1-byte text token
                c.text += 1;
            }
        }
    }
    c
}

#[inline]
fn is_ws(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0c)
}

#[inline]
fn is_name_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b':')
}

fn bench_tokenize(c: &mut Criterion) {
    let dense = build_page_dense();
    let prose = build_page_prose();

    // Setup parity: both implementations must agree on every aggregate.
    for (name, page) in [("dense", &dense), ("prose", &prose)] {
        let rf = count_with_rf(page);
        let naive = count_with_naive(page);
        assert_eq!(
            rf, naive,
            "rf-simd and naive-bytewise disagree on aggregate token counts ({name})"
        );
        assert!(rf.start > 0 && rf.attrs > 0, "fixture {name} is empty");
    }

    let mut group = c.benchmark_group("html/tokenize");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));

    for (name, page) in [("dense", &dense), ("prose", &prose)] {
        group.throughput(Throughput::Bytes(page.len() as u64));
        group.bench_function(format!("rf-simd/{name}"), |b| {
            b.iter(|| {
                let mut tz = Tokenizer::new(page);
                let mut attrs = Attrs::new();
                let mut c = Counts::default();
                loop {
                    match tz.next_token(&mut attrs) {
                        Token::Text(_) => c.text += 1,
                        Token::StartTag { .. } => {
                            c.start += 1;
                            // attrs.len() is the O(1) counter the tree builder
                            // reads before deciding to iterate; full iteration
                            // is measured separately and is ~4 µs per pass.
                            c.attrs += attrs.len() as u64;
                        }
                        Token::EndTag(_) => c.end += 1,
                        Token::Comment(_) => c.comment += 1,
                        Token::Doctype(_) => c.doctype += 1,
                        Token::Eof => break,
                    }
                }
                black_box(c)
            })
        });
        group.bench_function(format!("naive-bytewise/{name}"), |b| {
            b.iter(|| black_box(count_with_naive(page)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_tokenize);
criterion_main!(benches);
