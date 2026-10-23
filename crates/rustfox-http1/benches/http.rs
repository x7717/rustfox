//! Criterion benchmark: zero-copy HTTP/1.1 head parsing vs the std-string
//! way (`str::from_utf8` + `find` + `split` + `splitn` + `trim` + `parse`).
//!
//! `rf` — loops `Response::parse` over the concatenated buffer, XOR-folds
//! statuses and reads `content-length` (case-insensitive, zero-copy).
//! `rf-simd-call` — the v0.8 scanning path kept as an in-binary A/B
//! control: a byte-for-byte copy of the old parser (one
//! `rustfox_simd::find_byte` call per line/colon). This box is a noisy
//! 2-vCPU sandbox, so scanning variants are compared inside one process,
//! seconds apart, instead of across runs.
//! `naive-str` — the honest "just use strings" baseline implemented with
//! the std library only: one `from_utf8` over the buffer, per-response
//! `str::find("\r\n\r\n")`, `split("\r\n")`, `splitn(':')`, `trim()`,
//! `parse::<u16>()`.
//!
//! Setup asserts all three parse the same number of responses and produce
//! the same status sum — otherwise panic with the numbers.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rustfox_http1::{Header, HeaderList, HttpError, Response};
use rustfox_simd::find_byte;
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

const N_RESPONSES: usize = 5000;

/// 5000 synthetic responses concatenated into one buffer.
fn build_buffer() -> Vec<u8> {
    let mut rng = Lcg::new(0xDEAD_10CC);
    let mut buf = Vec::with_capacity(1_600_000);
    for _ in 0..N_RESPONSES {
        let n = rng.below(1_000_000);
        let m = rng.below(1000);
        let (code, reason) = match rng.below(5) {
            0 => ("200", "OK"),
            1 => ("204", "No Content"),
            2 => ("301", "Moved Permanently"),
            3 => ("404", "Not Found"),
            _ => ("500", "Internal Server Error"),
        };
        let head = format!(
            "HTTP/1.1 {code} {reason}\r\n\
             Server: nginx/1.24.{}\r\n\
             Content-Type: text/html; charset=utf-8\r\n\
             Content-Length: {}\r\n\
             Cache-Control: max-age={}\r\n\
             Set-Cookie: sid={n:x}{m:x}; Path=/; HttpOnly\r\n\
             X-Trace: trace-{n}-{m}\r\n\r\n",
            rng.below(9),
            512 + n % 4096,
            rng.below(86_400),
        );
        buf.extend_from_slice(head.as_bytes());
    }
    buf
}

/// rf reference pass: count + status sum + content-length total.
fn run_rf(buf: &[u8]) -> (usize, u64, u64) {
    let mut rest = buf;
    let mut count = 0usize;
    let mut sum = 0u64;
    let mut clen_total = 0u64;
    while !rest.is_empty() {
        let (resp, used) = Response::parse(rest)
            .unwrap_or_else(|e| panic!("rf failed on response {count}: {e:?}"));
        count += 1;
        sum += resp.status as u64;
        clen_total += resp.header(b"content-length").map_or(0, |v| v.len()) as u64;
        rest = &rest[used..];
    }
    (count, sum, clen_total)
}

// ---------------------------------------------------------------------------
// v0.8 scanning path, preserved verbatim as an in-binary A/B control
// (`rf-simd-call`). Response has no public constructor, so this returns
// (status, headers, head_len); the bench consumes them identically.
// ---------------------------------------------------------------------------

#[inline]
fn ows(c: u8) -> bool {
    c == b' ' || c == b'\t'
}

fn head_len_v08(src: &[u8]) -> Option<usize> {
    let mut from = 0usize;
    loop {
        let i = find_byte(&src[from..], b'\n')? + from;
        if i >= 3 && src[i - 1] == b'\r' && src[i - 2] == b'\n' && src[i - 3] == b'\r' {
            return Some(i + 1);
        }
        from = i + 1;
    }
}

fn parse_v08(src: &[u8]) -> Result<(u16, HeaderList<'_>, usize), HttpError> {
    let head_len = head_len_v08(src).ok_or(HttpError::NoTerminator)?;
    let head = &src[..head_len];

    let nl = find_byte(head, b'\n').ok_or(HttpError::BadStatus)?;
    let mut le = nl;
    if le > 0 && head[le - 1] == b'\r' {
        le -= 1;
    }
    let line = &head[..le];
    let sp = find_byte(line, b' ').ok_or(HttpError::BadStatus)?;
    let rest = &line[sp + 1..];
    if rest.len() < 3 {
        return Err(HttpError::BadStatus);
    }
    let d = &rest[..3];
    if !d.iter().all(|b| b.is_ascii_digit()) {
        return Err(HttpError::BadStatus);
    }
    let status = ((d[0] - b'0') as u16) * 100 + ((d[1] - b'0') as u16) * 10 + (d[2] - b'0') as u16;
    let after = &rest[3..];
    if !after.is_empty() && after[0] != b' ' {
        return Err(HttpError::BadStatus);
    }

    let mut headers = HeaderList::new();
    let mut pos = nl + 1;
    while pos < head_len {
        let nl2 = find_byte(&head[pos..], b'\n').ok_or(HttpError::BadHeader)? + pos;
        let mut le2 = nl2;
        if le2 > pos && head[le2 - 1] == b'\r' {
            le2 -= 1;
        }
        let hline = &head[pos..le2];
        pos = nl2 + 1;
        if hline.is_empty() {
            break;
        }
        let colon = find_byte(hline, b':').ok_or(HttpError::BadHeader)?;
        let mut ne = colon;
        while ne > 0 && ows(hline[ne - 1]) {
            ne -= 1;
        }
        let mut ns = 0;
        while ns < ne && ows(hline[ns]) {
            ns += 1;
        }
        if ns >= ne {
            return Err(HttpError::BadHeader);
        }
        let mut vs = colon + 1;
        while vs < hline.len() && ows(hline[vs]) {
            vs += 1;
        }
        let mut ve = hline.len();
        while ve > vs && ows(hline[ve - 1]) {
            ve -= 1;
        }
        headers.push(Header {
            name: &hline[ns..ne],
            value: &hline[vs..ve],
        });
    }
    Ok((status, headers, head_len))
}

/// v0.8-path reference pass (same consumption as `run_rf`).
fn run_v08(buf: &[u8]) -> (usize, u64, u64) {
    let mut rest = buf;
    let mut count = 0usize;
    let mut sum = 0u64;
    let mut clen_total = 0u64;
    while !rest.is_empty() {
        let (status, headers, used) =
            parse_v08(rest).unwrap_or_else(|e| panic!("v08 failed on response {count}: {e:?}"));
        count += 1;
        sum += status as u64;
        clen_total += headers.get(b"content-length").map_or(0, |v| v.len()) as u64;
        rest = &rest[used..];
    }
    (count, sum, clen_total)
}

/// Honest naive baseline: std string machinery, byte-scan via str::find.
fn run_naive(buf: &[u8]) -> (usize, u64, u64) {
    let s = std::str::from_utf8(buf).expect("fixture is utf-8");
    let mut rest = s;
    let mut count = 0usize;
    let mut sum = 0u64;
    let mut clen_total = 0u64;
    while !rest.is_empty() {
        let head_end = rest
            .find("\r\n\r\n")
            .unwrap_or_else(|| panic!("naive: no terminator at response {count}"))
            + 4;
        let head = &rest[..head_end];
        let mut lines = head.split("\r\n");
        let status_line = lines.next().expect("status line");
        let mut parts = status_line.splitn(3, ' ');
        let _version = parts.next().expect("version");
        let status: u16 = parts
            .next()
            .expect("status")
            .parse()
            .unwrap_or_else(|e| panic!("naive: bad status at {count}: {e}"));
        let _reason = parts.next().unwrap_or("");
        count += 1;
        sum += status as u64;
        for line in lines {
            if line.is_empty() {
                continue;
            }
            let (name, value) = line
                .split_once(':')
                .unwrap_or_else(|| panic!("naive: header without colon at {count}"));
            if name.trim().eq_ignore_ascii_case("content-length") {
                clen_total += value.trim().len() as u64;
            }
        }
        rest = &rest[head_end..];
    }
    (count, sum, clen_total)
}

fn bench_http(c: &mut Criterion) {
    let buf = build_buffer();

    // Setup parity: identical counts and status sums across all variants.
    let (rc, rs, rt) = run_rf(&buf);
    let (vc, vs_, vt) = run_v08(&buf);
    let (nc, ns, nt) = run_naive(&buf);
    assert_eq!(rc, N_RESPONSES, "rf must parse every response");
    assert_eq!(
        (rc, rs, rt),
        (vc, vs_, vt),
        "rf vs v08 mismatch: rf=({rc},{rs},{rt}) v08=({vc},{vs_},{vt})"
    );
    assert_eq!(
        (rc, rs, rt),
        (nc, ns, nt),
        "rf vs naive mismatch: rf=({rc},{rs},{rt}) naive=({nc},{ns},{nt})"
    );

    let mut group = c.benchmark_group("http1/parse");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.throughput(Throughput::Bytes(buf.len() as u64));

    group.bench_function("rf", |b| {
        b.iter(|| {
            let mut rest = &buf[..];
            let mut acc = 0u16;
            while !rest.is_empty() {
                let (resp, used) = Response::parse(rest).unwrap();
                acc ^= resp.status;
                black_box(resp.header(b"content-length").map_or(0, |v| v.len()));
                rest = &rest[used..];
            }
            black_box(acc);
        })
    });

    group.bench_function("rf-simd-call", |b| {
        b.iter(|| {
            let mut rest = &buf[..];
            let mut acc = 0u16;
            while !rest.is_empty() {
                let (status, headers, used) = parse_v08(rest).unwrap();
                acc ^= status;
                black_box(headers.get(b"content-length").map_or(0, |v| v.len()));
                rest = &rest[used..];
            }
            black_box(acc);
        })
    });

    group.bench_function("naive-str", |b| {
        b.iter(|| {
            let s = std::str::from_utf8(&buf).unwrap();
            let mut rest = s;
            let mut acc = 0u16;
            while !rest.is_empty() {
                let head_end = rest.find("\r\n\r\n").expect("terminator") + 4;
                let head = &rest[..head_end];
                let mut lines = head.split("\r\n");
                let status_line = lines.next().unwrap();
                let mut parts = status_line.splitn(3, ' ');
                let _version = parts.next().unwrap();
                let status: u16 = parts.next().unwrap().parse().unwrap();
                let _reason = parts.next().unwrap_or("");
                acc ^= status;
                for line in lines {
                    if line.is_empty() {
                        continue;
                    }
                    let (name, value) = line.split_once(':').expect("colon");
                    if name.trim().eq_ignore_ascii_case("content-length") {
                        black_box(value.trim().len());
                    }
                }
                rest = &rest[head_end..];
            }
            black_box(acc);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_http);
criterion_main!(benches);
