//! Criterion benchmark: zero-copy HTTP/1.1 head parsing vs the std-string
//! way (`str::from_utf8` + `find` + `split` + `splitn` + `trim` + `parse`).
//!
//! `rf` — loops `Response::parse` over the concatenated buffer, XOR-folds
//! statuses and reads `content-length` (case-insensitive, zero-copy).
//! `naive-str` — the honest "just use strings" baseline implemented with
//! the std library only: one `from_utf8` over the buffer, per-response
//! `str::find("\r\n\r\n")`, `split("\r\n")`, `splitn(':')`, `trim()`,
//! `parse::<u16>()`.
//!
//! Setup asserts both parse the same number of responses and produce the
//! same status sum — otherwise panic with the numbers.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rustfox_http1::Response;
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

    // Setup parity: identical counts and status sums.
    let (rc, rs, rt) = run_rf(&buf);
    let (nc, ns, nt) = run_naive(&buf);
    assert_eq!(rc, N_RESPONSES, "rf must parse every response");
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
