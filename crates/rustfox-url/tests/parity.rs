//! Parity tests: `Url::parse` (bitmap engine) vs `Url::parse_scans` (v0.7
//! engine) must be indistinguishable — every component, every error variant,
//! and the full internal state (via derived `Debug`) on:
//!
//! 1. all 2000 bench fixture URLs (same LCG generator as `benches/url.rs`),
//! 2. hand-picked grammar corners (empty/short/bracket/userinfo/port garbage),
//! 3. every prefix and padded variant of several base URLs (sweeps every
//!    state the state machine can be in, including the exact hybrid-dispatch
//!    threshold boundary),
//! 4. LCG-random strings, most of them invalid URLs.

use rustfox_url::Url;

/// Deterministic LCG (mirrors benches/url.rs, no `rand` dep).
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

/// 2000 mixed URLs — VERBATIM copy of `build_urls` from benches/url.rs so
/// the parity set is exactly the benchmark fixture set.
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

/// Both engines must agree on everything observable, including the private
/// ranges and the `hierarchical` flag (captured by the derived `Debug`).
fn assert_same(src: &str) {
    let a = Url::parse(src);
    let b = Url::parse_scans(src);
    match (&a, &b) {
        (Ok(x), Ok(y)) => {
            assert_eq!(x.scheme(), y.scheme(), "scheme mismatch for {src:?}");
            assert_eq!(x.userinfo(), y.userinfo(), "userinfo mismatch for {src:?}");
            assert_eq!(x.host(), y.host(), "host mismatch for {src:?}");
            assert_eq!(
                x.host_normalized(),
                y.host_normalized(),
                "host_normalized mismatch for {src:?}"
            );
            assert_eq!(x.port(), y.port(), "port mismatch for {src:?}");
            assert_eq!(x.path(), y.path(), "path mismatch for {src:?}");
            assert_eq!(x.query(), y.query(), "query mismatch for {src:?}");
            assert_eq!(x.fragment(), y.fragment(), "fragment mismatch for {src:?}");
            assert_eq!(x.as_str(), y.as_str());
            assert_eq!(
                format!("{x:?}"),
                format!("{y:?}"),
                "internal state mismatch for {src:?}"
            );
        }
        (Err(e1), Err(e2)) => {
            assert_eq!(e1, e2, "error variant mismatch for {src:?}");
        }
        _ => panic!("Ok/Err mismatch for {src:?}: bitmap={a:?} scans={b:?}"),
    }
}

#[test]
fn parity_on_all_2000_fixture_urls() {
    let urls = build_urls();
    assert_eq!(urls.len(), 2000);
    for u in &urls {
        assert_same(u);
    }
}

#[test]
fn parity_on_hand_picked_corners() {
    let cases = [
        // empty / scheme-less / scheme corners
        "",
        ":",
        "::",
        "://",
        ":x",
        "x",
        "x:",
        "x:/",
        "x://",
        "x:///",
        "x:////",
        "1http://x",
        "1a:x",
        "a1:x",
        "+a:x",
        "-a:x",
        ".a:x",
        "a/:x",
        "a b:c",
        "a:b:c",
        "a:b/c",
        "a:/b",
        // authority corners
        "http://",
        "http:///",
        "http:////x",
        "http://?q",
        "http://#f",
        "http://a",
        "http://a/",
        "http://a?",
        "http://a#",
        "http://a:",
        "http://a::",
        "http://a:1:",
        "http://a:b",
        "http://a:1b",
        "http://:",
        "http://::",
        "http://:@",
        "http://@",
        "http://@/",
        "http://a@b@c/",
        "http://a@:80/",
        "http://:80/",
        "http://:80x/",
        "http://host:/",
        "http://host:80:90/x",
        "http://host:80a/x",
        "http://host:8 0/x",
        "http://h:80@x/",
        // IPv6 corners
        "http://[",
        "http://[]",
        "http://[/",
        "http://[::1",
        "http://[::1]",
        "http://[::1]/",
        "http://[::1]:",
        "http://[::1]:443",
        "http://[::1]:443/",
        "http://[::1]:x/",
        "http://[::1]x",
        "http://[::1]x:1",
        "http://[[::1]]",
        "http://]:x",
        "http://ho[st:1/x",
        "http://u[@h:1/x",
        "http://u]@[::1]/x",
        "http://[::1]@x/",
        "http://x@[::1]/",
        "http://[::1]:443]x/",
        "ftp://[::ffff:1.2.3.4]:21/f",
        // file / opaque / tail corners
        "file://",
        "file:///",
        "file:///x",
        "file:///d/u?q#f",
        "file://h/x",
        "file://:80/x",
        "mailto:u@h?x#y",
        "mailto:?x",
        "mailto:#f",
        "mailto:",
        "data:,hello#x",
        "x:#",
        "x:?",
        "x:?#",
        "x:?a#b?c#d",
        "x:a/b?c/d#e/f",
        "x:?a@b[c]d:e#f",
        // case / unicode / real shapes
        "HTTPS://UPPER.CASE/X?Y#Z",
        "File:///ETC/Hosts",
        "git+ssh://user@git.local:22/path",
        "urn:isbn:0451450523",
        "http://%41/x",
        "http://xn--80ak6aa92e.com/ё",
        "https://пример.испытание/путь?запрос#фраг",
        "http://user:p@w@host:8080/p?q=1&x=2#z",
    ];
    for s in cases {
        assert_same(s);
    }
}

#[test]
fn parity_on_every_prefix_and_pad_of_base_urls() {
    let bases = [
        "http://user:pw@Host.Example.com:8080/a/b/c.html?q=1&x=2#frag",
        "http://[2001:db8::1]:443/x/y?q#f",
        "http://a@[::1]:80/x?q#f",
        "http://[::1]@x/y",
        "file:///etc/hosts.allow?q#f",
        "mailto:u@h?x#y",
        "ftp://files.example.com:21/pub/x.tar.gz",
        "https://h",
    ];
    for base in bases {
        // Every prefix: sweeps Hunt/Authority/Tail transitions and every
        // partial-token state the machine can pause in.
        for cut in 0..=base.len() {
            assert_same(&base[..cut]);
        }
        // Padded variants: sweeps the exact hybrid-dispatch threshold
        // neighborhood (scalar ⇄ bitmap) with realistic URLs.
        for pad in 0..96 {
            let mut s = String::from(base);
            for chunk in "aaaa".repeat(pad / 4 + 1).chars().take(pad) {
                s.push(chunk);
            }
            assert_same(&s);
        }
        // Same, but with structural padding that keeps the URL valid.
        for pad in 0..64 {
            let mut s = String::from(base);
            for i in 0..pad {
                if i % 3 == 0 {
                    s.push('/');
                } else {
                    s.push('a');
                }
            }
            assert_same(&s);
        }
    }
}

#[test]
fn parity_on_lcg_random_strings() {
    // Pool biased toward structural bytes so random strings constantly hit
    // authority/tail/bracket corners; includes invalid-URL bytes (space, %).
    let pool: &[&str] = &[
        "a", "z", "A", "Z", "5", "0", "9", ":", ":", "/", "/", "?", "?", "#", "@", "@", "[", "]",
        ".", "-", "+", "%", " ", "_", "ы", "Ё", "日",
    ];
    let mut rng = Lcg::new(0xDEAD_10CC_u64.wrapping_mul(0x2545F4914F6CDD1D) | 1);
    for _ in 0..30_000 {
        let len = (rng.next_u64() >> 33) as usize % 240;
        let mut s = String::with_capacity(len * 2);
        for _ in 0..len {
            s.push_str(pool[(rng.next_u64() >> 40) as usize % pool.len()]);
        }
        assert_same(&s);
    }
}

#[test]
fn parity_across_bitmap_cap_boundary() {
    // The bitmap path covers URLs up to BITMAP_CAP (2 KiB); beyond it the
    // scalar engine takes over. Both sides of the boundary must agree.
    for len in [1000usize, 2000, 2040, 2047, 2048, 2049, 2050, 4096, 5000] {
        let mut s = String::from("http://user@example.com/");
        s.extend(std::iter::repeat_n('a', len));
        s.push_str("/x?q#f");
        assert_same(&s);
    }
}
