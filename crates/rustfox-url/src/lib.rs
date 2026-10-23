//! rustfox-url: zero-copy URL parser (kernel replacement for `nsStandardURL`).
//!
//! # Where this sits in Gecko
//!
//! `nsStandardURL::ParseURL` (netwerk/base/nsStandardURL.cpp) is on every
//! network and DOM hot path: every `href` resolution, every redirect, every
//! `fetch()` re-parses or re-normalizes a URL string. The C++ implementation
//! parses eagerly into an owned `mSpec` plus an array of *mutable* position
//! fields, and — critically — eagerly lowercases the host into a freshly
//! allocated buffer (`nsStandardURL::BuildNormalizedSpec`). For the dominant
//! real-world case (`https://already-lowercase.example/…`) that allocation
//! and copy buy nothing.
//!
//! # This design
//!
//! * **Full zero-copy**: `parse` performs *no* allocation. `Url<'a>` stores
//!   only `Range<usize>` offsets into the source string; every accessor is a
//!   slice of the original. A browser resolves millions of URLs per session;
//!   per-parse allocations show up directly in profiles as malloc/free churn
//!   and cache misses. (Gecko itself moved in this direction: `rust-url`,
//!   which servo contributed to Gecko, avoids intermediate strings.)
//! * **SIMD splits (v0.7 engine, kept as `parse_scans`)**: authority/path
//!   termination (`/`, `?`, `#`) and the scheme terminator scan use
//!   `rustfox_simd::find_byte` / `find_any2` (AVX2: 32 bytes/step). URL
//!   components are short, so the win is not raw bandwidth but *latency*:
//!   one vector compare replaces a branchy byte loop, and the branch
//!   predictor stops being the bottleneck on random URL mixes.
//! * **One-pass structural bitmap (`parse_bitmap`, the engine behind
//!   `parse`)**: the whole URL is classified in a *single* SIMD sweep — per
//!   32-byte chunk, 7 vector compares (`:` `/` `?` `#` `@` `[` `]`) + 7
//!   movemasks OR-ed into one u32 chunk bitmask. Every component is then
//!   derived from the ordered mask stream in one walk (`trailing_zeros`),
//!   so the buffer is traversed exactly once instead of 3–5 times by the
//!   split searches. Hybrid dispatch by length (measured, see worklog
//!   Task 7-b): the sweep wins on *every* length tier of the fixture mix —
//!   even at ~55 B the branchless vector path beats a scalar structural
//!   test, whose data-dependent branch mispredicts — so the scalar single
//!   pass only serves inputs beyond the 2 KiB mask window (cold) and
//!   targets without SIMD.
//! * **Lazy host lowercasing**: `host_normalized` returns `Cow::Borrowed`
//!   unless the host actually contains `A-Z` (rare — most hosts in the wild
//!   and in caches are already lowercase). The check is one unrolled ASCII
//!   pass; the allocation happens only on the cold path.
//!
//! # Scope (honest)
//!
//! This is the *fast path* parser for absolute URLs with the WHATWG-ish
//! component semantics needed by the kernel benches: scheme, authority
//! (userinfo / host / port), path, query, fragment; IPv6 in brackets;
//! opaque paths (`mailto:`). It is **not** a full WHATWG state machine:
//! percent-normalization, path segment collapsing (`/../`), default-port
//! elision and query re-encoding are downstream concerns.
//!
//! TODO(IDNA): non-ASCII hosts (`https://пример.испытание/`) are returned
//! verbatim; the ASCII fast path benchmarked here deliberately skips
//! punycode/UTS-46 mapping (that belongs to a slower cold-path wrapper).

use rustfox_simd::find_byte;

/// Every failure mode of [`Url::parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UrlError {
    /// No `scheme:` prefix at all: scanning the scheme prefix hit a
    /// non-scheme byte before any `:` (input looks relative, e.g.
    /// `example.com/x`, `//host/path`) or the input has no `:` at all.
    NoScheme,
    /// A `:` is present but the prefix is not a valid scheme
    /// (empty scheme `"://x"`, or a non-ALPHA first byte, e.g. `"1http://x"`).
    BadScheme,
    /// Malformed authority: unterminated `[` IPv6 literal or a stray `]`.
    BadAuthority,
    /// Port suffix contains non-digit characters (e.g. `host:80a`).
    BadPort,
    /// Empty host on a scheme that requires one (`http:///path`).
    /// Only `file` is allowed an empty host.
    NoHost,
}

/// A parsed URL: indices into the borrowed source, nothing else.
///
/// `parse` is allocation-free; accessors slice the original string.
#[derive(Debug, Clone)]
pub struct Url<'a> {
    src: &'a str,
    /// `scheme` = `src[..scheme_end]`; `src.as_bytes()[scheme_end] == b':'`.
    scheme_end: usize,
    /// True when the URL had an `//` authority section (hierarchical form).
    hierarchical: bool,
    userinfo: Option<std::ops::Range<usize>>,
    host: std::ops::Range<usize>,
    port: Option<std::ops::Range<usize>>,
    path: std::ops::Range<usize>,
    query: Option<std::ops::Range<usize>>,
    fragment: Option<std::ops::Range<usize>>,
}

impl<'a> Url<'a> {
    /// Parse an absolute URL with zero allocations.
    ///
    /// Component grammar (simplified WHATWG):
    /// `scheme ":" [ "//" authority ] path [ "?" query ] [ "#" fragment ]`,
    /// `authority = [ userinfo "@" ] host [ ":" port ]`.
    ///
    /// Backed by the one-pass structural bitmap engine ([`Self::parse_bitmap`]);
    /// the previous split-by-searching engine remains available as
    /// [`Self::parse_scans`]. Both are byte-identical in components *and*
    /// errors (enforced by parity tests on the full bench fixture set and
    /// on randomized strings).
    pub fn parse(src: &'a str) -> Result<Url<'a>, UrlError> {
        Self::parse_bitmap(src)
    }

    /// One-pass structural bitmap engine.
    ///
    /// Hybrid dispatch by length (threshold measured, see worklog Task 7-b):
    /// * outside `BITMAP_MIN_LEN..=BITMAP_CAP` — scalar single pass (only
    ///   the > 2 KiB cold tail in practice);
    /// * otherwise — classify all chunks, then one ordered mask walk.
    pub fn parse_bitmap(src: &'a str) -> Result<Url<'a>, UrlError> {
        let len = src.len();
        if !(BITMAP_MIN_LEN..=BITMAP_CAP).contains(&len) {
            return engine_scalar(src);
        }
        match bitmap_tier() {
            #[cfg(target_arch = "x86_64")]
            BitmapTier::Avx2 => {
                // SAFETY: avx2 was detected at process start.
                unsafe { engine_bitmap(src, false) }
            }
            #[cfg(target_arch = "x86_64")]
            BitmapTier::Sse2 => {
                // SAFETY: sse2 is a baseline feature of every x86_64 target.
                unsafe { engine_bitmap(src, true) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            BitmapTier::Scalar => engine_scalar(src),
        }
    }

    /// v0.7 engine: split-by-searching. One SIMD scan per structural
    /// boundary — `find_any2` for `/`+`?`, a second `find_byte` for `#`,
    /// reverse scalar scans for `@`/`:` — each with its own call setup,
    /// tier check and vector prologue. Kept as the benchmark comparison
    /// baseline for [`Self::parse_bitmap`]; semantics are byte-identical.
    pub fn parse_scans(src: &'a str) -> Result<Url<'a>, UrlError> {
        let bytes = src.as_bytes();
        let scheme_end = parse_scheme(bytes)?;

        // `rest` starts right after the ':'.
        let after = scheme_end + 1;
        let mut u = Url {
            src,
            scheme_end,
            hierarchical: false,
            userinfo: None,
            host: after..after,
            port: None,
            path: after..bytes.len(),
            query: None,
            fragment: None,
        };

        if bytes[after..].starts_with(b"//") {
            u.hierarchical = true;
            let auth = after + 2;
            // Authority ends at the first '/', '?' or '#'. `/` and `?` are
            // found in one SIMD pass (find_any2); `#` in a second — the
            // authority region is short (a hostname), so two vectorized
            // scans still beat one scalar loop.
            let tail = &bytes[auth..];
            let end_rel = find_any2_min(tail, b'/', b'?', b'#');
            let auth_end = auth + end_rel;
            u.parse_authority(auth..auth_end)?;

            // Whatever follows the authority is path/query/fragment.
            u.parse_tail(auth_end);
        } else {
            // Opaque path (`mailto:user@host`, `data:...`). Query and
            // fragment still terminate it; a leading '?'/'#' yields an
            // empty path.
            u.parse_tail(after);
        }
        Ok(u)
    }

    /// Split `authority` (a `Range` into `src`) into userinfo, host, port.
    fn parse_authority(&mut self, auth: std::ops::Range<usize>) -> Result<(), UrlError> {
        let bytes = self.src.as_bytes();
        let mut rest = auth.clone();

        // Userinfo: the LAST '@' wins (per RFC 3986 the userinfo may not
        // contain '@' after percent-decoding, so the last one is the
        // delimiter). Authority sections are short; a reverse byte scan is
        // the cheap honest tool here.
        if let Some(at) = bytes[auth.clone()].iter().rposition(|&b| b == b'@') {
            let at = auth.start + at;
            self.userinfo = Some(auth.start..at);
            rest = at + 1..auth.end;
        }

        if rest.start >= rest.end {
            // Empty authority: allowed only for `file` (e.g. `file:///etc/hosts`).
            return if self.scheme_is(b"file") {
                self.host = rest.start..rest.start;
                Ok(())
            } else {
                Err(UrlError::NoHost)
            };
        }

        if bytes[rest.start] == b'[' {
            // IPv6 literal: `[...]:port`. The closing `]` is mandatory.
            let close = bytes[rest.clone()]
                .iter()
                .position(|&b| b == b']')
                .ok_or(UrlError::BadAuthority)?;
            let close = rest.start + close;
            self.host = rest.start..close + 1; // include the brackets
                                               // Port may follow `]`; it must be digits or empty.
            if close + 1 < auth.end {
                if bytes[close + 1] != b':' {
                    return Err(UrlError::BadAuthority);
                }
                self.set_port(close + 2..auth.end)?;
            }
            return Ok(());
        }

        // Reg-name: the port is after the LAST ':' (an IPvFuture/odd name
        // with colons keeps them in the host; the trailing segment must
        // still be all-digits or empty).
        let host_port = rest.clone();
        if let Some(colon) = bytes[host_port.clone()].iter().rposition(|&b| b == b':') {
            let colon = host_port.start + colon;
            self.host = host_port.start..colon;
            self.set_port(colon + 1..host_port.end)?;
        } else {
            self.host = host_port;
        }

        if self.host.is_empty() && !self.scheme_is(b"file") {
            return Err(UrlError::NoHost);
        }
        Ok(())
    }

    /// Port slice must be all digits or empty (empty => `port()` is `None`).
    fn set_port(&mut self, r: std::ops::Range<usize>) -> Result<(), UrlError> {
        let bytes = self.src.as_bytes();
        if bytes[r.clone()].iter().any(|&b| !b.is_ascii_digit()) {
            return Err(UrlError::BadPort);
        }
        // `host:` (empty port) is legal; normalize it to "no port".
        if r.start < r.end {
            self.port = Some(r);
        }
        Ok(())
    }

    /// Split the remainder after authority/scheme into path, query, fragment.
    fn parse_tail(&mut self, from: usize) {
        let bytes = self.src.as_bytes();
        let tail = &bytes[from..];
        // Path ends at the first '?' or '#'; one SIMD pass for both.
        let cut = find_any2_min(tail, b'?', b'?', b'#');
        self.path = from..from + cut;
        let mut cur = from + cut;
        if cur < bytes.len() && bytes[cur] == b'?' {
            cur += 1;
            let q = &bytes[cur..];
            let h = find_byte(q, b'#').unwrap_or(q.len());
            self.query = Some(cur..cur + h);
            cur += h;
        }
        if cur < bytes.len() && bytes[cur] == b'#' {
            self.fragment = Some(cur + 1..bytes.len());
        }
    }

    fn scheme_is(&self, name: &[u8]) -> bool {
        self.src.as_bytes()[..self.scheme_end].eq_ignore_ascii_case(name)
    }

    /// The full source string, byte-identical to the input.
    #[inline]
    pub fn as_str(&self) -> &'a str {
        self.src
    }

    /// Scheme without the `':'`. Case is preserved (zero-copy).
    #[inline]
    pub fn scheme(&self) -> &'a str {
        &self.src[..self.scheme_end]
    }

    /// `user:password` before the last `@` of the authority, if present.
    #[inline]
    pub fn userinfo(&self) -> Option<&'a str> {
        self.userinfo.clone().map(|r| &self.src[r])
    }

    /// Host token: reg-name (case preserved) or bracketed IPv6 literal
    /// `[...]` (brackets included). May be empty only for `file`.
    #[inline]
    pub fn host(&self) -> &'a str {
        &self.src[self.host.clone()]
    }

    /// Host lowercased on demand. The common case (already-lowercase host,
    /// the overwhelming share of real traffic) returns `Cow::Borrowed` —
    /// no allocation, no copy. `A-Z` triggers the cold `Owned` path.
    pub fn host_normalized(&self) -> std::borrow::Cow<'a, str> {
        let h = self.host();
        if h.bytes().any(|b| b.is_ascii_uppercase()) {
            std::borrow::Cow::Owned(h.to_ascii_lowercase())
        } else {
            std::borrow::Cow::Borrowed(h)
        }
    }

    /// Port digits, if a non-empty port was present. (`host:` → `None`,
    /// matching the "empty port is absent" normalization.)
    #[inline]
    pub fn port(&self) -> Option<&'a str> {
        self.port.clone().map(|r| &self.src[r])
    }

    /// Path including the leading `/` for hierarchical URLs. An empty
    /// hierarchical path is returned as `"/"` (WHATWG serialization);
    /// opaque paths are returned verbatim (possibly empty).
    #[inline]
    pub fn path(&self) -> &'a str {
        if self.hierarchical && self.path.is_empty() {
            return "/";
        }
        &self.src[self.path.clone()]
    }

    /// Query without the `'?'`, if present.
    #[inline]
    pub fn query(&self) -> Option<&'a str> {
        self.query.clone().map(|r| &self.src[r])
    }

    /// Fragment without the `'#'`, if present.
    #[inline]
    pub fn fragment(&self) -> Option<&'a str> {
        self.fragment.clone().map(|r| &self.src[r])
    }
}

/// Scan `scheme = ALPHA *( ALPHA / DIGIT / "+" / "-" / "." ) ":"`.
///
/// Schemes are tiny (`http`, `https`, `data`, `mailto`), so a scalar loop
/// over the prefix is cheaper than setting up vector constants; the SIMD
/// budget is spent on the long tail (authority/path) instead.
fn parse_scheme(bytes: &[u8]) -> Result<usize, UrlError> {
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b':' {
            break;
        }
        if !(c.is_ascii_alphanumeric() || c == b'+' || c == b'-' || c == b'.') {
            // First invalid byte before any ':' → this is not an absolute
            // URL with a scheme (relative URL, which this kernel rejects).
            return Err(UrlError::NoScheme);
        }
        i += 1;
    }
    if i == bytes.len() {
        return Err(UrlError::NoScheme); // no ':' anywhere
    }
    if i == 0 {
        return Err(UrlError::BadScheme); // "://..."
    }
    if !bytes[0].is_ascii_alphabetic() {
        return Err(UrlError::BadScheme); // "1http:..." — ':' present, prefix not a scheme
    }
    Ok(i)
}

/// First index of any of `a`, `b`, `c` in `hay` — `find_any2` for `/`+`?`
/// plus one `find_byte` for `#`, taking the minimum. All three scans are
/// SIMD; the authority region they run on is short.
#[inline]
fn find_any2_min(hay: &[u8], a: u8, b: u8, c: u8) -> usize {
    let ab = rustfox_simd::find_any2(hay, a, b).unwrap_or(hay.len());
    let h = find_byte(hay, c).unwrap_or(hay.len());
    ab.min(h)
}

// ---------------------------------------------------------------------------
// One-pass structural bitmap engine (`parse_bitmap`)
// ---------------------------------------------------------------------------
//
// The v0.7 engine splits by *searching*: every boundary costs its own scan
// with call setup, tier check and vector prologue, and each scan re-reads
// bytes the previous one already saw. This engine inverts the shape:
//
//   1. ONE SIMD sweep classifies the whole URL. For every 32-byte chunk:
//      7 equality compares against broadcast constants
//      (`:` `/` `?` `#` `@` `[` `]`), 7 movemasks, OR-fold into one u32
//      chunk bitmask (bit i = structural byte at chunk offset i).
//   2. Chunk masks live in a small fixed stack buffer (128 × u32 = 512 B,
//      covers URLs up to 2 KiB on both the AVX2 and SSE2 tiers). Longer
//      inputs are cold and take the scalar single pass.
//   3. ONE ordered walk over the mask buffer (`trailing_zeros` +
//      clear-lowest-bit) feeds structural bytes to [`Machine`], which
//      derives every component: scheme (first `:`), authority end (first
//      `/`, `?` or `#` after `//`), userinfo (last `@`), IPv6 brackets,
//      reg-name port (last `:` since the last `@`), path/query/fragment
//      (first `?`, then first `#`). The mask buffer is traversed exactly
//      once — no component re-scans it.
//
// Only two tiny validations still read raw bytes, because the bitmap
// deliberately classifies *structure*, not character membership: the scheme
// prefix (~4–8 B, `ALPHA *(ALNUM/+-/.)` check) and the port digits (~2–5 B).
// Both are L1-hot after the sweep, and the v0.7 engine pays for them too.

/// Chunk-mask buffer size. 64 chunks × 32 B = 2 KiB of URL on the bitmap
/// path (both tiers emit one u32 per 32 bytes); longer URLs take the scalar
/// engine.
const BITMAP_MAX_CHUNKS: usize = 64;
/// Longest URL handled by the bitmap path.
const BITMAP_CAP: usize = 32 * BITMAP_MAX_CHUNKS;
/// Hybrid dispatch threshold, measured on the bench fixture mix (see worklog
/// Task 7-b): below this, one scalar pass beats vector setup + mask
/// bookkeeping. Measured value is 0 — the bitmap sweep wins on every length
/// tier of the fixture (scalar feeder ~78–93 ns vs bitmap ~42 ns even at
/// 55 B), so the dispatch only redirects the > `BITMAP_CAP` cold tail.
const BITMAP_MIN_LEN: usize = 0;

/// Bit i of this mask is set iff byte value i is structural
/// (`:` `/` `?` `#` `@` `[` `]`). A single variable-shift bit-test classifies
/// a byte without a branch or jump table.
const STRUCTURAL_MASK: u128 = (1u128 << b':')
    | (1u128 << b'/')
    | (1u128 << b'?')
    | (1u128 << b'#')
    | (1u128 << b'@')
    | (1u128 << b'[')
    | (1u128 << b']');

#[inline]
fn is_structural(b: u8) -> bool {
    // Structural bytes are all ASCII; the `b < 128` guard keeps the u128
    // shift in range (bytes >= 0x80 are never structural).
    b < 128 && (STRUCTURAL_MASK >> b) & 1 == 1
}

/// Runtime SIMD tier for the classifier, detected once per process
/// (same `#[target_feature]` + `OnceLock` pattern as `rustfox-simd`; the
/// primitive is deliberately private to this crate).
static BITMAP_TIER: std::sync::OnceLock<BitmapTier> = std::sync::OnceLock::new();

#[derive(Clone, Copy)]
enum BitmapTier {
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Sse2,
    #[cfg(not(target_arch = "x86_64"))]
    Scalar,
}

#[inline]
fn bitmap_tier() -> BitmapTier {
    *BITMAP_TIER.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                BitmapTier::Avx2
            } else {
                BitmapTier::Sse2
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            BitmapTier::Scalar
        }
    })
}

/// Scalar engine: one pass over the bytes feeding [`Machine`] directly.
/// Used for the portable tier and for inputs outside the bitmap window
/// (shorter than the threshold or longer than [`BITMAP_CAP`] — cold).
fn engine_scalar(src: &str) -> Result<Url<'_>, UrlError> {
    let bytes = src.as_bytes();
    let mut m = Machine::new(src);
    for (i, &b) in bytes.iter().enumerate() {
        if is_structural(b) {
            m.feed(i, b)?;
        }
    }
    m.finish()
}

/// Bitmap engine: classify every 32-byte chunk into a u32 bitmask, then walk
/// the mask buffer exactly once.
///
/// # Safety
/// Caller must have detected the matching feature (`avx2` when `sse2_tier`
/// is false, `sse2` when it is true) at process start.
#[cfg(target_arch = "x86_64")]
unsafe fn engine_bitmap(src: &str, sse2_tier: bool) -> Result<Url<'_>, UrlError> {
    let bytes = src.as_bytes();
    let nchunks = bytes.len().div_ceil(32);
    debug_assert!(nchunks <= BITMAP_MAX_CHUNKS);

    // No zero-init: `classify_*` below writes every one of the `nchunks`
    // slots before the walk reads any of them.
    let mut buf: [std::mem::MaybeUninit<u32>; BITMAP_MAX_CHUNKS] =
        unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let masks: &mut [u32] =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u32>(), nchunks) };
    if sse2_tier {
        // SAFETY: sse2 tier checked by `parse_bitmap`.
        unsafe { classify_sse2(bytes, masks) };
    } else {
        // SAFETY: avx2 tier checked by `parse_bitmap`.
        unsafe { classify_avx2(bytes, masks) };
    }

    // Exactly one ordered walk over the chunk masks: LSB-first within a
    // chunk and chunks in order ⇒ structural bytes arrive in ascending
    // position order, which is what [`Machine`] requires.
    let mut m = Machine::new(src);
    for (ci, &chunk) in masks.iter().enumerate() {
        let base = ci * 32;
        let mut c = chunk;
        while c != 0 {
            let pos = base + c.trailing_zeros() as usize;
            // SAFETY: mask bits only mark positions < bytes.len() (the
            // classifier never sets a bit for a byte it did not read).
            let byte = unsafe { *bytes.as_ptr().add(pos) };
            m.feed(pos, byte)?;
            c &= c - 1;
        }
    }
    m.finish()
}

/// Classify `bytes` into one u32 mask per 32-byte chunk. The final partial
/// chunk is handled by an overlapping load (re-based to bit 0) when possible.
/// `masks.len()` must be `bytes.len().div_ceil(32)`.
///
/// Direct variant of the prescribed design: **7 compares + 7 movemasks**,
/// OR-folded per chunk (~21 µops per 32 B).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn classify_avx2(bytes: &[u8], masks: &mut [u32]) {
    use std::arch::x86_64::*;

    let n = bytes.len();
    let nfull = n / 32;
    for (i, slot) in masks[..nfull].iter_mut().enumerate() {
        // SAFETY: i * 32 + 32 <= n inside the full-chunk loop.
        let v = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(i * 32) as *const __m256i) };
        // SAFETY: avx2 enabled by this fn's attribute; constants hoist.
        *slot = unsafe { chunk_mask_cmp_avx2(v) };
    }
    let rem = n & 31;
    if rem != 0 {
        let m = if nfull > 0 {
            // SAFETY: n - 32 is a valid 32-byte window inside the slice.
            // The window's last `rem` bytes are exactly the final chunk;
            // shifting right by 32 - rem re-bases their bits to 0..rem.
            let v = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(n - 32) as *const __m256i) };
            (unsafe { chunk_mask_cmp_avx2(v) }) >> (32 - rem)
        } else {
            scalar_tail_mask(bytes)
        };
        masks[nfull] = m;
    }
}

/// Seven structural-byte equality masks of one 32-byte vector, OR-folded.
///
/// Variant note (Task 7-b measurement): a simdjson-style nibble-lookup
/// classifier (`pshufb`-based, ~11 µops/chunk vs ~21 here) was implemented
/// and A/B-measured on the idle bench machine: full-mix 53.3–53.5 ns/url vs
/// 52.1–52.6 ns/url for this 7-cmp fold — within noise, because at URL
/// lengths the per-URL fixed costs (mask walk + state machine) dominate.
/// The simpler prescribed design is shipped.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn chunk_mask_cmp_avx2(v: std::arch::x86_64::__m256i) -> u32 {
    use std::arch::x86_64::*;

    let mut m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(b':' as i8))) as u32;
    m |= _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(b'/' as i8))) as u32;
    m |= _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(b'?' as i8))) as u32;
    m |= _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(b'#' as i8))) as u32;
    m |= _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(b'@' as i8))) as u32;
    m |= _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(b'[' as i8))) as u32;
    m |= _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(b']' as i8))) as u32;
    m
}

/// SSE2 twin of [`classify_avx2`]: same 32-byte chunk layout, built from two
/// 16-byte vector halves (low half → bits 0..16, high half → bits 16..32),
/// so both tiers share the mask-walk code and buffer sizing.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn classify_sse2(bytes: &[u8], masks: &mut [u32]) {
    use std::arch::x86_64::*;

    let v_colon = _mm_set1_epi8(b':' as i8);
    let v_slash = _mm_set1_epi8(b'/' as i8);
    let v_query = _mm_set1_epi8(b'?' as i8);
    let v_frag = _mm_set1_epi8(b'#' as i8);
    let v_at = _mm_set1_epi8(b'@' as i8);
    let v_lbrack = _mm_set1_epi8(b'[' as i8);
    let v_rbrack = _mm_set1_epi8(b']' as i8);

    let n = bytes.len();
    let nfull = n / 32;
    for (i, slot) in masks[..nfull].iter_mut().enumerate() {
        let p = bytes.as_ptr().add(i * 32);
        let lo = _mm_loadu_si128(p as *const __m128i);
        let hi = _mm_loadu_si128(p.add(16) as *const __m128i);
        let mut m = _mm_movemask_epi8(_mm_cmpeq_epi8(lo, v_colon)) as u32;
        m |= _mm_movemask_epi8(_mm_cmpeq_epi8(lo, v_slash)) as u32;
        m |= _mm_movemask_epi8(_mm_cmpeq_epi8(lo, v_query)) as u32;
        m |= _mm_movemask_epi8(_mm_cmpeq_epi8(lo, v_frag)) as u32;
        m |= _mm_movemask_epi8(_mm_cmpeq_epi8(lo, v_at)) as u32;
        m |= _mm_movemask_epi8(_mm_cmpeq_epi8(lo, v_lbrack)) as u32;
        m |= _mm_movemask_epi8(_mm_cmpeq_epi8(lo, v_rbrack)) as u32;
        let mut h = _mm_movemask_epi8(_mm_cmpeq_epi8(hi, v_colon)) as u32;
        h |= _mm_movemask_epi8(_mm_cmpeq_epi8(hi, v_slash)) as u32;
        h |= _mm_movemask_epi8(_mm_cmpeq_epi8(hi, v_query)) as u32;
        h |= _mm_movemask_epi8(_mm_cmpeq_epi8(hi, v_frag)) as u32;
        h |= _mm_movemask_epi8(_mm_cmpeq_epi8(hi, v_at)) as u32;
        h |= _mm_movemask_epi8(_mm_cmpeq_epi8(hi, v_lbrack)) as u32;
        h |= _mm_movemask_epi8(_mm_cmpeq_epi8(hi, v_rbrack)) as u32;
        *slot = m | (h << 16);
    }
    let rem = n & 31;
    if rem != 0 {
        let m = if nfull > 0 {
            // SAFETY: n - 32 is a valid 32-byte window inside the slice; the
            // same overlap re-basing as in [`classify_avx2`].
            let p = unsafe { bytes.as_ptr().add(n - 32) };
            let lo = unsafe { _mm_loadu_si128(p as *const __m128i) };
            let hi = unsafe { _mm_loadu_si128(p.add(16) as *const __m128i) };
            let om = _mm_movemask_epi8(_mm_cmpeq_epi8(lo, v_colon)) as u32
                | _mm_movemask_epi8(_mm_cmpeq_epi8(lo, v_slash)) as u32
                | _mm_movemask_epi8(_mm_cmpeq_epi8(lo, v_query)) as u32
                | _mm_movemask_epi8(_mm_cmpeq_epi8(lo, v_frag)) as u32
                | _mm_movemask_epi8(_mm_cmpeq_epi8(lo, v_at)) as u32
                | _mm_movemask_epi8(_mm_cmpeq_epi8(lo, v_lbrack)) as u32
                | _mm_movemask_epi8(_mm_cmpeq_epi8(lo, v_rbrack)) as u32;
            let oh = _mm_movemask_epi8(_mm_cmpeq_epi8(hi, v_colon)) as u32
                | _mm_movemask_epi8(_mm_cmpeq_epi8(hi, v_slash)) as u32
                | _mm_movemask_epi8(_mm_cmpeq_epi8(hi, v_query)) as u32
                | _mm_movemask_epi8(_mm_cmpeq_epi8(hi, v_frag)) as u32
                | _mm_movemask_epi8(_mm_cmpeq_epi8(hi, v_at)) as u32
                | _mm_movemask_epi8(_mm_cmpeq_epi8(hi, v_lbrack)) as u32
                | _mm_movemask_epi8(_mm_cmpeq_epi8(hi, v_rbrack)) as u32;
            (om | (oh << 16)) >> (32 - rem)
        } else {
            scalar_tail_mask(bytes)
        };
        masks[nfull] = m;
    }
}

/// Bitmask of the structural bytes of `bytes` (bit k = byte k), used only
/// for sub-chunk inputs (< 32 bytes).
#[cfg(target_arch = "x86_64")]
#[inline]
fn scalar_tail_mask(bytes: &[u8]) -> u32 {
    let mut m = 0u32;
    for (k, &b) in bytes.iter().enumerate() {
        if is_structural(b) {
            m |= 1 << k;
        }
    }
    m
}

/// Parser phase: hunting the scheme terminator, inside the authority, or
/// past it (path/query/fragment region).
#[derive(Clone, Copy)]
enum Phase {
    Hunt,
    Authority,
    Tail,
}

/// Structural-byte state machine shared by the scalar feeder and the bitmap
/// walk — both feed `(position, byte)` in ascending order, so the component
/// derivation logic is literally the same code on both paths.
///
/// Key invariant making the scheme check O(1) on structural bytes: the
/// scheme grammar (`ALPHA *(ALNUM / "+" / "-" / ".")`) contains none of the
/// seven structural bytes, so the FIRST structural byte decides — if it is
/// not `':'`, the input is a relative reference (`NoScheme`), exactly what
/// the sequential `parse_scheme` scan concludes.
struct Machine<'a> {
    src: &'a str,
    phase: Phase,
    // Scheme.
    scheme_end: usize,
    is_file: bool,
    // Authority ([auth_start..auth_end)).
    hierarchical: bool,
    auth_start: usize,
    auth_end: usize,
    /// Last `@` in the authority (userinfo delimiter, RFC 3986).
    last_at: Option<usize>,
    /// Last `:` seen *since the last `@`* — at finalize time this is the
    /// last `:` of the reg-name host region.
    colon_since_at: Option<usize>,
    /// First `[` since the last `@`, and the first `]` after it.
    bracket_open: Option<usize>,
    bracket_close: Option<usize>,
    // Results.
    host: std::ops::Range<usize>,
    port: Option<std::ops::Range<usize>>,
    tail_start: usize,
    path_end: usize,
    query_start: Option<usize>,
    query_end: usize,
    fragment_start: Option<usize>,
}

impl<'a> Machine<'a> {
    fn new(src: &'a str) -> Self {
        let len = src.len();
        Machine {
            src,
            phase: Phase::Hunt,
            scheme_end: 0,
            is_file: false,
            hierarchical: false,
            auth_start: 0,
            auth_end: 0,
            last_at: None,
            colon_since_at: None,
            bracket_open: None,
            bracket_close: None,
            host: 0..0,
            port: None,
            tail_start: 0,
            path_end: len,
            query_start: None,
            query_end: len,
            fragment_start: None,
        }
    }

    #[inline]
    fn feed(&mut self, pos: usize, byte: u8) -> Result<(), UrlError> {
        match self.phase {
            Phase::Hunt => self.feed_hunt(pos, byte),
            Phase::Authority => self.feed_authority(pos, byte),
            Phase::Tail => {
                self.feed_tail(pos, byte);
                Ok(())
            }
        }
    }

    #[inline]
    fn feed_hunt(&mut self, pos: usize, byte: u8) -> Result<(), UrlError> {
        if byte != b':' {
            // First structural byte is not ':' ⇒ relative reference. Either
            // no ':' follows at all, or one does and this byte sits before
            // it — a scheme can never contain a structural byte — so the
            // sequential scanner would stop here and return NoScheme too.
            return Err(UrlError::NoScheme);
        }
        // Validate the scheme prefix scalarly (tiny: 4–8 bytes).
        let bytes = self.src.as_bytes();
        let prefix = &bytes[..pos];
        for &c in prefix {
            if !(c.is_ascii_alphanumeric() || c == b'+' || c == b'-' || c == b'.') {
                return Err(UrlError::NoScheme);
            }
        }
        if pos == 0 || !bytes[0].is_ascii_alphabetic() {
            return Err(UrlError::BadScheme); // "://…" or "1http://…"
        }
        self.scheme_end = pos;
        self.is_file = prefix.eq_ignore_ascii_case(b"file");
        let after = pos + 1;
        if bytes.get(after) == Some(&b'/') && bytes.get(after + 1) == Some(&b'/') {
            self.hierarchical = true;
            self.auth_start = after + 2;
            self.phase = Phase::Authority;
        } else {
            // Opaque path (`mailto:…`, `data:…`): host/port stay empty.
            self.tail_start = after;
            self.phase = Phase::Tail;
        }
        Ok(())
    }

    #[inline]
    fn feed_authority(&mut self, pos: usize, byte: u8) -> Result<(), UrlError> {
        if pos < self.auth_start {
            return Ok(()); // the two '/' bytes that opened the authority
        }
        match byte {
            b'/' | b'?' | b'#' => {
                self.auth_end = pos;
                self.finalize_authority()?;
                self.tail_start = pos;
                self.phase = Phase::Tail;
                // The terminating byte itself starts query/fragment.
                self.feed_tail(pos, byte);
                Ok(())
            }
            b'@' => {
                self.last_at = Some(pos);
                self.colon_since_at = None;
                self.bracket_open = None;
                self.bracket_close = None;
                Ok(())
            }
            b'[' => {
                if self.bracket_open.is_none() {
                    self.bracket_open = Some(pos);
                    self.bracket_close = None;
                }
                Ok(())
            }
            b']' => {
                if self.bracket_open.is_some() && self.bracket_close.is_none() {
                    self.bracket_close = Some(pos);
                }
                Ok(())
            }
            b':' => {
                self.colon_since_at = Some(pos);
                Ok(())
            }
            _ => Ok(()), // callers only feed the 7 structural bytes
        }
    }

    #[inline]
    fn feed_tail(&mut self, pos: usize, byte: u8) {
        match byte {
            b'?' if self.fragment_start.is_none() && self.query_start.is_none() => {
                self.path_end = pos;
                self.query_start = Some(pos + 1);
            }
            b'#' if self.fragment_start.is_none() => {
                if self.query_start.is_some() {
                    self.query_end = pos;
                } else {
                    self.path_end = pos;
                }
                self.fragment_start = Some(pos + 1);
            }
            _ => {} // '/', '@', '[', ']', ':' are path/query/fragment data
        }
    }

    /// Split the finished authority `[auth_start..auth_end)` into userinfo,
    /// host and port — the same decisions, in the same order, as
    /// `Url::parse_authority` of the v0.7 engine.
    fn finalize_authority(&mut self) -> Result<(), UrlError> {
        let h0 = match self.last_at {
            Some(a) => a + 1,
            None => self.auth_start,
        };
        if h0 >= self.auth_end {
            // Empty authority: allowed only for `file`.
            if !self.is_file {
                return Err(UrlError::NoHost);
            }
            self.host = h0..h0;
            return Ok(());
        }
        let bytes = self.src.as_bytes();
        if bytes[h0] == b'[' {
            // IPv6 literal: `[...]:port`; the closing `]` is mandatory.
            let close = self.bracket_close.ok_or(UrlError::BadAuthority)?;
            self.host = h0..close + 1; // include the brackets
            if close + 1 < self.auth_end {
                if bytes[close + 1] != b':' {
                    return Err(UrlError::BadAuthority);
                }
                self.set_port(close + 2..self.auth_end)?;
            }
            return Ok(());
        }
        // Reg-name: the port is after the LAST ':' of the host region.
        match self.colon_since_at {
            Some(c) => {
                self.host = h0..c;
                self.set_port(c + 1..self.auth_end)?;
            }
            None => self.host = h0..self.auth_end,
        }
        if self.host.is_empty() && !self.is_file {
            return Err(UrlError::NoHost);
        }
        Ok(())
    }

    /// Port slice must be all digits or empty (empty => `port()` is `None`).
    fn set_port(&mut self, r: std::ops::Range<usize>) -> Result<(), UrlError> {
        let bytes = self.src.as_bytes();
        if bytes[r.clone()].iter().any(|&b| !b.is_ascii_digit()) {
            return Err(UrlError::BadPort);
        }
        if r.start < r.end {
            self.port = Some(r);
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Url<'a>, UrlError> {
        match self.phase {
            Phase::Hunt => return Err(UrlError::NoScheme), // no ':' anywhere
            Phase::Authority => {
                // Authority ran to the end of the input.
                self.auth_end = self.src.len();
                self.finalize_authority()?;
                self.tail_start = self.auth_end;
            }
            Phase::Tail => {}
        }
        let len = self.src.len();
        // Opaque URLs keep the v0.7 initializer's empty host range
        // (`after..after`); hierarchical hosts were set by `finalize_authority`.
        let host = if self.hierarchical {
            self.host
        } else {
            let after = self.scheme_end + 1;
            after..after
        };
        let userinfo = self.last_at.map(|a| self.auth_start..a);
        let query = self.query_start.map(|q| q..self.query_end);
        let fragment = self.fragment_start.map(|f| f..len);
        Ok(Url {
            src: self.src,
            scheme_end: self.scheme_end,
            hierarchical: self.hierarchical,
            userinfo,
            host,
            port: self.port,
            path: self.tail_start..self.path_end,
            query,
            fragment,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_url_all_parts() {
        let u =
            Url::parse("http://user:pw@WWW.Example.com:8080/docs/index.html?q=1&x=2#top").unwrap();
        assert_eq!(u.scheme(), "http");
        assert_eq!(u.userinfo(), Some("user:pw"));
        assert_eq!(u.host(), "WWW.Example.com");
        assert_eq!(u.host_normalized(), "www.example.com");
        assert_eq!(u.port(), Some("8080"));
        assert_eq!(u.path(), "/docs/index.html");
        assert_eq!(u.query(), Some("q=1&x=2"));
        assert_eq!(u.fragment(), Some("top"));
        assert_eq!(
            u.as_str(),
            "http://user:pw@WWW.Example.com:8080/docs/index.html?q=1&x=2#top"
        );
    }

    #[test]
    fn bare_host_no_port_query_fragment() {
        let u = Url::parse("https://example.com").unwrap();
        assert_eq!(u.scheme(), "https");
        assert_eq!(u.userinfo(), None);
        assert_eq!(u.host(), "example.com");
        assert_eq!(u.port(), None);
        assert_eq!(u.path(), "/"); // empty hierarchical path normalizes to "/"
        assert_eq!(u.query(), None);
        assert_eq!(u.fragment(), None);
    }

    #[test]
    fn empty_path_becomes_slash_but_query_kept() {
        let u = Url::parse("https://example.com?x=1").unwrap();
        assert_eq!(u.path(), "/");
        assert_eq!(u.query(), Some("x=1"));
        assert_eq!(u.fragment(), None);
    }

    #[test]
    fn ipv6_with_port() {
        let u = Url::parse("http://[2001:db8::1]:443/x").unwrap();
        assert_eq!(u.host(), "[2001:db8::1]");
        assert_eq!(u.port(), Some("443"));
        assert_eq!(u.path(), "/x");
        // Zone id / bare literal without port
        let v = Url::parse("http://[::ffff:127.0.0.1]/y").unwrap();
        assert_eq!(v.host(), "[::ffff:127.0.0.1]");
        assert_eq!(v.port(), None);
        assert_eq!(v.path(), "/y");
    }

    #[test]
    fn garbage_port_is_bad_port() {
        assert!(matches!(
            Url::parse("http://host:80a/x"),
            Err(UrlError::BadPort)
        ));
        assert!(matches!(
            Url::parse("http://host:8 0/x"),
            Err(UrlError::BadPort)
        ));
    }

    #[test]
    fn missing_scheme_is_no_scheme() {
        assert!(matches!(
            Url::parse("example.com/path"),
            Err(UrlError::NoScheme)
        ));
        assert!(matches!(Url::parse("//host/path"), Err(UrlError::NoScheme)));
        assert!(matches!(Url::parse(""), Err(UrlError::NoScheme)));
        assert!(matches!(Url::parse("1http://x"), Err(UrlError::BadScheme)));
    }

    #[test]
    fn file_scheme_allows_empty_host() {
        let u = Url::parse("file:///home/user/report.txt").unwrap();
        assert_eq!(u.scheme(), "file");
        assert_eq!(u.host(), "");
        assert_eq!(u.path(), "/home/user/report.txt");
        // But http requires a host.
        assert!(matches!(Url::parse("http:///path"), Err(UrlError::NoHost)));
    }

    #[test]
    fn mailto_opaque_path() {
        let u = Url::parse("mailto:webmaster@example.org?subject=hi").unwrap();
        assert_eq!(u.scheme(), "mailto");
        assert_eq!(u.userinfo(), None);
        assert_eq!(u.host(), "");
        assert_eq!(u.port(), None);
        assert_eq!(u.path(), "webmaster@example.org"); // opaque: no "/" prefix
        assert_eq!(u.query(), Some("subject=hi"));
    }

    #[test]
    fn empty_port_and_fragment_only_authority() {
        // "host:" — empty port is legal and normalized to None.
        let u = Url::parse("ftp://host:/pub").unwrap();
        assert_eq!(u.host(), "host");
        assert_eq!(u.port(), None);
        assert_eq!(u.path(), "/pub");
        // Fragment terminates the authority (no '/' or '?' at all).
        let v = Url::parse("https://example.org#sec-2").unwrap();
        assert_eq!(v.host(), "example.org");
        assert_eq!(v.path(), "/");
        assert_eq!(v.fragment(), Some("sec-2"));
    }

    #[test]
    fn host_normalized_borrows_when_already_lowercase() {
        let src = "https://already-lower.example.com/x";
        let u = Url::parse(src).unwrap();
        match u.host_normalized() {
            std::borrow::Cow::Borrowed(h) => {
                assert_eq!(h, "already-lower.example.com");
                // Zero-copy proof: borrowed slice points into the source
                // (address arithmetic only — no unsafe deref needed).
                assert_eq!(h.as_ptr() as usize, src.as_ptr() as usize + 8);
            }
            other => panic!("expected Borrowed, got {other:?}"),
        }
        let mixed = Url::parse("https://Mixed.Case.COM/x").unwrap();
        assert!(matches!(
            mixed.host_normalized(),
            std::borrow::Cow::Owned(_)
        ));
        assert_eq!(mixed.host_normalized(), "mixed.case.com");
    }
}
