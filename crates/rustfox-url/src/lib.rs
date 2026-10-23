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
//! * **SIMD splits**: authority/path termination (`/`, `?`, `#`) and the
//!   scheme terminator scan use `rustfox_simd::find_byte` / `find_any2`
//!   (AVX2: 32 bytes/step). URL components are short, so the win is not raw
//!   bandwidth but *latency*: one vector compare replaces a branchy byte
//!   loop, and the branch predictor stops being the bottleneck on random
//!   URL mixes.
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
    pub fn parse(src: &'a str) -> Result<Url<'a>, UrlError> {
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
