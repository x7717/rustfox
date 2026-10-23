//! rustfox-http1: zero-copy HTTP/1.1 response-header parsing (kernel necko).
//!
//! # Where this sits in Gecko
//!
//! necko's `nsHttpHeaderArray::ParseLine` + `nsHttpTransaction` line
//! scanning run for every response of every request — the browser's
//! network hot path. The C++ pipeline tokenizes headers into an owned,
//! growable header array with per-header atom names, and the status line
//! goes through `nsCRT::strtok`-style string chopping. On a page with
//! hundreds of subresources, per-header allocation and case-folding are
//! measurable malloc pressure (see necko profiles: `ParseHeaders`).
//!
//! # This design
//!
//! * **Zero-copy**: `Response::parse` borrows the raw socket buffer and
//!   returns slices into it — version, reason, and every header name and
//!   value are `&'a [u8]`. The head length is returned so the caller can
//!   hand the remainder to the body path without any copy.
//! * **SIMD line scanning**: the head terminator `\r\n\r\n` and every
//!   line end are found with `rustfox_simd::find_byte(b'\n')` (AVX2,
//!   32 bytes/step) plus a 3-byte neighbor check; header names locate
//!   their `:` the same way. Header blocks are small, so this buys
//!   latency (one vector compare vs a branchy loop), not bandwidth.
//! * **Status digits without `str::parse`**: three ASCII digits are
//!   decoded with shifts/multiplies — no `Result` plumbing on the hot
//!   path, no locale/format machinery.
//! * **Scratch-free inline headers**: `HeaderList` keeps 32 headers in
//!   an inline array (typical responses have 5–15) and spills to a `Vec`
//!   only beyond that; `get()` compares names ASCII case-insensitively
//!   in place — HTTP field names are case-insensitive per RFC 9110 §5.1,
//!   so no canonicalized copies are ever made.
//!
//! Scope: response head parsing only (status line + headers + head
//! length). Chunked-body decoding, header folding continuation lines
//! (obsolete line folding, RFC 9112 §5.2) and connection management are
//! downstream concerns.

use rustfox_simd::find_byte;

/// Failure modes of [`Response::parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpError {
    /// No `\r\n\r\n` found: head not fully buffered yet (call again when
    /// more bytes arrive — necko's `TransactionParser` semantic).
    NoTerminator,
    /// Status line is malformed (no version/SP, status not 3 digits).
    BadStatus,
    /// A header line has no `:` or an empty field-name.
    BadHeader,
}

/// One header: raw (non-canonicalized) name and OWS-trimmed value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header<'a> {
    pub name: &'a [u8],
    pub value: &'a [u8],
}

/// Inline-first header list: 32 slots cover essentially every real
/// response; the spill `Vec` only materializes past that (and keeps its
/// capacity when the list is reused). `Option<Header>` costs one extra
/// discriminant byte at most; `Header` itself is two fat pointers.
#[derive(Debug, Default, Clone)]
pub struct HeaderList<'a> {
    inline: [Option<Header<'a>>; 32],
    n: usize,
    spill: Vec<Header<'a>>,
}

impl<'a> HeaderList<'a> {
    pub fn new() -> Self {
        HeaderList {
            inline: Default::default(),
            n: 0,
            spill: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.n = 0;
        self.spill.clear();
    }

    pub fn push(&mut self, h: Header<'a>) {
        if self.n < 32 {
            self.inline[self.n] = Some(h);
        } else {
            self.spill.push(h);
        }
        self.n += 1;
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = Header<'a>> + '_ {
        let inline = self.inline.iter().take(self.n.min(32)).filter_map(|h| *h);
        let spill = self.spill.iter().copied();
        inline.chain(spill)
    }

    /// ASCII case-insensitive lookup (`"content-length"` finds
    /// `"Content-Length"`), comparing in place — no allocation.
    pub fn get(&self, name: &[u8]) -> Option<&'a [u8]> {
        self.iter()
            .find(|h| eq_ignore_ascii_case(h.name, name))
            .map(|h| h.value)
    }
}

/// Parsed response head: zero-copy slices into the source buffer.
#[derive(Debug, Clone)]
pub struct Response<'a> {
    /// Status-line protocol token, e.g. `HTTP/1.1`.
    pub version: &'a [u8],
    /// Three ASCII digits decoded arithmetically.
    pub status: u16,
    /// Reason phrase (may be empty), after the single SP.
    pub reason: &'a [u8],
    headers: HeaderList<'a>,
}

impl<'a> Response<'a> {
    /// Parse one response head from `src`.
    ///
    /// Returns the parsed [`Response`] and the number of bytes consumed
    /// (everything through the `\r\n\r\n` terminator) so the caller can
    /// continue with the body on the same buffer.
    pub fn parse(src: &'a [u8]) -> Result<(Response<'a>, usize), HttpError> {
        let head_len = find_head_len(src).ok_or(HttpError::NoTerminator)?;
        let head = &src[..head_len];

        // ---- status line -------------------------------------------------
        let nl = find_byte(head, b'\n').ok_or(HttpError::BadStatus)?;
        let mut le = nl;
        if le > 0 && head[le - 1] == b'\r' {
            le -= 1;
        }
        let line = &head[..le];
        let sp = find_byte(line, b' ').ok_or(HttpError::BadStatus)?;
        let version = &line[..sp];
        let rest = &line[sp + 1..];
        if rest.len() < 3 {
            return Err(HttpError::BadStatus);
        }
        let d = &rest[..3];
        if !d.iter().all(|b| b.is_ascii_digit()) {
            return Err(HttpError::BadStatus);
        }
        // (b - '0') via subtraction; no str::parse on the hot path.
        let status =
            ((d[0] - b'0') as u16) * 100 + ((d[1] - b'0') as u16) * 10 + (d[2] - b'0') as u16;
        let after = &rest[3..];
        let reason: &[u8] = if after.is_empty() {
            &[]
        } else if after[0] == b' ' {
            &after[1..]
        } else {
            return Err(HttpError::BadStatus);
        };

        // ---- header lines ------------------------------------------------
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
                break; // the terminator's empty line
            }
            let colon = find_byte(hline, b':').ok_or(HttpError::BadHeader)?;
            // Name: trim OWS on both edges; empty name is a protocol error.
            let mut ne = colon;
            while ne > 0 && is_ows(hline[ne - 1]) {
                ne -= 1;
            }
            let mut ns = 0;
            while ns < ne && is_ows(hline[ns]) {
                ns += 1;
            }
            if ns >= ne {
                return Err(HttpError::BadHeader);
            }
            // Value: trim leading and trailing OWS (RFC 9110 field-value).
            let mut vs = colon + 1;
            while vs < hline.len() && is_ows(hline[vs]) {
                vs += 1;
            }
            let mut ve = hline.len();
            while ve > vs && is_ows(hline[ve - 1]) {
                ve -= 1;
            }
            headers.push(Header {
                name: &hline[ns..ne],
                value: &hline[vs..ve],
            });
        }

        Ok((
            Response {
                version,
                status,
                reason,
                headers,
            },
            head_len,
        ))
    }

    /// First header matching `name` (ASCII case-insensitive).
    #[inline]
    pub fn header(&self, name: &[u8]) -> Option<&'a [u8]> {
        self.headers.get(name)
    }

    /// All headers in wire order.
    #[inline]
    pub fn headers(&self) -> impl Iterator<Item = Header<'a>> + '_ {
        self.headers.iter()
    }

    /// Number of headers parsed.
    #[inline]
    pub fn n_headers(&self) -> usize {
        self.headers.len()
    }
}

/// Length of the head including the `\r\n\r\n` terminator.
///
/// Scan for `'\n'` with SIMD; each hit is the terminator's final byte iff
/// its three predecessors are `\r \n \r`. Headers are short, so this
/// check-per-`'\n'` loop runs a handful of iterations per response.
fn find_head_len(src: &[u8]) -> Option<usize> {
    let mut from = 0usize;
    loop {
        let i = find_byte(&src[from..], b'\n')? + from;
        if i >= 3 && src[i - 1] == b'\r' && src[i - 2] == b'\n' && src[i - 3] == b'\r' {
            return Some(i + 1);
        }
        from = i + 1;
    }
}

/// OWS: optional whitespace (SP / HTAB) per RFC 9110 §5.6.3.
#[inline]
fn is_ows(c: u8) -> bool {
    c == b' ' || c == b'\t'
}

/// ASCII case-insensitive equality over bytes.
#[inline]
fn eq_ignore_ascii_case(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.eq_ignore_ascii_case(y))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: &[u8] = b"HTTP/1.1 200 OK\r\n\
Server: nginx/1.24.0\r\n\
Content-Type: text/html; charset=utf-8\r\n\
Content-Length: 1234\r\n\
Cache-Control: max-age=60\r\n\
Connection: keep-alive\r\n\
X-Trace: trace-7-42\r\n\
\r\n";

    #[test]
    fn full_response_six_headers() {
        let (resp, used) = Response::parse(FULL).unwrap();
        assert_eq!(used, FULL.len());
        assert_eq!(resp.version, b"HTTP/1.1");
        assert_eq!(resp.status, 200);
        assert_eq!(resp.reason, b"OK");
        assert_eq!(resp.n_headers(), 6);
        // Case-insensitive lookups.
        assert_eq!(resp.header(b"content-length"), Some(&b"1234"[..]));
        assert_eq!(resp.header(b"CONTENT-LENGTH"), Some(&b"1234"[..]));
        assert_eq!(resp.header(b"Content-Length"), Some(&b"1234"[..]));
        assert_eq!(resp.header(b"server"), Some(&b"nginx/1.24.0"[..]));
        assert_eq!(resp.header(b"x-trACE"), Some(&b"trace-7-42"[..]));
        assert_eq!(resp.header(b"missing"), None);
        // Wire-order iteration (first three names, wire order).
        let names: Vec<&[u8]> = resp.headers().map(|h| h.name).collect();
        assert_eq!(
            names[..3],
            [&b"Server"[..], b"Content-Type", b"Content-Length"]
        );
    }

    #[test]
    fn header_count_and_values_wire_order() {
        let (resp, _) = Response::parse(FULL).unwrap();
        let hs: Vec<(String, String)> = resp
            .headers()
            .map(|h| {
                (
                    String::from_utf8(h.name.to_vec()).unwrap(),
                    String::from_utf8(h.value.to_vec()).unwrap(),
                )
            })
            .collect();
        assert_eq!(hs.len(), 6);
        assert_eq!(hs[0], ("Server".into(), "nginx/1.24.0".into()));
        assert_eq!(hs[2], ("Content-Length".into(), "1234".into()));
        assert_eq!(hs[4], ("Connection".into(), "keep-alive".into()));
        assert_eq!(hs[5], ("X-Trace".into(), "trace-7-42".into()));
    }

    #[test]
    fn missing_terminator_is_no_terminator() {
        assert!(matches!(
            Response::parse(b"HTTP/1.1 200 OK\r\nA: b\r\n"),
            Err(HttpError::NoTerminator)
        ));
        assert!(matches!(
            Response::parse(b"HTTP/1.1 200 OK\r\nA: b\r\n\r"),
            Err(HttpError::NoTerminator)
        ));
        assert!(matches!(Response::parse(b""), Err(HttpError::NoTerminator)));
    }

    #[test]
    fn response_without_headers() {
        let src = b"HTTP/1.1 404 Not Found\r\n\r\n";
        let (resp, used) = Response::parse(src).unwrap();
        assert_eq!(used, src.len());
        assert_eq!(resp.status, 404);
        assert_eq!(resp.reason, b"Not Found");
        assert!(resp.headers.is_empty());
        assert_eq!(resp.header(b"anything"), None);
    }

    #[test]
    fn empty_reason_variants() {
        // Trailing SP then CRLF: reason = "".
        let (r, _) = Response::parse(b"HTTP/1.1 204 \r\n\r\n").unwrap();
        assert_eq!(r.status, 204);
        assert_eq!(r.reason, b"");
        // No SP at all after digits: reason = "".
        let (r2, _) = Response::parse(b"HTTP/1.1 204\r\n\r\n").unwrap();
        assert_eq!(r2.status, 204);
        assert_eq!(r2.reason, b"");
    }

    #[test]
    fn body_after_head_head_len_exact() {
        let head = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n";
        let src = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello world";
        let (resp, used) = Response::parse(src).unwrap();
        assert_eq!(used, head.len());
        assert_eq!(&src[used..], b"hello world");
        assert_eq!(resp.header(b"content-length"), Some(&b"5"[..]));
    }

    #[test]
    fn ows_trimming() {
        let src = b"HTTP/1.1 200 OK\r\nName :   value  \r\n\tSpaced\t:  \t v \t \r\n\r\n";
        let (resp, _) = Response::parse(src).unwrap();
        assert_eq!(resp.header(b"Name"), Some(&b"value"[..]));
        assert_eq!(resp.header(b"name"), Some(&b"value"[..])); // case-insensitive get
        assert_eq!(resp.header(b"spaced"), Some(&b"v"[..]));
    }

    #[test]
    fn malformed_inputs() {
        // Status not 3 digits.
        assert!(matches!(
            Response::parse(b"HTTP/1.1 2x0 OK\r\n\r\n"),
            Err(HttpError::BadStatus)
        ));
        assert!(matches!(
            Response::parse(b"HTTP/1.1 20 OK\r\n\r\n"),
            Err(HttpError::BadStatus)
        ));
        assert!(matches!(
            Response::parse(b"HTTP/1.1\r\n\r\n"),
            Err(HttpError::BadStatus)
        ));
        // Junk after digits without SP.
        assert!(matches!(
            Response::parse(b"HTTP/1.1 200x\r\n\r\n"),
            Err(HttpError::BadStatus)
        ));
        // Header line without ':'.
        assert!(matches!(
            Response::parse(b"HTTP/1.1 200 OK\r\nno-colon-here\r\n\r\n"),
            Err(HttpError::BadHeader)
        ));
        // Header with empty name.
        assert!(matches!(
            Response::parse(b"HTTP/1.1 200 OK\r\n : v\r\n\r\n"),
            Err(HttpError::BadHeader)
        ));
    }

    #[test]
    fn headers_spill_past_thirty_two() {
        let mut hl = HeaderList::new();
        for _ in 0..40 {
            hl.push(Header {
                name: b"h",
                value: b"v",
            });
        }
        assert_eq!(hl.len(), 40);
        assert_eq!(hl.iter().count(), 40);
        hl.push(Header {
            name: b"late",
            value: b"spilled",
        });
        assert_eq!(hl.get(b"LATE"), Some(&b"spilled"[..]));
        hl.clear();
        assert!(hl.is_empty());
        assert_eq!(hl.get(b"late"), None);
    }
}
