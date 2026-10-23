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
//! * **Two-pass with recorded line ends**: pass 1 finds the head
//!   terminator `\r\n\r\n` and records the absolute offset of every `\n`
//!   it steps over (the same offsets pass 2 needs for the status and
//!   header lines) into a small stack array; the terminator hit itself is
//!   a single 4-byte compare. On a typical head that removes the second
//!   full round of per-line `\n` scans (~7 of ~15 searches per response).
//!   Each scan is `rustfox_simd::find_byte` — its own hybrid dispatch
//!   (AVX2 ≥ 32 B, inlined SWAR 8–31 B, scalar < 8 B) already matches
//!   short header lines better than a local probe: the line-end `\n`
//!   sits 15–45 B into the range, where one 32-byte AVX2 compare beats
//!   2–6 SWAR steps. Heads with more line ends than the recording cap
//!   simply fall back to per-line scans mid-parse (correctness never
//!   depends on the cap).
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
        let mut ends = [0u32; MAX_LINE_ENDS];
        let mut n_ends = 0usize;
        let head_len = find_head_len(src, &mut ends, &mut n_ends).ok_or(HttpError::NoTerminator)?;
        let head = &src[..head_len];

        // ---- status line -------------------------------------------------
        let mut hit = 0usize;
        let nl = next_line_end(head, 0, &ends, n_ends, &mut hit).ok_or(HttpError::BadStatus)?;
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
            let nl2 =
                next_line_end(head, pos, &ends, n_ends, &mut hit).ok_or(HttpError::BadHeader)?;
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

/// Cap on recorded line-end offsets (see [`find_head_len`]). A head with
/// more `\n`s than this consumes the recorded prefix, then continues
/// with per-line `find_byte` scans — recording degrades, correctness
/// does not (the recorded offsets are a valid prefix of the hit order).
const MAX_LINE_ENDS: usize = 40;

/// Next `\n` at or after `pos` in `head`, consuming the offsets pass 1
/// recorded while they last (they are exactly the `\n` positions from
/// the buffer start in order, and `parse` requests them in that same
/// order), falling back to a plain scan once the recording runs out.
#[inline]
fn next_line_end(
    head: &[u8],
    pos: usize,
    ends: &[u32; MAX_LINE_ENDS],
    n_ends: usize,
    hit: &mut usize,
) -> Option<usize> {
    if *hit < n_ends {
        let e = ends[*hit] as usize;
        *hit += 1;
        debug_assert!(e >= pos, "recorded line ends must be non-decreasing vs pos");
        Some(e)
    } else {
        find_byte(&head[pos..], b'\n').map(|off| pos + off)
    }
}

/// Length of the head including the `\r\n\r\n` terminator, plus the
/// absolute offset of every `\n` stepped over on the way (up to
/// [`MAX_LINE_ENDS`]).
///
/// Each `find_byte` hit is the terminator's final byte iff the 4 bytes
/// ending at it are `\r\n\r\n` — one 4-byte compare per hit instead of
/// three neighbor loads. The recorded offsets feed pass 2, which
/// otherwise re-scans every line end (`\n` hit *k* is the end of line
/// *k*: status line first, then header lines, then the terminator's own
/// `\n`, which yields the empty line that stops the header loop).
#[inline]
fn find_head_len(src: &[u8], ends: &mut [u32; MAX_LINE_ENDS], n_ends: &mut usize) -> Option<usize> {
    let mut from = 0usize;
    *n_ends = 0;
    loop {
        let i = find_byte(&src[from..], b'\n')? + from;
        if *n_ends < MAX_LINE_ENDS && i <= u32::MAX as usize {
            ends[*n_ends] = i as u32;
            *n_ends += 1;
        }
        if i >= 3 && src[i - 3..i + 1] == *b"\r\n\r\n" {
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

    // ---- recorded line ends (two-pass fusion) -----------------------------

    fn head_with_n_headers(n: usize) -> Vec<u8> {
        let mut head = b"HTTP/1.1 200 OK\r\n".to_vec();
        for i in 0..n {
            head.extend_from_slice(format!("X-{i:02}: v{i}\r\n").as_bytes());
        }
        head.extend_from_slice(b"\r\n");
        head
    }

    #[test]
    fn line_ends_at_and_past_recording_cap() {
        // hits = status `\n` + H header `\n`s + terminator `\n` = H + 2.
        // H = 38 → exactly MAX_LINE_ENDS hits: fully recorded path.
        // H = 39 and H = 45 → the recording caps mid-head and the parse
        // finishes on per-line find_byte scans. All must parse identically.
        for h in [MAX_LINE_ENDS - 2, MAX_LINE_ENDS - 1, MAX_LINE_ENDS + 5] {
            let src = head_with_n_headers(h);
            let (resp, used) = Response::parse(&src).unwrap_or_else(|e| panic!("h={h}: {e:?}"));
            assert_eq!(used, src.len(), "h={h}");
            assert_eq!(resp.n_headers(), h, "h={h}");
            assert_eq!(resp.status, 200);
            let mid = format!("X-{:02}", h / 2).into_bytes();
            assert_eq!(
                resp.header(&mid),
                Some(format!("v{}", h / 2).as_bytes()),
                "h={h}"
            );
            let last = format!("X-{:02}", h - 1).into_bytes();
            assert_eq!(
                resp.header(&last),
                Some(format!("v{}", h - 1).as_bytes()),
                "h={h}"
            );
        }
    }

    #[test]
    fn line_ends_no_headers_and_body_tail() {
        // Two recorded hits (status + terminator), empty line stops the
        // loop. The head INCLUDES the \r\n\r\n terminator: 14 + 2 = 16.
        let src = b"HTTP/1.1 204\r\n\r\nbody-here";
        let (resp, used) = Response::parse(src).unwrap();
        assert_eq!(used, 16);
        assert_eq!(resp.n_headers(), 0);
        assert_eq!(&src[used..], b"body-here");
    }
}
