//! rustfox-html: zero-copy SIMD HTML tokenizer (kernel `html5ever`).
//!
//! # Where this sits in Gecko
//!
//! Every byte of every page flows through the HTML tokenizer
//! (`html5ever` in Servo / `nsHtml5Tokenizer` in Gecko). The tokenizer's
//! inner loops are byte classification: find the next `<` in text, find
//! `>`/`=`/whitespace inside a tag, match attribute-value quote runs.
//! The upstream tokenizer materializes each attribute name/value as an
//! owned, case-folded string (`nsHtml5AttributeName::create`, attribute
//! name interning) before the tree builder ever looks at them — a stable
//! stream of small mallocs on the hottest parse path.
//!
//! # This design
//!
//! * **Zero-copy tokens**: every token is a `&'a [u8]` slice of the input
//!   buffer. No name interning, no case-folding copies, no per-token
//!   allocation. Tags/attributes are compared later via
//!   [`name_eq_folded`] / [`name_hash_folded`] — the ASCII case fold
//!   (`| 0x20`) happens *inside the comparison*, not in a buffer.
//! * **SIMD scans**: text runs scan for `<` and tag interiors scan for
//!   structural bytes through `rustfox_simd::find_byte` (AVX2, 32
//!   bytes/step). Text nodes between tags are the bulk of a real page, so
//!   the `find('<')` scan alone covers most of the tokenizer's time.
//! * **Scratch attribute buffer**: the caller passes a reusable
//!   [`Attrs`] to every [`Tokenizer::next_token`] call. One buffer lives
//!   for the whole parse and is `clear()`ed per start tag (with the
//!   16-slot inline array absorbing the typical 0–4 attributes), instead
//!   of one `Vec` per tag. This is the classic scratch-pattern win:
//!   allocation count goes from O(tags) to O(1).
//! * **Simplified HTML5 state machine**: data / tag-open / tag-name /
//!   before-attribute / attribute-name / after-attribute / attribute-
//!   value (quoted + unquoted) / markup-declaration / comment / doctype /
//!   end-tag. Foreign content, CDATA, character references (`&amp;`) and
//!   error-recovery quirks are downstream concerns — the fast path
//!   (well-formed ASCII-heavy markup) is what is benchmarked here.
//!
//! Attribute names are *not* lowercased into storage: HTML tag/attr
//! comparison is ASCII case-insensitive, so the fold is deferred to
//! [`name_eq_folded`]/[`name_hash_folded`], keeping every byte of the
//! source buffer untouched.
//!
//! # Fused API: flat event arena
//!
//! The pull API pays per-token interface costs that dominate token-dense
//! pages: a 500 KB dense page is ~33k [`Tokenizer::next_token`] calls,
//! each constructing a [`Token`], resetting the scratch [`Attrs`] and
//! re-dispatching through the enum. Iteration 6 measured the result:
//! **0.73× vs a fused byte-wise loop** on dense markup (SIMD text scans
//! win it back on prose, 1.86×) — the SIMD gain is real, the interface
//! tax eats it.
//!
//! [`tokenize_into`] removes the interface entirely: **one** fused loop
//! over the buffer emits flat POD [`Event`]s into [`TokenBuf::events`]
//! and pushes attributes into the [`TokenBuf::attrs`] arena — no
//! per-token calls, no per-token allocations, and one reusable buffer
//! for the whole parse. Token semantics are byte-for-byte those of the
//! pull [`Tokenizer`] (same simplified HTML5 state machine, same
//! truncation behavior); the parity oracle in the test suite asserts
//! exact range equality against the pull tokenizer on hand fixtures,
//! LCG-generated markup, truncated inputs and the bench pages.
//!
//! Measured (500 KB pages, criterion): fused beats the pull API on both
//! fixtures — dense ~1.05–1.15× (408–414 vs 435–447 µs), prose
//! ~1.02–1.07× (93–99 vs 99–101 µs) — and reaches **~2.0× over a fused
//! byte-wise loop on prose**. On token-dense markup it stays
//! ~1.2–1.25× *behind* the byte-wise loop: the scanner itself matches
//! naive (SIMD cannot pay off on 1–15-byte runs), so the remaining gap
//! is the fused API's materialize-then-consume cost — ~1 MB of arena
//! stores per 500 KB page (~113 µs isolated push floor, ~29 µs net
//! after overlap) plus a ~36 µs consumer pass, where the byte-wise
//! baseline counts inline with zero extra memory traffic.

use rustfox_simd::find_byte;

/// One attribute: raw (non-folded) name and optionally its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attr<'a> {
    pub name: &'a [u8],
    /// `None` for boolean attributes (`disabled`), `Some` otherwise
    /// (the value may legitimately be the empty slice: `id=""`).
    pub value: Option<&'a [u8]>,
}

/// Reusable scratch storage for the attributes of one start tag.
///
/// 16 slots inline (a `#[derive(Default)]`-able array of `Option<Attr>`;
/// `Option<&[u8]>` is niche-optimized to the same 16 bytes as `&[u8]`),
/// spilling to a `Vec` only past 16 — which real markup almost never
/// needs. The same instance is reused across every tag of a parse:
/// `clear()` just resets the counter, no deallocation.
#[derive(Debug, Default)]
pub struct Attrs<'a> {
    inline: [Option<Attr<'a>>; 16],
    n: usize,
    spill: Vec<Attr<'a>>,
}

impl<'a> Attrs<'a> {
    pub fn new() -> Self {
        Attrs {
            inline: Default::default(),
            n: 0,
            spill: Vec::new(),
        }
    }

    /// Reset for reuse. Keeps the spill capacity (steady-state parses
    /// stop allocating after the first oversized tag).
    #[inline]
    pub fn clear(&mut self) {
        self.n = 0;
        self.spill.clear();
    }

    #[inline]
    pub fn push(&mut self, a: Attr<'a>) {
        if self.n < 16 {
            self.inline[self.n] = Some(a);
        } else {
            self.spill.push(a);
        }
        self.n += 1;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = Attr<'a>> + '_ {
        let inline = self.inline.iter().take(self.n.min(16)).filter_map(|a| *a);
        let spill = self.spill.iter().copied();
        inline.chain(spill)
    }

    /// Exact (byte-for-byte, case-sensitive) name lookup.
    pub fn get(&self, name: &[u8]) -> Option<&'a [u8]> {
        self.iter().find(|a| a.name == name).and_then(|a| a.value)
    }
}

/// A token: every payload is a borrow of the source buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token<'a> {
    /// Raw bytes between tags (unfiltered — `&amp;` decoding is downstream).
    Text(&'a [u8]),
    /// `<name ...>`; attributes land in the caller's scratch [`Attrs`].
    StartTag {
        name: &'a [u8],
        self_closing: bool,
    },
    /// `</name>`
    EndTag(&'a [u8]),
    /// `<!-- ... -->` contents without the delimiters.
    Comment(&'a [u8]),
    /// `<!...>` contents without the delimiters.
    Doctype(&'a [u8]),
    Eof,
}

/// Simplified HTML5 tokenizer over a byte buffer.
pub struct Tokenizer<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    pub fn new(src: &'a [u8]) -> Self {
        Tokenizer { src, pos: 0 }
    }

    /// Produce the next token, stashing start-tag attributes into
    /// `attrs` (cleared by this call). The caller reuses one `Attrs`
    /// for the whole stream — the scratch pattern that removes
    /// per-tag allocations entirely.
    #[inline(always)]
    pub fn next_token(&mut self, attrs: &mut Attrs<'a>) -> Token<'a> {
        let n = self.src.len();
        if self.pos >= n {
            return Token::Eof;
        }
        if self.src[self.pos] != b'<' {
            // Data state: bulk of real pages. One SIMD scan for '<'.
            let end = find_byte(&self.src[self.pos..], b'<').map_or(n, |i| self.pos + i);
            let tok = Token::Text(&self.src[self.pos..end]);
            self.pos = end;
            return tok;
        }
        // We sit on '<'. Dispatch on the next byte.
        let next = self.src.get(self.pos + 1).copied();
        match next {
            // "<!--" → comment up to "-->" (or EOF: truncated comment).
            Some(b'!') if self.src[self.pos..].starts_with(b"<!--") => {
                let start = self.pos + 4;
                let end = find_sub(&self.src[start..], b"-->").map_or(n, |i| start + i);
                let tok = Token::Comment(&self.src[start..end]);
                self.pos = (end + 3).min(n);
                tok
            }
            // "<!" (not comment) → doctype/CDATA-ish blob up to '>'.
            Some(b'!') => {
                let start = self.pos + 2;
                let end = find_byte(&self.src[start..], b'>').map_or(n, |i| start + i);
                let tok = Token::Doctype(&self.src[start..end]);
                self.pos = (end + 1).min(n);
                tok
            }
            // "</name>" — name chars, then skip to '>'.
            Some(b'/') => {
                let start = self.pos + 2;
                let mut i = start;
                while i < n && is_tag_name_byte(self.src[i]) {
                    i += 1;
                }
                let name = &self.src[start..i];
                let end = find_byte(&self.src[i..], b'>').map_or(n, |j| i + j);
                self.pos = (end + 1).min(n);
                Token::EndTag(name)
            }
            // "<name ..." → start tag + attribute loop.
            Some(c) if c.is_ascii_alphabetic() => {
                let start = self.pos + 1;
                let mut i = start;
                while i < n && is_tag_name_byte(self.src[i]) {
                    i += 1;
                }
                let name = &self.src[start..i];
                let self_closing = self.parse_attrs(i, attrs);
                Token::StartTag { name, self_closing }
            }
            // Lone '<' before anything name-like: literal text byte.
            _ => {
                let tok = Token::Text(&self.src[self.pos..self.pos + 1]);
                self.pos += 1;
                tok
            }
        }
    }

    /// Attribute loop starting just after the tag name (at `i`).
    /// Returns the self-closing flag.
    #[inline]
    fn parse_attrs(&mut self, mut i: usize, attrs: &mut Attrs<'a>) -> bool {
        let n = self.src.len();
        attrs.clear();
        let mut self_closing = false;
        loop {
            // before-attribute-name: skip whitespace
            while i < n && is_ws(self.src[i]) {
                i += 1;
            }
            if i >= n {
                // Truncated tag: emit what we have and consume the rest,
                // otherwise the tokenizer would re-scan the same '<' forever.
                self.pos = n;
                return self_closing;
            }
            match self.src[i] {
                b'>' => {
                    self.pos = i + 1;
                    return self_closing;
                }
                b'/' => {
                    // "/>" → self-closing; a stray '/' is skipped.
                    self_closing = true;
                    i += 1;
                }
                _ => {
                    // attribute name up to ws / '=' / '>' / '/'
                    let name_start = i;
                    while i < n {
                        let c = self.src[i];
                        if is_ws(c) || matches!(c, b'=' | b'>' | b'/') {
                            break;
                        }
                        i += 1;
                    }
                    let name = &self.src[name_start..i];
                    // after-attribute-name: optional "= value"
                    let ws_end = i;
                    while i < n && is_ws(self.src[i]) {
                        i += 1;
                    }
                    let value = if i < n && self.src[i] == b'=' {
                        i += 1;
                        while i < n && is_ws(self.src[i]) {
                            i += 1;
                        }
                        let v = if i < n && (self.src[i] == b'"' || self.src[i] == b'\'') {
                            let q = self.src[i];
                            i += 1;
                            let vs = i;
                            let end = find_byte(&self.src[i..], q).map_or(n, |j| i + j);
                            i = (end + 1).min(n);
                            &self.src[vs..end.min(n)]
                        } else {
                            let vs = i;
                            while i < n && !is_ws(self.src[i]) && self.src[i] != b'>' {
                                i += 1;
                            }
                            &self.src[vs..i]
                        };
                        Some(v)
                    } else {
                        i = ws_end; // no '=': rewind whitespace before the next attr
                        None
                    };
                    if !name.is_empty() {
                        attrs.push(Attr { name, value });
                    }
                }
            }
        }
    }
}

/// ASCII whitespace per the HTML5 tokenizer.
#[inline]
fn is_ws(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0c)
}

/// Tag-name byte class (simplified: alnum + a few structural chars —
/// enough for real HTML/SVG/MathML element names).
#[inline]
fn is_tag_name_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b':')
}

/// Naive substring search for "-->" in the comment state (runs at most
/// once per comment; not worth vectorizing here).
fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

/// FxHash (rustc-hash) over ASCII case-folded bytes.
///
/// `hash = (hash.rotate_left(5) ^ folded_byte) * SEED` per byte, seeded
/// with the rustc-hash constant. `A-Z` folds via `| 0x20` *inside the
/// hash* — the source name is never copied or lowercased. This is the
/// key the tree builder would use to dispatch `<DIV>` and `<div>` to the
/// same handler without materializing either.
pub fn name_hash_folded(name: &[u8]) -> u64 {
    const SEED: u64 = 0x517c_c1b7_2722_0a95;
    let mut hash = SEED;
    for &b in name {
        let f = if b.is_ascii_uppercase() { b | 0x20 } else { b };
        hash = (hash.rotate_left(5) ^ f as u64).wrapping_mul(SEED);
    }
    hash
}

/// ASCII case-insensitive byte equality (a-z ≡ A-Z), zero allocation.
#[inline]
pub fn name_eq_folded(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.fold_case() == y.fold_case())
}

trait FoldCase {
    fn fold_case(&self) -> u8;
}

impl FoldCase for u8 {
    #[inline]
    fn fold_case(&self) -> u8 {
        if self.is_ascii_uppercase() {
            self | 0x20
        } else {
            *self
        }
    }
}

// ---------------------------------------------------------------------------
// Fused tokenizer: one pass, flat event arena
// ---------------------------------------------------------------------------

/// Event kind: text run between tags (the data state).
pub const KIND_TEXT: u8 = 0;
/// Event kind: `<name ...>` start tag.
pub const KIND_START_TAG: u8 = 1;
/// Event kind: `</name>` end tag.
pub const KIND_END_TAG: u8 = 2;
/// Event kind: `<!-- ... -->` comment (payload without the delimiters).
pub const KIND_COMMENT: u8 = 3;
/// Event kind: `<!...>` doctype/declaration blob (payload without `<!`/`>`).
pub const KIND_DOCTYPE: u8 = 4;
/// Bit 7 of [`Event::kind`]: the self-closing flag of a start tag.
///
/// Set for `/>` — and, exactly like the pull tokenizer, for *any* stray
/// `/` seen inside the tag (`<a / href="x">` reads as self-closing too).
/// Mask it off with [`Event::kind_base`].
pub const KIND_SELF_CLOSING: u8 = 0x80;

/// One flat token event: POD, no references, no heap.
///
/// Layout: `kind: u8` + four `u32` payload words = 17 bytes of
/// information in the natural 4-byte-aligned 20-byte cell (3 padding
/// bytes after `kind`; a separate kind byte plus four `u32`s cannot fit
/// 16). All offsets are `u32`, so source buffers may be up to 4 GiB.
///
/// * `kind` — [`KIND_TEXT`]…[`KIND_DOCTYPE`], `|`-ed with
///   [`KIND_SELF_CLOSING`] on start tags.
/// * `a..b` — half-open `[a, b)` byte range into the source buffer: the
///   text/comment/doctype payload, or the raw (non-folded) tag **name**
///   for start/end tags. Fold only at comparison time
///   ([`name_eq_folded`]/[`name_hash_folded`]) — the same contract as
///   the pull API.
/// * `c..c+d` — for `KIND_START_TAG`: the attribute span in
///   [`TokenBuf::attrs`]. `c` is `attrs.len()` at emit time (so an
///   attribute-less tag is the empty span `c..c`), `d` the attribute
///   count. `0..0` for every other kind.
/// * EOF is implicit: the event stream ends where the buffer ends —
///   exactly where the pull tokenizer reports `Token::Eof`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    /// Kind tag ([`KIND_TEXT`]…), plus [`KIND_SELF_CLOSING`] on start tags.
    pub kind: u8,
    /// Payload/name range start.
    pub a: u32,
    /// Payload/name range end.
    pub b: u32,
    /// Start tag: index of the first attribute in [`TokenBuf::attrs`].
    pub c: u32,
    /// Start tag: number of attributes.
    pub d: u32,
}

impl Event {
    /// Base kind with the self-closing flag masked off (`kind & 0x7F`).
    #[inline]
    pub fn kind_base(&self) -> u8 {
        self.kind & 0x7F
    }

    /// Start-tag self-closing flag (bit 7 of `kind`); `false` for all
    /// other kinds.
    #[inline]
    pub fn is_self_closing(&self) -> bool {
        self.kind & KIND_SELF_CLOSING != 0
    }

    /// Payload bytes `[a, b)` of the source buffer.
    #[inline]
    pub fn bytes<'s>(&self, src: &'s [u8]) -> &'s [u8] {
        &src[self.a as usize..self.b as usize]
    }

    /// The attribute span `[c, c+d)` of `buf.attrs` (empty for
    /// non-start-tags).
    #[inline]
    pub fn attrs<'b>(&self, buf: &'b TokenBuf) -> &'b [RawAttr] {
        &buf.attrs[self.c as usize..(self.c + self.d) as usize]
    }
}

/// One attribute as raw byte ranges into the source buffer (non-folded —
/// the same contract as [`Attr`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawAttr {
    /// `[start, end)` of the attribute name.
    pub name: (u32, u32),
    /// `[start, end)` of the value; `None` for boolean attributes
    /// (`disabled`), `Some((p, p))` for an empty quoted value (`id=""`).
    pub value: Option<(u32, u32)>,
}

impl RawAttr {
    /// Name bytes of the source buffer.
    #[inline]
    pub fn name_bytes<'s>(&self, src: &'s [u8]) -> &'s [u8] {
        &src[self.name.0 as usize..self.name.1 as usize]
    }

    /// Value bytes of the source buffer, or `None` for boolean attributes.
    #[inline]
    pub fn value_bytes<'s>(&self, src: &'s [u8]) -> Option<&'s [u8]> {
        self.value.map(|(s, e)| &src[s as usize..e as usize])
    }
}

/// Flat output arena of the fused tokenizer.
///
/// One [`TokenBuf`] can be reused across parses (the scratch pattern):
/// [`TokenBuf::clear`] resets both vectors' lengths and keeps their
/// capacity, so a steady-state parse of same-shaped pages performs zero
/// allocations. `attrs` grows monotonically during a single
/// [`tokenize_into`] call and start-tag events reference it via
/// `c..c+d`; all ranges are valid until the next `clear()`.
#[derive(Debug, Default)]
pub struct TokenBuf {
    /// Token events in source order.
    pub events: Vec<Event>,
    /// Attribute arena referenced by start-tag events.
    pub attrs: Vec<RawAttr>,
}

impl TokenBuf {
    /// Empty buffer.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset for reuse, keeping both arenas' capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.events.clear();
        self.attrs.clear();
    }
}

/// Tokenize `src` into a fresh [`TokenBuf`].
///
/// Convenience wrapper around [`tokenize_into`] with a `Default` buffer;
/// prefer [`tokenize_into`] with a reused buffer on hot paths.
pub fn tokenize(src: &[u8]) -> TokenBuf {
    let mut buf = TokenBuf::new();
    tokenize_into(src, &mut buf);
    buf
}

// -- byte-class LUT + hybrid scans (fused-loop internals) -------------------

/// Byte-class bitmap for the fused tokenizer's inner loops: one LUT load
/// replaces the 3–4 compares per byte that the pull tokenizer emits for
/// the same predicates (identical byte sets — parity is exact).
const C_WS: u8 = 1 << 0; // HTML5 ASCII whitespace
const C_EQ: u8 = 1 << 1; // '='
const C_GT: u8 = 1 << 2; // '>'
const C_SLASH: u8 = 1 << 3; // '/'
const C_NAME: u8 = 1 << 5; // tag-name byte: ASCII alnum | '-' | '_' | ':'

const C_ATTR_STOP: u8 = C_WS | C_EQ | C_GT | C_SLASH;
const C_UNQ_STOP: u8 = C_WS | C_GT;

static BYTE_CLASS: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut c = 0usize;
    while c < 256 {
        let b = c as u8;
        let mut f = 0u8;
        if matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0C) {
            f |= C_WS;
        }
        if b == b'=' {
            f |= C_EQ;
        }
        if b == b'>' {
            f |= C_GT;
        }
        if b == b'/' {
            f |= C_SLASH;
        }
        if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b':' {
            f |= C_NAME;
        }
        t[c] = f;
        c += 1;
    }
    t
};

/// First index ≥ `from` whose byte carries any of the `class` bits, or
/// `n`. Plain LUT loop: these stop-sets have >2 members, so there is no
/// single-byte SIMD helper — but real attr names / ws runs are ≤16 bytes,
/// where the loop beats any call setup anyway.
#[inline]
fn scan_stop(src: &[u8], from: usize, n: usize, class: u8) -> usize {
    let mut j = from;
    while j < n && BYTE_CLASS[src[j] as usize] & class == 0 {
        j += 1;
    }
    j
}

/// First `needle` at/after `from`, or `n`. Hybrid: a 16-byte SWAR probe
/// (two 8-byte window tests, no per-byte loop, no call setup) covers the
/// short runs that dominate token-dense markup (quoted values, text
/// gaps, `</div>`) at ~1 ns; longer runs fall through to the SIMD
/// `find_byte`.
#[inline]
fn scan_byte(src: &[u8], from: usize, n: usize, needle: u8) -> usize {
    let probe_end = (from + 16).min(n);
    let target = (needle as u64).wrapping_mul(0x0101_0101_0101_0101);
    let mut j = from;
    while j + 8 <= probe_end {
        let mut w = [0u8; 8];
        w.copy_from_slice(&src[j..j + 8]);
        let x = u64::from_le_bytes(w) ^ target;
        let m = x.wrapping_sub(0x0101_0101_0101_0101) & !x & 0x8080_8080_8080_8080;
        if m != 0 {
            return j + (m.trailing_zeros() >> 3) as usize;
        }
        j += 8;
    }
    while j < probe_end && src[j] != needle {
        j += 1;
    }
    if j < probe_end || j >= n {
        return j;
    }
    find_byte(&src[j..], needle).map_or(n, |off| j + off)
}

/// Fused tokenizer: a **single** pass over `src` emitting flat
/// [`Event`]s and attribute ranges into `buf` — no per-token calls, no
/// per-token allocations. `buf` is `clear()`ed first, so one buffer can
/// be reused across parses (capacity is retained; a fresh buffer gets a
/// length-based capacity hint, capped to keep pathological text-only
/// inputs from over-reserving).
///
/// Token semantics are exactly those of [`Tokenizer::next_token`] (the
/// same simplified HTML5 state machine: comments run to `-->`, doctypes
/// to `>`, end/start tags with quoted/unquoted/boolean attributes, a
/// lone `<` is a 1-byte text token, truncated inputs terminate).
pub fn tokenize_into(src: &[u8], buf: &mut TokenBuf) {
    buf.clear();
    // Capacity hint from input length: dense markup runs ~1 event per
    // 12–16 bytes and ~1 attr per 16–24 bytes. One upfront allocation
    // beats log-growth reallocations inside the loop; on a reused buffer
    // this is a no-op. Capped: a text-only page emits ~1 event per few
    // hundred bytes, so an uncapped hint would over-reserve hugely.
    buf.events.reserve((src.len() / 16).min(1 << 16));
    buf.attrs.reserve((src.len() / 24).min(1 << 16));

    let n = src.len();
    let events = &mut buf.events;
    let attrs = &mut buf.attrs;
    let mut i = 0usize;

    while i < n {
        if src[i] != b'<' {
            // Data state: the bulk of real pages. Hybrid scan for '<'.
            let end = scan_byte(src, i, n, b'<');
            events.push(Event {
                kind: KIND_TEXT,
                a: i as u32,
                b: end as u32,
                c: 0,
                d: 0,
            });
            i = end;
            continue;
        }

        // We sit on '<'. Dispatch on the next byte — every arm below is
        // inlined into this single loop (no per-token calls).
        let Some(next) = src.get(i + 1).copied() else {
            // Lone '<' at EOF: a 1-byte text token.
            events.push(Event {
                kind: KIND_TEXT,
                a: i as u32,
                b: (i + 1) as u32,
                c: 0,
                d: 0,
            });
            i += 1;
            continue;
        };
        match next {
            b'!' if src[i..].starts_with(b"<!--") => {
                // "<!--" → comment up to "-->" (or EOF: truncated).
                let start = i + 4;
                let end = find_sub_arrow(&src[start..]).map_or(n, |off| start + off);
                events.push(Event {
                    kind: KIND_COMMENT,
                    a: start as u32,
                    b: end as u32,
                    c: 0,
                    d: 0,
                });
                i = (end + 3).min(n);
            }
            b'!' => {
                // "<!" (not comment) → doctype/CDATA-ish blob up to '>'.
                let start = i + 2;
                let end = scan_byte(src, start, n, b'>');
                events.push(Event {
                    kind: KIND_DOCTYPE,
                    a: start as u32,
                    b: end as u32,
                    c: 0,
                    d: 0,
                });
                i = (end + 1).min(n);
            }
            b'/' => {
                // "</name>": name bytes, then skip to '>'.
                let name_s = i + 2;
                let mut j = name_s;
                while j < n && BYTE_CLASS[src[j] as usize] & C_NAME != 0 {
                    j += 1;
                }
                let name_e = j;
                let end = scan_byte(src, j, n, b'>');
                events.push(Event {
                    kind: KIND_END_TAG,
                    a: name_s as u32,
                    b: name_e as u32,
                    c: 0,
                    d: 0,
                });
                i = (end + 1).min(n);
            }
            first if first.is_ascii_alphabetic() => {
                // "<name ...>": tag name, then the attribute loop — all
                // inline, attributes pushed straight into the arena.
                let name_s = i + 1;
                let mut j = name_s;
                while j < n && BYTE_CLASS[src[j] as usize] & C_NAME != 0 {
                    j += 1;
                }
                let name_e = j;
                let attr_from = attrs.len() as u32;
                let mut self_closing = false;
                loop {
                    // before-attribute-name: skip whitespace
                    while j < n && BYTE_CLASS[src[j] as usize] & C_WS != 0 {
                        j += 1;
                    }
                    if j >= n {
                        // Truncated tag: emit what we have and consume the
                        // rest (the pull tokenizer does the same — otherwise
                        // it would re-scan the same '<' forever).
                        i = n;
                        break;
                    }
                    let class = BYTE_CLASS[src[j] as usize];
                    if class & C_GT != 0 {
                        i = j + 1;
                        break;
                    }
                    if class & C_SLASH != 0 {
                        // "/>" → self-closing; a stray '/' is skipped.
                        self_closing = true;
                        j += 1;
                        continue;
                    }
                    // attribute name up to ws / '=' / '>' / '/'
                    let name_a = j;
                    j = scan_stop(src, j, n, C_ATTR_STOP);
                    let name_b = j;
                    // after-attribute-name: optional "= value"
                    let ws_end = j;
                    while j < n && BYTE_CLASS[src[j] as usize] & C_WS != 0 {
                        j += 1;
                    }
                    let value = if j < n && BYTE_CLASS[src[j] as usize] & C_EQ != 0 {
                        j += 1;
                        while j < n && BYTE_CLASS[src[j] as usize] & C_WS != 0 {
                            j += 1;
                        }
                        if j < n && (src[j] == b'"' || src[j] == b'\'') {
                            let q = src[j];
                            j += 1;
                            let vs = j;
                            let end = scan_byte(src, j, n, q);
                            j = (end + 1).min(n);
                            Some((vs as u32, end as u32))
                        } else {
                            let vs = j;
                            j = scan_stop(src, j, n, C_UNQ_STOP);
                            Some((vs as u32, j as u32))
                        }
                    } else {
                        j = ws_end; // no '=': rewind before the next attr
                        None
                    };
                    if name_b > name_a {
                        attrs.push(RawAttr {
                            name: (name_a as u32, name_b as u32),
                            value,
                        });
                    }
                }
                let kind = if self_closing {
                    KIND_START_TAG | KIND_SELF_CLOSING
                } else {
                    KIND_START_TAG
                };
                events.push(Event {
                    kind,
                    a: name_s as u32,
                    b: name_e as u32,
                    c: attr_from,
                    d: attrs.len() as u32 - attr_from,
                });
            }
            _ => {
                // Lone '<' before anything name-like: literal text byte.
                events.push(Event {
                    kind: KIND_TEXT,
                    a: i as u32,
                    b: (i + 1) as u32,
                    c: 0,
                    d: 0,
                });
                i += 1;
            }
        }
    }
}

/// First `-->` in `hay`, skipping non-candidate runs via
/// `find_byte('-')` (a match can only start at a `-`). Same result as
/// the pull tokenizer's `find_sub`, ~3× fewer byte probes on long
/// comments. Called once per comment, not per token.
#[inline]
fn find_sub_arrow(hay: &[u8]) -> Option<usize> {
    let mut from = 0usize;
    while from + 3 <= hay.len() {
        let f = find_byte(&hay[from..], b'-')? + from;
        if f + 3 > hay.len() {
            return None; // no room for another full candidate
        }
        if hay[f + 1] == b'-' && hay[f + 2] == b'>' {
            return Some(f);
        }
        from = f + 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_sequence() {
        let src = r#"<div id="a" class="x y">hi<!--c--><br/></div>"#;
        let mut tz = Tokenizer::new(src.as_bytes());
        let mut attrs = Attrs::new();
        let t0 = tz.next_token(&mut attrs);
        assert_eq!(
            t0,
            Token::StartTag {
                name: b"div",
                self_closing: false
            }
        );
        let got: Vec<(String, Option<String>)> = attrs
            .iter()
            .map(|a| {
                (
                    String::from_utf8(a.name.to_vec()).unwrap(),
                    a.value.map(|v| String::from_utf8(v.to_vec()).unwrap()),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                ("id".to_string(), Some("a".to_string())),
                ("class".to_string(), Some("x y".to_string())),
            ]
        );
        assert_eq!(attrs.get(b"class"), Some(&b"x y"[..]));
        assert_eq!(attrs.get(b"missing"), None);

        assert_eq!(tz.next_token(&mut attrs), Token::Text(b"hi"));
        assert_eq!(tz.next_token(&mut attrs), Token::Comment(b"c"));
        assert_eq!(
            tz.next_token(&mut attrs),
            Token::StartTag {
                name: b"br",
                self_closing: true
            }
        );
        assert_eq!(tz.next_token(&mut attrs), Token::EndTag(b"div"));
        assert_eq!(tz.next_token(&mut attrs), Token::Eof);
    }

    #[test]
    fn unquoted_and_single_quoted_values() {
        let src = br#"<a href=/item/7 target=_blank rel='next prev'>x</a>"#;
        let mut tz = Tokenizer::new(src);
        let mut attrs = Attrs::new();
        assert_eq!(
            tz.next_token(&mut attrs),
            Token::StartTag {
                name: b"a",
                self_closing: false
            }
        );
        assert_eq!(attrs.get(b"href"), Some(&b"/item/7"[..]));
        assert_eq!(attrs.get(b"target"), Some(&b"_blank"[..]));
        assert_eq!(attrs.get(b"rel"), Some(&b"next prev"[..]));
        assert_eq!(attrs.len(), 3);
        assert_eq!(tz.next_token(&mut attrs), Token::Text(b"x"));
        assert_eq!(tz.next_token(&mut attrs), Token::EndTag(b"a"));
    }

    #[test]
    fn lone_less_than_is_text() {
        let mut tz = Tokenizer::new(b"a < b");
        let mut attrs = Attrs::new();
        assert_eq!(tz.next_token(&mut attrs), Token::Text(b"a "));
        assert_eq!(tz.next_token(&mut attrs), Token::Text(b"<"));
        assert_eq!(tz.next_token(&mut attrs), Token::Text(b" b"));
        assert_eq!(tz.next_token(&mut attrs), Token::Eof);
    }

    #[test]
    fn doctype_and_boolean_attr() {
        let src = b"<!DOCTYPE html><input disabled value=>";
        let mut tz = Tokenizer::new(src);
        let mut attrs = Attrs::new();
        assert_eq!(tz.next_token(&mut attrs), Token::Doctype(b"DOCTYPE html"));
        assert_eq!(
            tz.next_token(&mut attrs),
            Token::StartTag {
                name: b"input",
                self_closing: false
            }
        );
        assert_eq!(attrs.get(b"disabled"), None); // boolean: no value
        assert_eq!(attrs.get(b"value"), Some(&b""[..])); // empty value is Some("")
        assert_eq!(tz.next_token(&mut attrs), Token::Eof);
    }

    #[test]
    fn folded_hash_and_eq() {
        assert_eq!(name_hash_folded(b"DIV"), name_hash_folded(b"div"));
        assert_eq!(name_hash_folded(b"DiV"), name_hash_folded(b"dIv"));
        assert_ne!(name_hash_folded(b"div"), name_hash_folded(b"span"));
        assert!(name_eq_folded(b"DiV", b"div"));
        assert!(!name_eq_folded(b"div", b"divx"));
        assert!(!name_eq_folded(b"abc", b"abd"));
        assert!(name_eq_folded(b"", b""));
    }

    #[test]
    fn attrs_spill_past_sixteen() {
        let mut attrs = Attrs::new();
        for _ in 0..24 {
            attrs.push(Attr {
                name: b"attr",
                value: Some(b"x"),
            });
        }
        assert_eq!(attrs.len(), 24);
        assert_eq!(attrs.iter().count(), 24);
        assert!(!attrs.is_empty());
        // get() must see the spilled entries too.
        attrs.push(Attr {
            name: b"late",
            value: Some(b"spilled"),
        });
        assert_eq!(attrs.len(), 25);
        assert_eq!(attrs.get(b"late"), Some(&b"spilled"[..]));
        // clear() resets both lanes and keeps working.
        attrs.clear();
        assert!(attrs.is_empty());
        attrs.push(Attr {
            name: b"fresh",
            value: None,
        });
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs.get(b"fresh"), None);
    }

    #[test]
    fn truncated_inputs_terminate() {
        for src in [
            "<div class=\"x",
            "<!-- never closed",
            "<!doctype",
            "</div",
            "<a href=\"unterminated",
            "<",
            "<d",
            "plain text",
        ] {
            let mut tz = Tokenizer::new(src.as_bytes());
            let mut attrs = Attrs::new();
            let mut count = 0;
            loop {
                match tz.next_token(&mut attrs) {
                    Token::Eof => break,
                    _ => count += 1,
                }
                assert!(count < 100, "tokenizer loop on {src:?}");
            }
            assert!(count >= 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Fused-vs-pull parity tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod fused_parity {
    use super::*;

    /// Pull-tokenizer output normalized to absolute source ranges — the
    /// ground truth the fused arena must reproduce exactly.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum RefTok {
        Text(usize, usize),
        Start(usize, usize, bool, Vec<RefAttr>),
        End(usize, usize),
        Comment(usize, usize),
        Doctype(usize, usize),
    }

    /// Normalized attribute: raw `(start, end)` name range plus optional
    /// `(start, end)` value range.
    type RefAttr = ((u32, u32), Option<(u32, u32)>);

    /// Offset of a tokenizer slice within `src`. Every pull payload borrows
    /// from `src`, so pointer arithmetic recovers the absolute range
    /// (empty slices keep their in-bounds pointer, so `off` is exact).
    fn off(base: usize, s: &[u8]) -> usize {
        s.as_ptr() as usize - base
    }

    /// Oracle: run the existing pull tokenizer, snapshot attributes per
    /// start tag, and normalize every token to absolute ranges.
    fn ref_sequence(src: &[u8]) -> Vec<RefTok> {
        let base = src.as_ptr() as usize;
        let mut tz = Tokenizer::new(src);
        let mut attrs = Attrs::new();
        let mut out = Vec::new();
        loop {
            match tz.next_token(&mut attrs) {
                Token::Eof => break,
                Token::Text(s) => {
                    let p = off(base, s);
                    out.push(RefTok::Text(p, p + s.len()));
                }
                Token::StartTag { name, self_closing } => {
                    let p = off(base, name);
                    let snapshot: Vec<_> = attrs
                        .iter()
                        .map(|a| {
                            let np = off(base, a.name);
                            (
                                (np as u32, (np + a.name.len()) as u32),
                                a.value.map(|v| {
                                    let vp = off(base, v);
                                    (vp as u32, (vp + v.len()) as u32)
                                }),
                            )
                        })
                        .collect();
                    out.push(RefTok::Start(p, p + name.len(), self_closing, snapshot));
                }
                Token::EndTag(s) => {
                    let p = off(base, s);
                    out.push(RefTok::End(p, p + s.len()));
                }
                Token::Comment(s) => {
                    let p = off(base, s);
                    out.push(RefTok::Comment(p, p + s.len()));
                }
                Token::Doctype(s) => {
                    let p = off(base, s);
                    out.push(RefTok::Doctype(p, p + s.len()));
                }
            }
        }
        out
    }

    fn sequence_from_buf(buf: &TokenBuf) -> Vec<RefTok> {
        buf.events
            .iter()
            .map(|e| match e.kind_base() {
                KIND_TEXT => RefTok::Text(e.a as usize, e.b as usize),
                KIND_START_TAG => RefTok::Start(
                    e.a as usize,
                    e.b as usize,
                    e.is_self_closing(),
                    e.attrs(buf).iter().map(|ra| (ra.name, ra.value)).collect(),
                ),
                KIND_END_TAG => RefTok::End(e.a as usize, e.b as usize),
                KIND_COMMENT => RefTok::Comment(e.a as usize, e.b as usize),
                KIND_DOCTYPE => RefTok::Doctype(e.a as usize, e.b as usize),
                other => panic!("corrupt event kind {other:#x}"),
            })
            .collect()
    }

    fn fused_sequence(src: &[u8]) -> Vec<RefTok> {
        sequence_from_buf(&tokenize(src))
    }

    fn assert_parity(src: &[u8]) {
        let want = ref_sequence(src);
        let got = fused_sequence(src);
        assert_eq!(
            want,
            got,
            "fused != pull for prefix {:?}",
            String::from_utf8_lossy(&src[..src.len().min(96)])
        );
    }

    #[test]
    fn parity_hand_fixtures() {
        let fixtures: &[&[u8]] = &[
            b"<div id=\"a\" class=\"x y\">hi<!--c--><br/></div>",
            b"<a href=/item/7 target=_blank rel='next prev'>x</a>",
            b"a < b",
            b"<!DOCTYPE html><input disabled value=>",
            b"<!---->",
            b"<!-->",
            b"<!->",
            b"<!--",
            b"<!--x--",
            b"<>",
            b"</>",
            b"</ div>",
            b"<",
            b"<d",
            b"<!",
            b"<!-",
            b"<a>",
            b"<a b=c/>",
            b"<a b=c/ d=e>",
            b"<a b= c>",
            b"<a b = c >",
            b"<a b =c>",
            b"<a b= >",
            b"<a b='>",
            b"<a/'q'>",
            b"<a b=\"x>y\">z</a>",
            b"<a b=c d=e f/>",
            b"<input disabled value=><br/>",
            b"<div data-x=a-b:c_d><span class='m n'>t</span></div>",
            b"<a a1=a2 a3 = \"a4\" a5='a6' a7>a8</a9>",
            b"x<y<z<<w>><",
            b"plain text no markup",
            b"<div / class=x>y</div>",
            b"<div />text<span/ >t2</span>",
            b"<a b=\"\">c</a><a d=''>e</a><a f=></a>",
        ];
        for src in fixtures {
            assert_parity(src);
        }
    }

    #[test]
    fn parity_truncated_inputs() {
        // the same 8 strings as truncated_inputs_terminate, plus extra
        // truncation shapes (open quote / open '=' / bare '<!')
        for src in [
            "<div class=\"x",
            "<!-- never closed",
            "<!doctype",
            "</div",
            "<a href=\"unterminated",
            "<",
            "<d",
            "plain text",
            "<a b='",
            "<a b=",
            "<!",
            "</",
            "<div ",
            "<div /",
        ] {
            assert_parity(src.as_bytes());
        }
    }

    // -- deterministic LCG corpus (no `rand` dep) ---------------------------

    /// Mirrors the bench LCG exactly (same constants, same `new` seeding),
    /// so the fixture builders below produce byte-identical pages.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1)
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

    const TAGS: &[&[u8]] = &[
        b"div", b"a", b"span", b"input", b"br", b"P", b"IMG", b"h2", b"x-y:z_",
    ];
    const ATTR_NAMES: &[&[u8]] = &[
        b"class",
        b"id",
        b"href",
        b"data-i",
        b"disabled",
        b"Q",
        b"src",
        b"title",
    ];

    /// One markup piece of kind `kind` (0..14): tags with every attribute
    /// value style (boolean / double-quoted / unquoted / single-quoted
    /// with spaces / '>' inside quotes / whitespace around '='), text
    /// runs, lone '<' runs, comments (closed, empty, unterminated),
    /// doctypes, end tags, structural junk bytes, wide tags.
    fn gen_piece(rng: &mut Lcg, kind: u64, out: &mut Vec<u8>) {
        match kind {
            0 | 1 => {
                out.push(b'<');
                out.extend_from_slice(TAGS[rng.below(TAGS.len() as u64) as usize]);
                for _ in 0..rng.below(6) {
                    out.push(b' ');
                    out.extend_from_slice(ATTR_NAMES[rng.below(ATTR_NAMES.len() as u64) as usize]);
                    match rng.below(6) {
                        0 => {} // boolean attribute
                        1 => out.extend_from_slice(format!("=\"v{}\"", rng.below(1000)).as_bytes()),
                        2 => out.extend_from_slice(format!("=v{}", rng.below(1000)).as_bytes()),
                        3 => {
                            out.extend_from_slice(format!("='s {} x/'", rng.below(100)).as_bytes())
                        }
                        4 => out.extend_from_slice(b"=\"gt>inside\""),
                        _ => out.extend_from_slice(b"  =  spaced"),
                    }
                }
                if rng.below(5) == 0 {
                    out.push(b'/'); // self-closing or stray slash
                }
                out.push(b'>');
            }
            2 => {
                out.extend_from_slice(b"</");
                out.extend_from_slice(TAGS[rng.below(TAGS.len() as u64) as usize]);
                if rng.below(3) == 0 {
                    out.extend_from_slice(b" trail");
                }
                out.push(b'>');
            }
            3 => {
                for _ in 0..3 + rng.below(10) {
                    out.extend_from_slice(b"word ");
                }
            }
            4 => out.extend_from_slice(b"a < b <c d<<e "),
            5 => out.extend_from_slice(b"<!-- c -->"),
            6 => out.extend_from_slice(b"<!---->"),
            7 => out.extend_from_slice(b"<!DOCTYPE html>"),
            8 => out.extend_from_slice(format!("<!doctype {}>", rng.below(1000)).as_bytes()),
            9 => out.extend_from_slice(b"<!-- never closed"),
            10 => out.extend_from_slice(b"<!x blob>"),
            11 => match rng.below(3) {
                0 => out.extend_from_slice(b"</>"),
                1 => out.extend_from_slice(b"</div"),
                _ => out.extend_from_slice(b"</"),
            },
            12 => {
                const JUNK: &[u8] = b"<>=-/\"'!- \t\nabcx";
                for _ in 0..1 + rng.below(14) {
                    out.push(JUNK[rng.below(JUNK.len() as u64) as usize]);
                }
            }
            _ => {
                out.extend_from_slice(b"<m");
                for j in 0..24 {
                    out.extend_from_slice(format!(" a{j}={j}").as_bytes());
                }
                out.push(b'>');
            }
        }
    }

    fn gen_markup(seed: u64, pieces: usize) -> Vec<u8> {
        let mut rng = Lcg(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1);
        let mut out = Vec::with_capacity(pieces * 32);
        for _ in 0..pieces {
            let kind = rng.below(14);
            gen_piece(&mut rng, kind, &mut out);
        }
        out
    }

    #[test]
    fn parity_lcg_corpus() {
        for seed in 1..=8u64 {
            let doc = gen_markup(seed * 0x9E37_79B9, 2000);
            assert_parity(&doc);
        }
    }

    #[test]
    fn parity_each_piece_kind_isolated() {
        // no shape hides behind random ordering: 6 pieces of one kind each
        for kind in 0..14u64 {
            let mut rng = Lcg(0xDEAD_BEEF_u64.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1);
            let mut doc = Vec::new();
            for _ in 0..6 {
                gen_piece(&mut rng, kind, &mut doc);
            }
            assert_parity(&doc);
        }
    }

    // -- bench fixtures (byte-identical mirrors of benches/tokenize.rs) -----

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

    #[test]
    fn parity_bench_fixtures() {
        assert_parity(&build_page_dense());
        assert_parity(&build_page_prose());
    }

    #[test]
    fn tokenize_into_reuses_arena() {
        let src_a = b"<div class=\"a\" disabled>x<!--c--></div>" as &[u8];
        let src_b = b"<p id=q data-n=5>text < more<br/>" as &[u8];

        let mut buf = TokenBuf::new();
        tokenize_into(src_a, &mut buf);
        assert_eq!(sequence_from_buf(&buf), fused_sequence(src_a));

        let cap_events = buf.events.capacity();
        let cap_attrs = buf.attrs.capacity();
        tokenize_into(src_b, &mut buf);
        assert_eq!(sequence_from_buf(&buf), fused_sequence(src_b));
        // capacity retained across reuse (steady state = zero allocation)
        assert!(buf.events.capacity() >= cap_events);
        assert!(buf.attrs.capacity() >= cap_attrs);

        // clear() empties both arenas together
        buf.clear();
        assert!(buf.events.is_empty() && buf.attrs.is_empty());
        assert!(tokenize(b"").events.is_empty());
    }

    #[test]
    fn event_is_flat_pod() {
        // 17 bytes of payload (kind + 4×u32) in the natural 4-byte-aligned
        // cell — the property that matters: Copy POD, no references/heap.
        assert_eq!(std::mem::size_of::<Event>(), 20);
        assert_eq!(std::mem::align_of::<Event>(), 4);

        let buf = tokenize(b"<div id=\"a\">x</div>");
        let e = &buf.events[0];
        assert_eq!(e.kind_base(), KIND_START_TAG);
        assert!(e.bytes(b"<div id=\"a\">x</div>") == b"div");
        assert_eq!(e.attrs(&buf)[0].name_bytes(b"<div id=\"a\">x</div>"), b"id");
        assert_eq!(
            e.attrs(&buf)[0].value_bytes(b"<div id=\"a\">x</div>"),
            Some(b"a" as &[u8])
        );
        // non-start-tag events: empty attr span, plain payload
        let t = &buf.events[1];
        assert_eq!(t.kind_base(), KIND_TEXT);
        assert!(t.attrs(&buf).is_empty());
        assert_eq!(t.bytes(b"<div id=\"a\">x</div>"), b"x");
    }
}
