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
