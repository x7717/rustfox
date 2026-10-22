//! Selector model + parser.
//!
//! Grammar (subset of CSS, enough for the hot-path work):
//! ```text
//! selector := compound ( combinator compound )*
//! combinator := ' ' | '>'
//! compound := ( ident | '*' )? ( '.' ident )*
//! ```

use rustfox_dom::{ClassId, Dom, TagId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Combinator {
    Child,
    Descendant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Simple {
    Tag(TagId),
    Class(ClassId),
    Universal,
}

/// One compound selector (`div.c1.c2`); matches when *all* simples match.
#[derive(Debug, Clone)]
pub struct Compound {
    pub simples: Vec<Simple>,
}

impl Compound {
    #[inline]
    pub fn tag(&self) -> Option<TagId> {
        self.simples.iter().find_map(|s| match s {
            Simple::Tag(t) => Some(*t),
            _ => None,
        })
    }

    #[inline]
    pub fn classes(&self) -> impl Iterator<Item = ClassId> + '_ {
        self.simples.iter().filter_map(|s| match s {
            Simple::Class(c) => Some(*c),
            _ => None,
        })
    }

    #[inline]
    pub fn is_universal_only(&self) -> bool {
        self.simples.len() == 1 && matches!(self.simples[0], Simple::Universal)
    }
}

/// Full complex selector; `compounds.last()` is the rightmost (subject) compound.
#[derive(Debug, Clone)]
pub struct Selector {
    pub compounds: Vec<Compound>,
    /// combinators[i] joins compounds[i] and compounds[i+1]
    pub combinators: Vec<Combinator>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(pub String);

struct Parser<'a> {
    chars: Vec<char>,
    pos: usize,
    dom: &'a mut Dom,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str, dom: &'a mut Dom) -> Self {
        Self {
            chars: src.chars().collect(),
            pos: 0,
            dom,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ') | Some('\t') | Some('\n')) {
            self.pos += 1;
        }
    }

    fn parse_ident(&mut self) -> Option<String> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            None
        } else {
            Some(self.chars[start..self.pos].iter().collect())
        }
    }

    fn parse_compound(&mut self) -> Result<Compound, ParseError> {
        let mut simples: Vec<Simple> = Vec::new();

        match self.peek() {
            Some('*') => {
                self.pos += 1;
                simples.push(Simple::Universal);
            }
            Some(c) if c.is_ascii_alphanumeric() || c == '_' => {
                let ident = self
                    .parse_ident()
                    .ok_or_else(|| ParseError("bad ident".into()))?;
                let tag = self.dom.intern_tag(&ident);
                simples.push(Simple::Tag(tag));
            }
            _ => {}
        }

        while self.peek() == Some('.') {
            self.pos += 1;
            let ident = self
                .parse_ident()
                .ok_or_else(|| ParseError("expected class name after '.'".into()))?;
            let class = self.dom.intern_class(&ident);
            simples.push(Simple::Class(class));
        }

        if simples.is_empty() {
            return Err(ParseError("empty compound".into()));
        }
        Ok(Compound { simples })
    }

    fn parse_selector(&mut self) -> Result<Selector, ParseError> {
        self.skip_ws();
        let mut compounds = vec![self.parse_compound()?];
        let mut combinators = Vec::new();

        loop {
            self.skip_ws();
            match self.peek() {
                None => break,
                Some('>') => {
                    self.pos += 1;
                    combinators.push(Combinator::Child);
                }
                _ => combinators.push(Combinator::Descendant),
            }
            self.skip_ws();
            let compound = self.parse_compound()?;
            compounds.push(compound);
        }

        Ok(Selector {
            compounds,
            combinators,
        })
    }
}

/// Parse one complex selector, interning names into `dom`'s tables so that
/// matching works on shared `TagId`/`ClassId` values.
pub fn parse_selector(src: &str, dom: &mut Dom) -> Result<Selector, ParseError> {
    let mut p = Parser::new(src, dom);
    p.parse_selector()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_compounds_and_combinators() {
        let mut dom = Dom::new();
        let sel = parse_selector("div.a.b > span .c", &mut dom).unwrap();
        assert_eq!(sel.compounds.len(), 3);
        assert_eq!(
            sel.combinators,
            vec![Combinator::Child, Combinator::Descendant]
        );
        assert_eq!(dom.tag_name(sel.compounds[0].tag().unwrap()), "div");
        assert_eq!(sel.compounds[0].classes().count(), 2);
        assert_eq!(sel.compounds[1].tag().unwrap(), dom.intern_tag("span"));
        assert!(sel.compounds[2].tag().is_none());
    }

    #[test]
    fn rejects_garbage() {
        let mut dom = Dom::new();
        assert!(parse_selector("", &mut dom).is_err());
        assert!(parse_selector(".", &mut dom).is_err());
        assert!(parse_selector("div >", &mut dom).is_err());
    }
}
