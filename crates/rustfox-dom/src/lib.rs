//! rustfox-dom: arena-based DOM with string interning.
//!
//! Tag and class names are interned into `u32` ids so that hot-path
//! comparisons (style matching) are integer equality checks instead of
//! string comparisons — the same trick Stylo uses with its atom table.

use rustc_hash::FxHashMap;

#[derive(Debug, Default)]
pub struct Interner {
    map: FxHashMap<String, u32>,
    names: Vec<String>,
}

impl Interner {
    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.map.get(name) {
            return id;
        }
        let id = self.names.len() as u32;
        self.names.push(name.to_string());
        self.map.insert(name.to_string(), id);
        id
    }

    pub fn name(&self, id: u32) -> &str {
        &self.names[id as usize]
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TagId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClassId(pub u32);

#[derive(Debug, Clone)]
pub struct Node {
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub tag: TagId,
    pub classes: Vec<ClassId>,
}

#[derive(Debug, Default)]
pub struct Dom {
    pub nodes: Vec<Node>,
    tags: Interner,
    classes: Interner,
}

impl Dom {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern_tag(&mut self, name: &str) -> TagId {
        TagId(self.tags.intern(name))
    }

    pub fn intern_class(&mut self, name: &str) -> ClassId {
        ClassId(self.classes.intern(name))
    }

    pub fn tag_name(&self, id: TagId) -> &str {
        self.tags.name(id.0)
    }

    pub fn class_name(&self, id: ClassId) -> &str {
        self.classes.name(id.0)
    }

    pub fn create_element(&mut self, tag: &str, classes: &[&str]) -> NodeId {
        let tag_id = self.intern_tag(tag);
        let class_ids: Vec<ClassId> = classes.iter().map(|c| self.intern_class(c)).collect();
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            parent: None,
            children: Vec::new(),
            tag: tag_id,
            classes: class_ids,
        });
        id
    }

    pub fn append(&mut self, parent: NodeId, child: NodeId) {
        self.nodes[parent.0 as usize].children.push(child);
        self.nodes[child.0 as usize].parent = Some(parent);
    }

    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.nodes[id.0 as usize].parent
    }

    pub fn children(&self, id: NodeId) -> &[NodeId] {
        &self.nodes[id.0 as usize].children
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Roots of the forest: nodes without a parent.
    pub fn roots(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parent.is_none())
            .map(|(i, _)| NodeId(i as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_is_stable() {
        let mut dom = Dom::new();
        let a = dom.intern_tag("div");
        let b = dom.intern_tag("div");
        let c = dom.intern_tag("span");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(dom.tag_name(a), "div");
    }

    #[test]
    fn tree_structure() {
        let mut dom = Dom::new();
        let root = dom.create_element("div", &["container"]);
        let child = dom.create_element("span", &[]);
        dom.append(root, child);
        assert_eq!(dom.parent(child), Some(root));
        assert_eq!(dom.children(root), &[child]);
        assert_eq!(dom.roots().count(), 1);
    }
}
