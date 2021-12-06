// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_PARSER_PARSE_TREE_H_
#define TOOLCHAIN_PARSER_PARSE_TREE_H_

#include <iterator>

#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"

namespace Carbon {

// A tree of parsed tokens based on the language grammar.
//
// This is a purely syntactic parse tree without any semantics yet attached. It
// is based on the token stream and the grammar of the language without even
// name lookup.
//
// The tree is designed to make depth-first traversal especially efficient, with
// postorder and reverse postorder (RPO, a topological order) not even requiring
// extra state.
//
// The nodes of the tree follow a flyweight pattern and are handles into the
// tree. The tree itself must be available to query for information about those
// nodes.
//
// Nodes also have a precise one-to-one correspondence to tokens from the parsed
// token stream. Each node can be thought of as the tree-position of a
// particular token from the stream.
//
// The tree is immutable once built, but is designed to support reasonably
// efficient patterns that build a new tree with a specific transformation
// applied.
class ParseTree {
 public:
  class Node;
  class PostorderIterator;
  class SiblingIterator;

  // Parses the token buffer into a `ParseTree`.
  //
  // This is the factory function which is used to build parse trees.
  static auto Parse(TokenizedBuffer& tokens, DiagnosticConsumer& consumer)
      -> ParseTree;

  // Tests whether there are any errors in the parse tree.
  [[nodiscard]] auto HasErrors() const -> bool { return has_errors_; }

  // Returns the number of nodes in this parse tree.
  [[nodiscard]] auto Size() const -> int { return node_impls_.size(); }

  // Returns an iterable range over the parse tree nodes in depth-first
  // postorder.
  [[nodiscard]] auto Postorder() const
      -> llvm::iterator_range<PostorderIterator>;

  // Returns an iterable range over the parse tree node and all of its
  // descendants in depth-first postorder.
  [[nodiscard]] auto Postorder(Node n) const
      -> llvm::iterator_range<PostorderIterator>;

  // Returns an iterable range over the direct children of a node in the parse
  // tree. This is a forward range, but is constant time to increment. The order
  // of children is the same as would be found in a reverse postorder traversal.
  [[nodiscard]] auto Children(Node n) const
      -> llvm::iterator_range<SiblingIterator>;

  // Returns an iterable range over the roots of the parse tree. This is a
  // forward range, but is constant time to increment. The order of roots is the
  // same as would be found in a reverse postorder traversal.
  [[nodiscard]] auto Roots() const -> llvm::iterator_range<SiblingIterator>;

  // Tests whether a particular node contains an error and may not match the
  // full expected structure of the grammar.
  [[nodiscard]] auto HasErrorInNode(Node n) const -> bool;

  // Returns the kind of the given parse tree node.
  [[nodiscard]] auto GetNodeKind(Node n) const -> ParseNodeKind;

  // Returns the token the given parse tree node models.
  [[nodiscard]] auto GetNodeToken(Node n) const -> TokenizedBuffer::Token;

  // Returns the text backing the token for the given node.
  //
  // This is a convenience method for chaining from a node through its token to
  // the underlying source text.
  [[nodiscard]] auto GetNodeText(Node n) const -> llvm::StringRef;

  // Prints a description of the parse tree to the provided `raw_ostream`.
  //
  // While the parse tree is represented as a postorder sequence, we print it in
  // preorder to make it easier to visualize and read. The node indices are the
  // postorder indices. The print out represents each node as a YAML record,
  // with children nested within it.
  //
  // A single node without children is formatted as:
  // ```
  // {node_index: 0, kind: 'foo', text: '...'}
  // ```
  // A node with two children, one of them with an error:
  // ```
  // {node_index: 2, kind: 'foo', text: '...', children: [
  //   {node_index: 0, kind: 'bar', text: '...', has_error: yes},
  //   {node_index: 1, kind: 'baz', text: '...'}]}
  // ```
  // The top level is formatted as an array of these nodes.
  // ```
  // [
  // {node_index: 1, kind: 'foo', text: '...'},
  // {node_index: 0, kind: 'foo', text: '...'},
  // ...
  // ]
  // ```
  //
  // This can be parsed as YAML using tools like `python-yq` combined with `jq`
  // on the command line. The format is also reasonably amenable to other
  // line-oriented shell tools from `grep` to `awk`.
  auto Print(llvm::raw_ostream& output) const -> void;

  // Verifies the parse tree structure.
  //
  // This tries to check any invariants of the parse tree structure and write
  // out information about it to stderr. Returns false if anything fails to
  // verify. This is primarily intended to be used as a debugging aid. A typical
  // usage is to `assert` on the result. This routine doesn't directly assert so
  // that it can be used even when asserts are disabled or within a debugger.
  [[nodiscard]] auto Verify() const -> bool;

 private:
  class Parser;
  friend Parser;

  // The in-memory representation of data used for a particular node in the
  // tree.
  struct NodeImpl {
    explicit NodeImpl(ParseNodeKind k, TokenizedBuffer::Token t,
                      int subtree_size_arg)
        : kind(k), token(t), subtree_size(subtree_size_arg) {}

    // The kind of this node. Note that this is only a single byte.
    ParseNodeKind kind;

    // We have 3 bytes of padding here that we can pack flags or other compact
    // data into.

    // Whether this node is or contains a parse error.
    //
    // When this is true, this node and its children may not have the expected
    // grammatical production structure. Prior to reasoning about any specific
    // subtree structure, this flag must be checked.
    //
    // Not every node in the path from the root to an error will have this field
    // set to true. However, any node structure that fails to conform to the
    // expected grammatical production will be contained within a subtree with
    // this flag set. Whether parents of that subtree also have it set is
    // optional (and will depend on the particular parse implementation
    // strategy). The goal is that you can rely on grammar-based structural
    // invariants *until* you encounter a node with this set.
    bool has_error = false;

    // The token root of this node.
    TokenizedBuffer::Token token;

    // The size of this node's subtree of the parse tree. This is the number of
    // nodes (and thus tokens) that are covered by this node (and its
    // descendents) in the parse tree.
    //
    // During a *reverse* postorder (RPO) traversal of the parse tree, this can
    // also be thought of as the offset to the next non-descendant node. When
    // this node is not the first child of its parent (which is the last child
    // visited in RPO), that is the offset to the next sibling. When this node
    // *is* the first child of its parent, this will be an offset to the node's
    // parent's next sibling, or if it the parent is also a first child, the
    // grandparent's next sibling, and so on.
    //
    // This field should always be a positive integer as at least this node is
    // part of its subtree.
    int32_t subtree_size;
  };

  static_assert(sizeof(NodeImpl) == 12,
                "Unexpected size of node implementation!");

  // Wires up the reference to the tokenized buffer. The global `parse` routine
  // should be used to actually parse the tokens into a tree.
  explicit ParseTree(TokenizedBuffer& tokens_arg) : tokens_(&tokens_arg) {}

  // Depth-first postorder sequence of node implementation data.
  llvm::SmallVector<NodeImpl, 0> node_impls_;

  TokenizedBuffer* tokens_;

  // Indicates if any errors were encountered while parsing.
  //
  // This doesn't indicate how much of the tree is structurally accurate with
  // respect to the grammar. That can be identified by looking at the `HasError`
  // flag for a given node (see above for details). This simply indicates that
  // some errors were encountered somewhere. A key implication is that when this
  // is true we do *not* have the expected 1:1 mapping between tokens and parsed
  // nodes as some tokens may have been skipped.
  bool has_errors_ = false;
};

// A lightweight handle representing a node in the tree.
//
// Objects of this type are small and cheap to copy and store. They don't
// contain any of the information about the node, and serve as a handle that
// can be used with the underlying tree to query for detailed information.
//
// That said, nodes can be compared and are part of a depth-first pre-order
// sequence across all nodes in the parse tree.
class ParseTree::Node {
 public:
  // Node handles are default constructable, but such a node cannot be used
  // for anything. It just allows it to be initialized later through
  // assignment. Any other operation on a default constructed node is an
  // error.
  Node() = default;

  friend auto operator==(Node lhs, Node rhs) -> bool {
    return lhs.index_ == rhs.index_;
  }
  friend auto operator!=(Node lhs, Node rhs) -> bool {
    return lhs.index_ != rhs.index_;
  }
  friend auto operator<(Node lhs, Node rhs) -> bool {
    return lhs.index_ < rhs.index_;
  }
  friend auto operator<=(Node lhs, Node rhs) -> bool {
    return lhs.index_ <= rhs.index_;
  }
  friend auto operator>(Node lhs, Node rhs) -> bool {
    return lhs.index_ > rhs.index_;
  }
  friend auto operator>=(Node lhs, Node rhs) -> bool {
    return lhs.index_ >= rhs.index_;
  }

  // Returns an opaque integer identifier of the node in the tree. Clients
  // should not expect any particular semantics from this value.
  //
  // FIXME: Maybe we can switch to stream operator overloads?
  [[nodiscard]] auto GetIndex() const -> int { return index_; }

  // Prints the node index.
  auto Print(llvm::raw_ostream& output) const -> void;

 private:
  friend ParseTree;
  friend Parser;
  friend PostorderIterator;
  friend SiblingIterator;

  // Constructs a node with a specific index into the parse tree's postorder
  // sequence of node implementations.
  explicit Node(int index_arg) : index_(index_arg) {}

  // The index of this node's implementation in the postorder sequence.
  int32_t index_;
};

// A random-access iterator to the depth-first postorder sequence of parse nodes
// in the parse tree. It produces `ParseTree::Node` objects which are opaque
// handles and must be used in conjunction with the `ParseTree` itself.
class ParseTree::PostorderIterator
    : public llvm::iterator_facade_base<PostorderIterator,
                                        std::random_access_iterator_tag, Node,
                                        int, Node*, Node> {
 public:
  // Default construction is only provided to satisfy iterator requirements. It
  // produces an unusable iterator, and you must assign a valid iterator to it
  // before performing any operations.
  PostorderIterator() = default;

  auto operator==(const PostorderIterator& rhs) const -> bool {
    return node_ == rhs.node_;
  }
  auto operator<(const PostorderIterator& rhs) const -> bool {
    return node_ < rhs.node_;
  }

  auto operator*() const -> Node { return node_; }

  auto operator-(const PostorderIterator& rhs) const -> int {
    return node_.index_ - rhs.node_.index_;
  }

  auto operator+=(int offset) -> PostorderIterator& {
    node_.index_ += offset;
    return *this;
  }
  auto operator-=(int offset) -> PostorderIterator& {
    node_.index_ -= offset;
    return *this;
  }

  // Prints the underlying node index.
  auto Print(llvm::raw_ostream& output) const -> void;

 private:
  friend class ParseTree;

  explicit PostorderIterator(Node n) : node_(n) {}

  Node node_;
};

// A forward iterator across the silbings at a particular level in the parse
// tree. It produces `ParseTree::Node` objects which are opaque handles and must
// be used in conjunction with the `ParseTree` itself.
//
// While this is a forward iterator and may not have good locality within the
// `ParseTree` data structure, it is still constant time to increment and
// suitable for algorithms relying on that property.
//
// The siblings are discovered through a reverse postorder (RPO) tree traversal
// (which is made constant time through cached distance information), and so the
// relative order of siblings matches their RPO order.
class ParseTree::SiblingIterator
    : public llvm::iterator_facade_base<
          SiblingIterator, std::forward_iterator_tag, Node, int, Node*, Node> {
 public:
  SiblingIterator() = default;

  auto operator==(const SiblingIterator& rhs) const -> bool {
    return node_ == rhs.node_;
  }
  auto operator<(const SiblingIterator& rhs) const -> bool {
    // Note that child iterators walk in reverse compared to the postorder
    // index.
    return node_ > rhs.node_;
  }

  auto operator*() const -> Node { return node_; }

  using iterator_facade_base::operator++;
  auto operator++() -> SiblingIterator& {
    node_.index_ -= std::abs(tree_->node_impls_[node_.index_].subtree_size);
    return *this;
  }

  // Prints the underlying node index.
  auto Print(llvm::raw_ostream& output) const -> void;

 private:
  friend class ParseTree;

  explicit SiblingIterator(const ParseTree& tree_arg, Node n)
      : tree_(&tree_arg), node_(n) {}

  const ParseTree* tree_;

  Node node_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_PARSER_PARSE_TREE_H_
