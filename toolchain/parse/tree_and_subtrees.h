// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_TREE_AND_SUBTREES_H_
#define CARBON_TOOLCHAIN_PARSE_TREE_AND_SUBTREES_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Parse {

// Calculates and stores subtree data for a parse tree. Supports APIs that
// require subtree knowledge.
//
// This requires a complete tree.
class TreeAndSubtrees {
 public:
  // A range of tokens, returned by GetSubtreeTokenRange.
  struct TokenRange {
    Lex::TokenIndex begin;
    Lex::TokenIndex end;
  };

  class SiblingIterator;

  explicit TreeAndSubtrees(const Lex::TokenizedBuffer& tokens,
                           const Tree& tree);

  // The following `Extract*` function provide an alternative way of accessing
  // the nodes of a tree. It is intended to be more convenient and type-safe,
  // but slower and can't be used on nodes that are marked as having an error.
  // It is appropriate for uses that are less performance sensitive, like
  // diagnostics. Example usage:
  // ```
  // auto file = tree->ExtractFile();
  // for (AnyDeclId decl_id : file.decls) {
  //   // `decl_id` is convertible to a `NodeId`.
  //   if (std::optional<FunctionDecl> fn_decl =
  //       tree->ExtractAs<FunctionDecl>(decl_id)) {
  //     // fn_decl->params is a `TuplePatternId` (which extends `NodeId`)
  //     // that is guaranteed to reference a `TuplePattern`.
  //     std::optional<TuplePattern> params = tree->Extract(fn_decl->params);
  //     // `params` has a value unless there was an error in that node.
  //   } else if (auto class_def = tree->ExtractAs<ClassDefinition>(decl_id)) {
  //     // ...
  //   }
  // }
  // ```

  // Extract a `File` object representing the parse tree for the whole file.
  // #include "toolchain/parse/typed_nodes.h" to get the definition of `File`
  // and the types representing its children nodes. This is implemented in
  // extract.cpp.
  auto ExtractFile() const -> File;

  // Converts this node_id to a typed node of a specified type, if it is a valid
  // node of that kind.
  template <typename T>
  auto ExtractAs(NodeId node_id) const -> std::optional<T>;

  // Converts to a typed node, if it is not an error.
  template <typename IdT>
  auto Extract(IdT id) const
      -> std::optional<typename NodeForId<IdT>::TypedNode>;

  // Verifies that each node in the tree can be successfully extracted.
  //
  // This is fairly slow, and is primarily intended to be used as a debugging
  // aid. This doesn't directly CHECK so that it can be used within a debugger.
  auto Verify() const -> ErrorOr<Success>;

  // Prints the parse tree in postorder format. See also use PrintPreorder.
  //
  // Output represents each node as a YAML record. A node is formatted as:
  //   ```
  //   {kind: 'foo', text: '...'}
  //   ```
  //
  // The top level is formatted as an array of these nodes.
  //   ```
  //   [
  //   {kind: 'foo', text: '...'},
  //   {kind: 'foo', text: '...'},
  //   ...
  //   ]
  //   ```
  //
  // Nodes are indented in order to indicate depth. For example, a node with two
  // children, one of them with an error:
  //   ```
  //     {kind: 'bar', text: '...', has_error: yes},
  //     {kind: 'baz', text: '...'}
  //   {kind: 'foo', text: '...', subtree_size: 2}
  //   ```
  //
  // This can be parsed as YAML using tools like `python-yq` combined with `jq`
  // on the command line. The format is also reasonably amenable to other
  // line-oriented shell tools from `grep` to `awk`.
  auto Print(llvm::raw_ostream& output) const -> void;

  // Prints the parse tree in preorder. The format is YAML, and similar to
  // Print. However, nodes are marked as children with postorder (storage)
  // index. For example, a node with two children, one of them with an error:
  //   ```
  //   {node_index: 2, kind: 'foo', text: '...', subtree_size: 2, children: [
  //     {node_index: 0, kind: 'bar', text: '...', has_error: yes},
  //     {node_index: 1, kind: 'baz', text: '...'}]}
  //   ```
  auto PrintPreorder(llvm::raw_ostream& output) const -> void;

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void;

  // Returns the range of tokens in the node's subtree.
  auto GetSubtreeTokenRange(NodeId node_id) const -> TokenRange;

  // Converts the node to a diagnostic location, covering either the full
  // subtree or only the token.
  auto NodeToDiagnosticLoc(NodeId node_id, bool token_only) const
      -> Diagnostics::ConvertedLoc;

  // Returns an iterable range over the parse tree node and all of its
  // descendants in depth-first postorder.
  auto postorder(NodeId n) const
      -> llvm::iterator_range<Tree::PostorderIterator>;

  // Returns an iterable range over the direct children of a node in the parse
  // tree. This is a forward range, but is constant time to increment. The order
  // of children is the same as would be found in a reverse postorder traversal.
  auto children(NodeId n) const -> llvm::iterator_range<SiblingIterator>;

  // Returns an iterable range over the roots of the parse tree. This is a
  // forward range, but is constant time to increment. The order of roots is the
  // same as would be found in a reverse postorder traversal.
  auto roots() const -> llvm::iterator_range<SiblingIterator>;

  auto tree() const -> const Tree& { return *tree_; }

 private:
  friend class TypedNodesTestPeer;

  // Extract a node of type `T` from a sibling range. This is expected to
  // consume the complete sibling range. Malformed tree errors are written
  // to `*trace`, if `trace != nullptr`. This is implemented in extract.cpp.
  template <typename T>
  auto TryExtractNodeFromChildren(
      NodeId node_id, llvm::iterator_range<SiblingIterator> children,
      ErrorBuilder* trace) const -> std::optional<T>;

  // Extract a node of type `T` from a sibling range. This is expected to
  // consume the complete sibling range. Malformed tree errors are fatal.
  template <typename T>
  auto ExtractNodeFromChildren(
      NodeId node_id, llvm::iterator_range<SiblingIterator> children) const
      -> T;

  // Like ExtractAs(), but malformed tree errors are not fatal. Should only be
  // used by `Verify()` or by tests.
  template <typename T>
  auto VerifyExtractAs(NodeId node_id, ErrorBuilder* trace) const
      -> std::optional<T>;

  // Wrapper around `VerifyExtractAs` to dispatch based on a runtime node kind.
  // Returns true if extraction was successful.
  auto VerifyExtract(NodeId node_id, NodeKind kind, ErrorBuilder* trace) const
      -> bool;

  // Prints a single node for Print(). Returns true when preorder and there are
  // children.
  auto PrintNode(llvm::raw_ostream& output, NodeId n, int depth,
                 bool preorder) const -> bool;

  // The associated tokens.
  const Lex::TokenizedBuffer* tokens_;

  // The associated tree.
  const Tree* tree_;

  // For each node in the tree, the size of the node's subtree. This is the
  // number of nodes (and thus tokens) that are covered by the node (and its
  // descendents) in the parse tree. It's one for nodes with no children.
  //
  // During a *reverse* postorder (RPO) traversal of the parse tree, this can
  // also be thought of as the offset to the next non-descendant node. When the
  // node is not the first child of its parent (which is the last child visited
  // in RPO), that is the offset to the next sibling. When the node *is* the
  // first child of its parent, this will be an offset to the node's parent's
  // next sibling, or if it the parent is also a first child, the grandparent's
  // next sibling, and so on.
  llvm::SmallVector<int32_t> subtree_sizes_;
};

// A standard signature for a callback to support lazy construction.
using GetTreeAndSubtreesFn = llvm::function_ref<auto()->const TreeAndSubtrees&>;

// A forward iterator across the siblings at a particular level in the parse
// tree. It produces `Tree::NodeId` objects which are opaque handles and must
// be used in conjunction with the `Tree` itself.
//
// While this is a forward iterator and may not have good locality within the
// `Tree` data structure, it is still constant time to increment and
// suitable for algorithms relying on that property.
//
// The siblings are discovered through a reverse postorder (RPO) tree traversal
// (which is made constant time through cached distance information), and so the
// relative order of siblings matches their RPO order.
class TreeAndSubtrees::SiblingIterator
    : public llvm::iterator_facade_base<SiblingIterator,
                                        std::forward_iterator_tag, NodeId, int,
                                        const NodeId*, NodeId>,
      public Printable<SiblingIterator> {
 public:
  explicit SiblingIterator() = delete;

  friend auto operator==(const SiblingIterator& lhs, const SiblingIterator& rhs)
      -> bool {
    return lhs.node_ == rhs.node_;
  }

  auto operator*() const -> NodeId { return node_; }

  using iterator_facade_base::operator++;
  auto operator++() -> SiblingIterator& {
    node_.index -= std::abs(tree_->subtree_sizes_[node_.index]);
    return *this;
  }

  // Prints the underlying node index.
  auto Print(llvm::raw_ostream& output) const -> void;

 private:
  friend class TreeAndSubtrees;

  explicit SiblingIterator(const TreeAndSubtrees& tree, NodeId node)
      : tree_(&tree), node_(node) {}

  const TreeAndSubtrees* tree_;
  NodeId node_;
};

template <typename T>
auto TreeAndSubtrees::ExtractNodeFromChildren(
    NodeId node_id, llvm::iterator_range<SiblingIterator> children) const -> T {
  auto result = TryExtractNodeFromChildren<T>(node_id, children, nullptr);
  if (!result.has_value()) {
    // On error try again, this time capturing a trace.
    ErrorBuilder trace;
    TryExtractNodeFromChildren<T>(node_id, children, &trace);
    CARBON_FATAL("Malformed parse node:\n{0}",
                 static_cast<Error>(trace).message());
  }
  return *result;
}

template <typename T>
auto TreeAndSubtrees::ExtractAs(NodeId node_id) const -> std::optional<T> {
  static_assert(HasKindMember<T>, "Not a parse node type");
  if (!tree_->IsValid<T>(node_id)) {
    return std::nullopt;
  }

  return ExtractNodeFromChildren<T>(node_id, children(node_id));
}

template <typename T>
auto TreeAndSubtrees::VerifyExtractAs(NodeId node_id, ErrorBuilder* trace) const
    -> std::optional<T> {
  static_assert(HasKindMember<T>, "Not a parse node type");
  if (!tree_->IsValid<T>(node_id)) {
    if (trace) {
      *trace << "VerifyExtractAs error: wrong kind "
             << tree_->node_kind(node_id) << ", expected " << T::Kind << "\n";
    }
    return std::nullopt;
  }

  return TryExtractNodeFromChildren<T>(node_id, children(node_id), trace);
}

template <typename IdT>
auto TreeAndSubtrees::Extract(IdT id) const
    -> std::optional<typename NodeForId<IdT>::TypedNode> {
  if (!tree_->IsValid(id)) {
    return std::nullopt;
  }

  using T = typename NodeForId<IdT>::TypedNode;
  return ExtractNodeFromChildren<T>(id, children(id));
}

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_TREE_AND_SUBTREES_H_
