// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_TREE_H_
#define CARBON_TOOLCHAIN_PARSE_TREE_H_

#include <iterator>

#include "common/check.h"
#include "common/error.h"
#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {

struct DeferredDefinition;

// The index of a deferred function definition within the parse tree's deferred
// definition store.
struct DeferredDefinitionIndex : public IndexBase {
  using ValueType = DeferredDefinition;

  static const DeferredDefinitionIndex Invalid;

  using IndexBase::IndexBase;
};

constexpr DeferredDefinitionIndex DeferredDefinitionIndex::Invalid =
    DeferredDefinitionIndex(InvalidIndex);

// A function whose definition is deferred because it is defined inline in a
// class or similar scope.
//
// Such functions are type-checked out of order, with their bodies checked after
// the enclosing declaration is complete. Some additional information is tracked
// for these functions in the parse tree to support this reordering.
struct DeferredDefinition {
  // The node that starts the function definition.
  FunctionDefinitionStartId start_id;
  // The function definition node.
  FunctionDefinitionId definition_id = NodeId::Invalid;
  // The index of the next method that is not nested within this one.
  DeferredDefinitionIndex next_definition_index =
      DeferredDefinitionIndex::Invalid;
};

// Defined in typed_nodes.h. Include that to call `Tree::ExtractFile()`.
struct File;

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
class Tree : public Printable<Tree> {
 public:
  class PostorderIterator;
  class SiblingIterator;

  // For PackagingDirective.
  enum class ApiOrImpl : uint8_t {
    Api,
    Impl,
  };

  // Names in packaging, whether the file's packaging or an import. Links back
  // to the node for diagnostics.
  struct PackagingNames {
    ImportDirectiveId node_id;
    IdentifierId package_id = IdentifierId::Invalid;
    StringLiteralValueId library_id = StringLiteralValueId::Invalid;
  };

  // The file's packaging.
  struct PackagingDirective {
    PackagingNames names;
    ApiOrImpl api_or_impl;
  };

  // Wires up the reference to the tokenized buffer. The `Parse` function should
  // be used to actually parse the tokens into a tree.
  explicit Tree(Lex::TokenizedBuffer& tokens_arg) : tokens_(&tokens_arg) {
    // If the tree is valid, there will be one node per token, so reserve once.
    node_impls_.reserve(tokens_->expected_parse_tree_size());
  }

  // Tests whether there are any errors in the parse tree.
  auto has_errors() const -> bool { return has_errors_; }

  // Returns the number of nodes in this parse tree.
  auto size() const -> int { return node_impls_.size(); }

  // Returns an iterable range over the parse tree nodes in depth-first
  // postorder.
  auto postorder() const -> llvm::iterator_range<PostorderIterator>;

  // Returns an iterable range over the parse tree node and all of its
  // descendants in depth-first postorder.
  auto postorder(NodeId n) const -> llvm::iterator_range<PostorderIterator>;

  // Returns an iterable range over the direct children of a node in the parse
  // tree. This is a forward range, but is constant time to increment. The order
  // of children is the same as would be found in a reverse postorder traversal.
  auto children(NodeId n) const -> llvm::iterator_range<SiblingIterator>;

  // Returns an iterable range over the roots of the parse tree. This is a
  // forward range, but is constant time to increment. The order of roots is the
  // same as would be found in a reverse postorder traversal.
  auto roots() const -> llvm::iterator_range<SiblingIterator>;

  // Tests whether a particular node contains an error and may not match the
  // full expected structure of the grammar.
  auto node_has_error(NodeId n) const -> bool;

  // Returns the kind of the given parse tree node.
  auto node_kind(NodeId n) const -> NodeKind;

  // Returns the token the given parse tree node models.
  auto node_token(NodeId n) const -> Lex::TokenIndex;

  auto node_subtree_size(NodeId n) const -> int32_t;

  // Returns whether this node is a valid node of the specified type.
  template <typename T>
  auto IsValid(NodeId node_id) const -> bool {
    return node_kind(node_id) == T::Kind && !node_has_error(node_id);
  }

  template <typename IdT>
  auto IsValid(IdT id) const -> bool {
    using T = typename NodeForId<IdT>::TypedNode;
    CARBON_DCHECK(node_kind(id) == T::Kind);
    return !node_has_error(id);
  }

  // Converts `n` to a constrained node id `T` if the `node_kind(n)` matches
  // the constraint on `T`.
  template <typename T>
  auto TryAs(NodeId n) const -> std::optional<T> {
    CARBON_DCHECK(n.is_valid());
    if (ConvertTo<T>::AllowedFor(node_kind(n))) {
      return T(n);
    } else {
      return std::nullopt;
    }
  }

  // Converts to `n` to a constrained node id `T`. Checks that the
  // `node_kind(n)` matches the constraint on `T`.
  template <typename T>
  auto As(NodeId n) const -> T {
    CARBON_DCHECK(n.is_valid());
    CARBON_CHECK(ConvertTo<T>::AllowedFor(node_kind(n)));
    return T(n);
  }

  auto packaging_directive() const -> const std::optional<PackagingDirective>& {
    return packaging_directive_;
  }
  auto imports() const -> llvm::ArrayRef<PackagingNames> { return imports_; }
  auto deferred_definitions() const
      -> const ValueStore<DeferredDefinitionIndex>& {
    return deferred_definitions_;
  }

  // See the other Print comments.
  auto Print(llvm::raw_ostream& output) const -> void;

  // Prints a description of the parse tree to the provided `raw_ostream`.
  //
  // The tree may be printed in either preorder or postorder. Output represents
  // each node as a YAML record; in preorder, children are nested.
  //
  // In both, a node is formatted as:
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
  // In postorder, nodes are indented in order to indicate depth. For example, a
  // node with two children, one of them with an error:
  //   ```
  //     {kind: 'bar', text: '...', has_error: yes},
  //     {kind: 'baz', text: '...'}
  //   {kind: 'foo', text: '...', subtree_size: 2}
  //   ```
  //
  // In preorder, nodes are marked as children with postorder (storage) index.
  // For example, a node with two children, one of them with an error:
  //   ```
  //   {node_index: 2, kind: 'foo', text: '...', subtree_size: 2, children: [
  //     {node_index: 0, kind: 'bar', text: '...', has_error: yes},
  //     {node_index: 1, kind: 'baz', text: '...'}]}
  //   ```
  //
  // This can be parsed as YAML using tools like `python-yq` combined with `jq`
  // on the command line. The format is also reasonably amenable to other
  // line-oriented shell tools from `grep` to `awk`.
  auto Print(llvm::raw_ostream& output, bool preorder) const -> void;

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
  // and the types representing its children nodes.
  auto ExtractFile() const -> File;

  // Converts this node_id to a typed node of a specified type, if it is a valid
  // node of that kind.
  template <typename T>
  auto ExtractAs(NodeId node_id) const -> std::optional<T>;

  // Converts to a typed node, if it is not an error.
  template <typename IdT>
  auto Extract(IdT id) const
      -> std::optional<typename NodeForId<IdT>::TypedNode>;

  // Verifies the parse tree structure. Checks invariants of the parse tree
  // structure and returns verification errors.
  //
  // This is fairly slow, and is primarily intended to be used as a debugging
  // aid. This routine doesn't directly CHECK so that it can be used within a
  // debugger.
  auto Verify() const -> ErrorOr<Success>;

  // Like ExtractAs(), but malformed tree errors are not fatal. Should only be
  // used by `Verify()`.
  template <typename T>
  auto VerifyExtractAs(NodeId node_id, ErrorBuilder* trace) const
      -> std::optional<T>;

 private:
  friend class Context;

  template <typename T>
  struct ConvertTo;

  // The in-memory representation of data used for a particular node in the
  // tree.
  struct NodeImpl {
    explicit NodeImpl(NodeKind kind, bool has_error, Lex::TokenIndex token,
                      int subtree_size)
        : kind(kind),
          has_error(has_error),
          token(token),
          subtree_size(subtree_size) {}

    // The kind of this node. Note that this is only a single byte.
    NodeKind kind;

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
    Lex::TokenIndex token;

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

  // Prints a single node for Print(). Returns true when preorder and there are
  // children.
  auto PrintNode(llvm::raw_ostream& output, NodeId n, int depth,
                 bool preorder) const -> bool;

  // Extract a node of type `T` from a sibling range. This is expected to
  // consume the complete sibling range. Malformed tree errors are written
  // to `*trace`, if `trace != nullptr`.
  template <typename T>
  auto TryExtractNodeFromChildren(
      llvm::iterator_range<Tree::SiblingIterator> children,
      ErrorBuilder* trace) const -> std::optional<T>;

  // Extract a node of type `T` from a sibling range. This is expected to
  // consume the complete sibling range. Malformed tree errors are fatal.
  template <typename T>
  auto ExtractNodeFromChildren(
      llvm::iterator_range<Tree::SiblingIterator> children) const -> T;

  // Depth-first postorder sequence of node implementation data.
  llvm::SmallVector<NodeImpl> node_impls_;

  Lex::TokenizedBuffer* tokens_;

  // Indicates if any errors were encountered while parsing.
  //
  // This doesn't indicate how much of the tree is structurally accurate with
  // respect to the grammar. That can be identified by looking at the `HasError`
  // flag for a given node (see above for details). This simply indicates that
  // some errors were encountered somewhere. A key implication is that when this
  // is true we do *not* have the expected 1:1 mapping between tokens and parsed
  // nodes as some tokens may have been skipped.
  bool has_errors_ = false;

  std::optional<PackagingDirective> packaging_directive_;
  llvm::SmallVector<PackagingNames> imports_;
  ValueStore<DeferredDefinitionIndex> deferred_definitions_;
};

// A random-access iterator to the depth-first postorder sequence of parse nodes
// in the parse tree. It produces `Tree::NodeId` objects which are opaque
// handles and must be used in conjunction with the `Tree` itself.
class Tree::PostorderIterator
    : public llvm::iterator_facade_base<PostorderIterator,
                                        std::random_access_iterator_tag, NodeId,
                                        int, const NodeId*, NodeId>,
      public Printable<Tree::PostorderIterator> {
 public:
  PostorderIterator() = delete;

  auto operator==(const PostorderIterator& rhs) const -> bool {
    return node_ == rhs.node_;
  }
  // While we don't want users to directly leverage the index of `NodeId` for
  // ordering, when we're explicitly walking in postorder, that becomes
  // reasonable so add the ordering here and reach down for the index
  // explicitly.
  auto operator<=>(const PostorderIterator& rhs) const -> std::strong_ordering {
    return node_.index <=> rhs.node_.index;
  }

  auto operator*() const -> NodeId { return node_; }

  auto operator-(const PostorderIterator& rhs) const -> int {
    return node_.index - rhs.node_.index;
  }

  auto operator+=(int offset) -> PostorderIterator& {
    node_.index += offset;
    return *this;
  }
  auto operator-=(int offset) -> PostorderIterator& {
    node_.index -= offset;
    return *this;
  }

  // Prints the underlying node index.
  auto Print(llvm::raw_ostream& output) const -> void;

 private:
  friend class Tree;

  explicit PostorderIterator(NodeId n) : node_(n) {}

  NodeId node_;
};

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
class Tree::SiblingIterator
    : public llvm::iterator_facade_base<SiblingIterator,
                                        std::forward_iterator_tag, NodeId, int,
                                        const NodeId*, NodeId>,
      public Printable<Tree::SiblingIterator> {
 public:
  explicit SiblingIterator() = delete;

  auto operator==(const SiblingIterator& rhs) const -> bool {
    return node_ == rhs.node_;
  }

  auto operator*() const -> NodeId { return node_; }

  using iterator_facade_base::operator++;
  auto operator++() -> SiblingIterator& {
    node_.index -= std::abs(tree_->node_impls_[node_.index].subtree_size);
    return *this;
  }

  // Prints the underlying node index.
  auto Print(llvm::raw_ostream& output) const -> void;

 private:
  friend class Tree;

  explicit SiblingIterator(const Tree& tree_arg, NodeId n)
      : tree_(&tree_arg), node_(n) {}

  const Tree* tree_;

  NodeId node_;
};

template <typename T>
auto Tree::ExtractNodeFromChildren(
    llvm::iterator_range<Tree::SiblingIterator> children) const -> T {
  auto result = TryExtractNodeFromChildren<T>(children, nullptr);
  if (!result.has_value()) {
    // On error try again, this time capturing a trace.
    ErrorBuilder trace;
    TryExtractNodeFromChildren<T>(children, &trace);
    CARBON_FATAL() << "Malformed parse node:\n"
                   << static_cast<Error>(trace).message();
  }
  return *result;
}

template <typename T>
auto Tree::ExtractAs(NodeId node_id) const -> std::optional<T> {
  static_assert(HasKindMember<T>, "Not a parse node type");
  if (!IsValid<T>(node_id)) {
    return std::nullopt;
  }

  return ExtractNodeFromChildren<T>(children(node_id));
}

template <typename T>
auto Tree::VerifyExtractAs(NodeId node_id, ErrorBuilder* trace) const
    -> std::optional<T> {
  static_assert(HasKindMember<T>, "Not a parse node type");
  if (!IsValid<T>(node_id)) {
    return std::nullopt;
  }

  return TryExtractNodeFromChildren<T>(children(node_id), trace);
}

template <typename IdT>
auto Tree::Extract(IdT id) const
    -> std::optional<typename NodeForId<IdT>::TypedNode> {
  if (!IsValid(id)) {
    return std::nullopt;
  }

  using T = typename NodeForId<IdT>::TypedNode;
  return ExtractNodeFromChildren<T>(children(id));
}

template <const NodeKind& K>
struct Tree::ConvertTo<NodeIdForKind<K>> {
  static auto AllowedFor(NodeKind kind) -> bool { return kind == K; }
};

template <NodeCategory C>
struct Tree::ConvertTo<NodeIdInCategory<C>> {
  static auto AllowedFor(NodeKind kind) -> bool {
    return !!(kind.category() & C);
  }
};

template <typename... T>
struct Tree::ConvertTo<NodeIdOneOf<T...>> {
  static auto AllowedFor(NodeKind kind) -> bool {
    return ((kind == T::Kind) || ...);
  }
};

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_TREE_H_
