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
#include "toolchain/base/value_store.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {

struct DeferredDefinition;

// The index of a deferred function definition within the parse tree's deferred
// definition store.
struct DeferredDefinitionIndex : public IndexBase<DeferredDefinitionIndex> {
  static constexpr llvm::StringLiteral Label = "deferred_def";
  using ValueType = DeferredDefinition;

  using IndexBase::IndexBase;
};

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
  FunctionDefinitionId definition_id = NodeId::None;
  // The index of the next method that is not nested within this one.
  DeferredDefinitionIndex next_definition_index = DeferredDefinitionIndex::None;
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

  // Names in packaging, whether the file's packaging or an import. Links back
  // to the node for diagnostics.
  struct PackagingNames {
    ImportDeclId node_id;
    PackageNameId package_id = PackageNameId::None;
    // TODO: Move LibraryNameId to Base and use it here.
    StringLiteralValueId library_id = StringLiteralValueId::None;
    // Whether an import is exported. This is on the file's packaging
    // declaration even though it doesn't apply, for consistency in structure.
    bool is_export = false;
  };

  // The file's packaging.
  struct PackagingDecl {
    PackagingNames names;
    bool is_impl;
  };

  // Wires up the reference to the tokenized buffer. The `Parse` function should
  // be used to actually parse the tokens into a tree.
  explicit Tree(Lex::TokenizedBuffer& tokens_arg) : tokens_(&tokens_arg) {
    // If the tree is valid, there will be one node per token, so reserve once.
    node_impls_.reserve(tokens_->expected_max_parse_tree_size());
  }

  auto has_errors() const -> bool { return has_errors_; }

  auto set_has_errors(bool has_errors) -> void { has_errors_ = has_errors; }

  // Returns the number of nodes in this parse tree.
  auto size() const -> int { return node_impls_.size(); }

  // Returns an iterable range over the parse tree nodes in depth-first
  // postorder.
  auto postorder() const -> llvm::iterator_range<PostorderIterator>;

  // Tests whether a particular node contains an error and may not match the
  // full expected structure of the grammar.
  auto node_has_error(NodeId n) const -> bool {
    CARBON_DCHECK(n.has_value());
    return node_impls_[n.index].has_error();
  }

  // Returns the kind of the given parse tree node.
  auto node_kind(NodeId n) const -> NodeKind {
    CARBON_DCHECK(n.has_value());
    return node_impls_[n.index].kind();
  }

  // Returns the token the given parse tree node models.
  auto node_token(NodeId n) const -> Lex::TokenIndex;

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
    CARBON_DCHECK(n.has_value());
    if (ConvertTo<T>::AllowedFor(node_kind(n))) {
      return T::UnsafeMake(n);
    } else {
      return std::nullopt;
    }
  }

  // Converts to `n` to a constrained node id `T`. Checks that the
  // `node_kind(n)` matches the constraint on `T`.
  template <typename T>
  auto As(NodeId n) const -> T {
    CARBON_DCHECK(n.has_value());
    CARBON_DCHECK(ConvertTo<T>::AllowedFor(node_kind(n)));
    return T::UnsafeMake(n);
  }

  auto packaging_decl() const -> const std::optional<PackagingDecl>& {
    return packaging_decl_;
  }
  auto imports() const -> llvm::ArrayRef<PackagingNames> { return imports_; }
  auto deferred_definitions() const
      -> const ValueStore<DeferredDefinitionIndex>& {
    return deferred_definitions_;
  }

  // Builds TreeAndSubtrees to print the tree.
  auto Print(llvm::raw_ostream& output) const -> void;

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void;

  // Verifies the parse tree structure. Checks invariants of the parse tree
  // structure and returns verification errors.
  //
  // In opt builds, this does some minimal checking. In debug builds, it'll
  // build a TreeAndSubtrees and run further verification. This doesn't directly
  // CHECK so that it can be used within a debugger.
  auto Verify() const -> ErrorOr<Success>;

  auto tokens() const -> const Lex::TokenizedBuffer& { return *tokens_; }

 private:
  friend class Context;
  friend class TypedNodesTestPeer;

  template <typename T>
  struct ConvertTo;

  // The in-memory representation of data used for a particular node in the
  // tree.
  class NodeImpl {
   public:
    explicit NodeImpl(NodeKind kind, bool has_error, Lex::TokenIndex token)
        : kind_(kind), has_error_(has_error), token_index_(token.index) {
      CARBON_DCHECK(token.index >= 0, "Unexpected token for node: {0}", token);
    }

    auto kind() const -> NodeKind { return kind_; }
    auto set_kind(NodeKind kind) -> void { kind_ = kind; }
    auto has_error() const -> bool { return has_error_; }
    auto token() const -> Lex::TokenIndex {
      return Lex::TokenIndex(token_index_);
    }

   private:
    // The kind of this node. Note that this is only a single byte.
    NodeKind kind_;
    static_assert(sizeof(kind_) == 1, "TokenKind must pack to 8 bits");

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
    bool has_error_ : 1;

    // The token root of this node.
    unsigned token_index_ : Lex::TokenIndex::Bits;
  };

  static_assert(sizeof(NodeImpl) == 4,
                "Unexpected size of node implementation!");

  // Sets the kind of a node. This is intended to allow putting the tree into a
  // state where verification can fail, in order to make the failure path of
  // `Verify` testable.
  auto SetNodeKindForTesting(NodeId node_id, NodeKind kind) -> void {
    node_impls_[node_id.index].set_kind(kind);
  }

  // Depth-first postorder sequence of node implementation data.
  llvm::SmallVector<NodeImpl> node_impls_;

  Lex::TokenizedBuffer* tokens_;

  // True if any lowering-blocking issues were encountered while parsing. Trees
  // are expected to still be structurally valid for checking.
  //
  // This doesn't indicate how much of the tree is structurally accurate with
  // respect to the grammar. That can be identified by looking at
  // `node_has_error` (see above for details). This simply indicates that some
  // errors were encountered somewhere. A key implication is that when this is
  // true we do *not* enforce the expected 1:1 mapping between tokens and parsed
  // nodes, because some tokens may have been skipped.
  bool has_errors_ = false;

  std::optional<PackagingDecl> packaging_decl_;
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
  // Returns an iterable range between the two parse tree nodes, in depth-first
  // postorder. The range is inclusive of the bounds: [begin, end].
  static auto MakeRange(NodeId begin, NodeId end)
      -> llvm::iterator_range<PostorderIterator>;

  // Prefer using the `postorder` range calls, but direct construction is
  // allowed if needed.
  explicit PostorderIterator(NodeId n) : node_(n) {}

  PostorderIterator() = delete;

  friend auto operator==(const PostorderIterator& lhs,
                         const PostorderIterator& rhs) -> bool {
    return lhs.node_ == rhs.node_;
  }
  // While we don't want users to directly leverage the index of `NodeId` for
  // ordering, when we're explicitly walking in postorder, that becomes
  // reasonable so add the ordering here and reach down for the index
  // explicitly.
  friend auto operator<=>(const PostorderIterator& lhs,
                          const PostorderIterator& rhs)
      -> std::strong_ordering {
    return lhs.node_.index <=> rhs.node_.index;
  }

  auto operator*() const -> NodeId { return node_; }

  friend auto operator-(const PostorderIterator& lhs,
                        const PostorderIterator& rhs) -> int {
    return lhs.node_.index - rhs.node_.index;
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

  NodeId node_;
};

template <const NodeKind& K>
struct Tree::ConvertTo<NodeIdForKind<K>> {
  static auto AllowedFor(NodeKind kind) -> bool { return kind == K; }
};

template <NodeCategory::RawEnumType C>
struct Tree::ConvertTo<NodeIdInCategory<C>> {
  static auto AllowedFor(NodeKind kind) -> bool {
    return kind.category().HasAnyOf(C);
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
