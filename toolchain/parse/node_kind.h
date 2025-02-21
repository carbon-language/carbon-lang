// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_NODE_KIND_H_
#define CARBON_TOOLCHAIN_PARSE_NODE_KIND_H_

#include <cstdint>

#include "common/enum_base.h"
#include "common/ostream.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Parse {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// Represents a set of keyword modifiers, using a separate bit per modifier.
class NodeCategory : public Printable<NodeCategory> {
 public:
  // Provide values as an enum. This doesn't expose these as NodeCategory
  // instances just due to the duplication of declarations that would cause.
  //
  // We expect this to grow, so are using a bigger size than needed.
  // NOLINTNEXTLINE(performance-enum-size)
  enum RawEnumType : uint32_t {
    Decl = 1 << 0,
    Expr = 1 << 1,
    ImplAs = 1 << 2,
    MemberExpr = 1 << 3,
    MemberName = 1 << 4,
    Modifier = 1 << 5,
    Pattern = 1 << 6,
    Statement = 1 << 7,
    IntConst = 1 << 8,
    Requirement = 1 << 9,
    NonExprIdentifierName = 1 << 10,
    PackageName = 1 << 11,
    // If you add a new category here, also add it to the Print function.
    None = 0,

    LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/PackageName)
  };

  // Support implicit conversion so that the difference with the member enum is
  // opaque.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr NodeCategory(RawEnumType value) : value_(value) {}

  // Returns true if there's a non-empty set intersection.
  constexpr auto HasAnyOf(NodeCategory other) const -> bool {
    return value_ & other.value_;
  }

  // Returns the set inverse.
  constexpr auto operator~() const -> NodeCategory { return ~value_; }

  friend auto operator==(NodeCategory lhs, NodeCategory rhs) -> bool {
    return lhs.value_ == rhs.value_;
  }

  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  RawEnumType value_;
};

CARBON_DEFINE_RAW_ENUM_CLASS(NodeKind, uint8_t) {
#define CARBON_PARSE_NODE_KIND(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/parse/node_kind.def"
};

// A class wrapping an enumeration of the different kinds of nodes in the parse
// tree.
//
// In order to allow the children of a node to be determined without relying on
// the subtree size field in the parse node, each node kind must have one of:
//
// - a bracketing node kind, which is always the kind for the first child, and
//   is never the kind of any other child, or
// - a fixed child count,
//
// or both. This is required even for nodes for which `Tree::node_has_errors`
// returns `true`.
class NodeKind : public CARBON_ENUM_BASE(NodeKind) {
 public:
#define CARBON_PARSE_NODE_KIND(Name) CARBON_ENUM_CONSTANT_DECL(Name)
#include "toolchain/parse/node_kind.def"

  // Validates that a `node_kind` parser node can be generated for a
  // `lex_token_kind` lexer token.
  auto CheckMatchesTokenKind(Lex::TokenKind lex_token_kind, bool has_error)
      -> void;

  // Returns true if the node is bracketed.
  auto has_bracket() const -> bool;

  // Returns the bracketing node kind for the current node kind. Requires that
  // has_bracket is true.
  auto bracket() const -> NodeKind;

  // Returns true if the node is has a fixed child count.
  auto has_child_count() const -> bool;

  // Returns the number of children that the node must have, often 0. Requires
  // that has_child_count is true.
  auto child_count() const -> int32_t;

  // Returns which categories this node kind is in.
  auto category() const -> NodeCategory;

  // Number of different kinds, usable in a constexpr context.
  static const int ValidCount;

  using EnumBase::AsInt;
  using EnumBase::Make;

  class Definition;
  struct DefinitionArgs;

  // Provides a definition for this parse node kind. Should only be called
  // once, to construct the kind as part of defining it in `typed_nodes.h`.
  constexpr auto Define(DefinitionArgs args) const -> Definition;

 private:
  // Looks up the definition for this instruction kind.
  auto definition() const -> const Definition&;
};

#define CARBON_PARSE_NODE_KIND(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(NodeKind, Name)
#include "toolchain/parse/node_kind.def"

constexpr int NodeKind::ValidCount = 0
#define CARBON_PARSE_NODE_KIND(Name) +1
#include "toolchain/parse/node_kind.def"
    ;

static_assert(
    NodeKind::ValidCount != 0,
    "The above `constexpr` definition of `ValidCount` makes it available in "
    "a `constexpr` context despite being declared as merely `const`. We use it "
    "in a static assert here to ensure that.");

// We expect the parse node kind to fit compactly into 8 bits.
static_assert(sizeof(NodeKind) == 1, "Kind objects include padding!");

// Optional arguments that can be supplied when defining a node kind. At least
// one of `bracketed_by` and `child_count` is required.
struct NodeKind::DefinitionArgs {
  // The category for the node.
  NodeCategory category = NodeCategory::None;
  // The kind of the bracketing node, which is the first child.
  std::optional<NodeKind> bracketed_by = std::nullopt;
  // The fixed child count.
  int32_t child_count = -1;
};

// A definition of a parse node kind. This is a NodeKind value, plus
// ancillary data such as the name to use for the node kind in LLVM IR. These
// are not copyable, and only one instance of this type is expected to exist per
// parse node kind, specifically `TypedNode::Kind`. Use `NodeKind` instead as a
// thin wrapper around a parse node kind index.
class NodeKind::Definition : public NodeKind {
 public:
  // Not copyable.
  Definition(const Definition&) = delete;
  auto operator=(const Definition&) -> Definition& = delete;

  // Returns true if the node is bracketed.
  constexpr auto has_bracket() const -> bool { return bracket_ != *this; }

  // Returns the bracketing node kind for the current node kind. Requires that
  // has_bracket is true.
  constexpr auto bracket() const -> NodeKind {
    CARBON_CHECK(has_bracket(), "{0}", *this);
    return bracket_;
  }

  // Returns true if the node is has a fixed child count.
  constexpr auto has_child_count() const -> bool { return child_count_ >= 0; }

  // Returns the number of children that the node must have, often 0. Requires
  // that has_child_count is true.
  constexpr auto child_count() const -> int32_t {
    CARBON_CHECK(has_child_count(), "{0}", *this);
    return child_count_;
  }

  // Returns which categories this node kind is in.
  constexpr auto category() const -> NodeCategory { return category_; }

 private:
  friend class NodeKind;

  // This is factored out and non-constexpr to improve the compile-time error
  // message if the check below fails.
  auto MustSpecifyEitherBracketingNodeOrChildCount() {
    CARBON_FATAL("Must specify either bracketing node or fixed child count.");
  }

  constexpr explicit Definition(NodeKind kind, DefinitionArgs args)
      : NodeKind(kind),
        category_(args.category),
        bracket_(args.bracketed_by.value_or(kind)),
        child_count_(args.child_count) {
    if (!has_bracket() && !has_child_count()) {
      MustSpecifyEitherBracketingNodeOrChildCount();
    }
  }

  NodeCategory category_ = NodeCategory::None;
  // Nodes are never self-bracketed, so we use *this to indicate that the node
  // is not bracketed.
  NodeKind bracket_ = *this;
  int32_t child_count_ = -1;
};

constexpr auto NodeKind::Define(DefinitionArgs args) const -> Definition {
  return Definition(*this, args);
}

// HasKindMember<T> is true if T has a `static const NodeKind::Definition Kind`
// member.
template <typename T, typename KindType = const NodeKind::Definition*>
inline constexpr bool HasKindMember = false;
template <typename T>
inline constexpr bool HasKindMember<T, decltype(&T::Kind)> = true;

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_NODE_KIND_H_
