// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_NODE_KIND_H_
#define CARBON_TOOLCHAIN_PARSE_NODE_KIND_H_

#include <cstdint>

#include "common/enum_base.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Parse {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// Represents a set of keyword modifiers, using a separate bit per modifier.
//
// We expect this to grow, so are using a bigger size than needed.
// NOLINTNEXTLINE(performance-enum-size)
enum class NodeCategory : uint32_t {
  Decl = 1 << 0,
  Expr = 1 << 1,
  ImplAs = 1 << 2,
  MemberExpr = 1 << 3,
  MemberName = 1 << 4,
  Modifier = 1 << 5,
  NameComponent = 1 << 6,
  Pattern = 1 << 7,
  Statement = 1 << 8,
  None = 0,

  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/Statement)
};

constexpr auto operator!(NodeCategory k) -> bool {
  return !static_cast<uint32_t>(k);
}

auto operator<<(llvm::raw_ostream& output, NodeCategory category)
    -> llvm::raw_ostream&;

CARBON_DEFINE_RAW_ENUM_CLASS(NodeKind, uint8_t) {
#define CARBON_PARSE_NODE_KIND(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/parse/node_kind.def"
};

// A class wrapping an enumeration of the different kinds of nodes in the parse
// tree.
class NodeKind : public CARBON_ENUM_BASE(NodeKind) {
 public:
#define CARBON_PARSE_NODE_KIND(Name) CARBON_ENUM_CONSTANT_DECL(Name)
#include "toolchain/parse/node_kind.def"

  // Validates that a `node_kind` parser node can be generated for a
  // `lex_token_kind` lexer token.
  auto CheckMatchesTokenKind(Lex::TokenKind lex_token_kind, bool has_error)
      -> void;

  // Returns true if the node is bracketed; otherwise, child_count is used.
  auto has_bracket() const -> bool;

  // Returns the bracketing node kind for the current node kind. Requires that
  // has_bracket is true.
  auto bracket() const -> NodeKind;

  // Returns true if the node is has a fixed child count; otherwise, bracket is
  // used.
  auto has_child_count() const -> bool;

  // Returns the number of children that the node must have, often 0. Requires
  // that has_bracket is false.
  auto child_count() const -> int32_t;

  // Returns which categories this node kind is in.
  auto category() const -> NodeCategory;

  // Number of different kinds, usable in a constexpr context.
  static const int ValidCount;

  using EnumBase::AsInt;
  using EnumBase::Make;

  class Definition;

  // Provides a definition for this parse node kind. Should only be called
  // once, to construct the kind as part of defining it in `typed_nodes.h`.
  //
  // The arguments can be:
  // - A NodeCategory value, specifying the category for the node.
  // - A BracketedBy<Type> value, specifying the bracketing node.
  // - A ChildCount instance, specifying the fixed number of children.
  // At least one of BracketedBy and ChildCount is mandatory.
  template <typename... Args>
  constexpr auto Define(Args... args) const -> Definition;

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

// Wrapper type used to indicate the bracketing node in a node kind definition.
struct BracketedByKind {
  NodeKind kind;
};

// Convenience template to allow `BracketedByKind`s to be named as
// `BracketedBy<Kind>`.
template <typename Bracket>
constexpr BracketedByKind BracketedBy = {Bracket::Kind};

// Wrapper type used to indicate the constant child count in a node kind
// definition.
struct ChildCount {
  int32_t count;
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

  // Returns true if the node is bracketed; otherwise, child_count is used.
  constexpr auto has_bracket() const -> bool { return bracket_ != *this; }

  // Returns the bracketing node kind for the current node kind. Requires that
  // has_bracket is true.
  constexpr auto bracket() const -> NodeKind {
    CARBON_CHECK(has_bracket()) << *this;
    return bracket_;
  }

  // Returns true if the node is has a fixed child count; otherwise, bracket is
  // used.
  constexpr auto has_child_count() const -> bool { return child_count_ >= 0; }

  // Returns the number of children that the node must have, often 0. Requires
  // that has_bracket is false.
  constexpr auto child_count() const -> int32_t {
    CARBON_CHECK(has_child_count()) << *this;
    return child_count_;
  }

  // Returns which categories this node kind is in.
  constexpr auto category() const -> NodeCategory { return category_; }

 private:
  friend class NodeKind;

  // This is factored out and non-constexpr to improve the compile-time error
  // message if the check below fails.
  auto MustSpecifyEitherBracketingNodeOrChildCount() {
    CARBON_FATAL()
        << "Must specify either bracketing node or fixed child count.";
  }

  template <typename... Args>
  constexpr explicit Definition(NodeKind kind, Args... args) : NodeKind(kind) {
    (ApplyArg(args), ...);

    if (!has_bracket() && !has_child_count()) {
      MustSpecifyEitherBracketingNodeOrChildCount();
    }
  }

  constexpr auto ApplyArg(NodeCategory category) -> void {
    category_ = category;
  }
  constexpr auto ApplyArg(BracketedByKind bracket) -> void {
    bracket_ = bracket.kind;
  }
  constexpr auto ApplyArg(ChildCount child_count) -> void {
    child_count_ = child_count.count;
  }

  NodeCategory category_ = NodeCategory::None;
  // Nodes are never self-bracketed, so we use *this to indicate that the node
  // is not bracketed.
  NodeKind bracket_ = *this;
  int32_t child_count_ = -1;
};

template <typename... Args>
constexpr auto NodeKind::Define(Args... args) const -> Definition {
  return Definition(*this, args...);
}

// HasKindMember<T> is true if T has a `static const NodeKind::Definition Kind`
// member.
template <typename T, typename KindType = const NodeKind::Definition*>
inline constexpr bool HasKindMember = false;
template <typename T>
inline constexpr bool HasKindMember<T, decltype(&T::Kind)> = true;

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_NODE_KIND_H_
