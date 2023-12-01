// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DECL_STATE_H_
#define CARBON_TOOLCHAIN_CHECK_DECL_STATE_H_

#include "llvm/ADT/BitmaskEnum.h"

namespace Carbon::Check {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// Represents a set of keyword modifiers, using a separate bit per modifier.
enum class KeywordModifierSet {
  // At most one of these access modifiers allowed for a given declaration,
  // and if present it must be first:
  Private = 1 << 0,
  Protected = 1 << 1,

  // At most one of these declaration modifiers allowed for a given
  // declaration:
  Abstract = 1 << 2,
  Base = 1 << 3,
  Default = 1 << 4,
  Final = 1 << 5,
  Impl = 1 << 6,
  Virtual = 1 << 7,

  // Sets of modifiers:
  Access = Private | Protected,
  Class = Abstract | Base,
  Method = Abstract | Impl | Virtual,
  Interface = Default | Final,
  None = 0,

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ Virtual)
};

inline auto operator!(KeywordModifierSet k) -> bool {
  return !static_cast<unsigned>(k);
}

// State stored for each declaration we are currently in: the kind of
// declaration and the keyword modifiers that apply to that declaration.
struct DeclState {
  // What kind of declaration
  enum DeclKind { FileScope, Class, Constraint, Fn, Interface, Let, Var };

  explicit DeclState(DeclKind decl_kind, Parse::Node parse_node)
      : first_node(parse_node), kind(decl_kind) {}

  // Nodes of modifiers on this declaration. `Invalid` if no modifier of that
  // kind is present.
  Parse::Node saw_access_modifier = Parse::Node::Invalid;
  Parse::Node saw_decl_modifier = Parse::Node::Invalid;

  // Node corresponding to the first token of the declaration.
  Parse::Node first_node;

  // These fields are last because they are smaller.

  // Invariant: contains just the modifiers represented by `saw_access_modifier`
  // and `saw_other_modifier`.
  KeywordModifierSet modifier_set = KeywordModifierSet::None;

  DeclKind kind;
};

// Stack of `DeclState` values, representing all the declarations we are
// currently nested within.
// Invariant: Bottom of the stack always has a "DeclState::FileScope" entry.
class DeclStateStack {
 public:
  DeclStateStack() {
    s_.emplace_back(DeclState::FileScope, Parse::Node::Invalid);
  }

  // Enters a declaration of kind `k`, with `parse_node` for the introducer
  // token.
  auto Push(DeclState::DeclKind k, Parse::Node parse_node) -> void {
    s_.push_back(DeclState(k, parse_node));
  }

  // Gets the state of declaration at the top of the stack -- the innermost
  // declaration currently being processed.
  auto innermost() -> DeclState& { return s_.back(); }

  // Gets the state for the declaration containing the innermost declaration.
  // Requires that the innermost declaration is not `FileScope`.
  auto containing() const -> const DeclState& {
    CARBON_CHECK(s_.size() >= 2);
    return s_[s_.size() - 2];
  }

  // Exits a declaration of kind `k`.
  auto Pop(DeclState::DeclKind k) -> void {
    CARBON_CHECK(s_.back().kind == k);
    s_.pop_back();
    CARBON_CHECK(!s_.empty());
  }

 private:
  llvm::SmallVector<DeclState> s_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DECL_STATE_H_
