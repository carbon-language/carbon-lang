// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DECL_INTRODUCER_STATE_H_
#define CARBON_TOOLCHAIN_CHECK_DECL_INTRODUCER_STATE_H_

#include "toolchain/check/keyword_modifier_set.h"
#include "toolchain/parse/node_ids.h"

namespace Carbon::Check {

// State stored for each declaration we are introducing: the kind of
// declaration and the keyword modifiers that apply to that declaration
// introducer.
struct DeclIntroducerState {
  // The kind of declaration.
  enum DeclKind : int8_t {
    Adapt,
    Alias,
    Base,
    Class,
    Constraint,
    Export,
    Fn,
    Impl,
    Import,
    Interface,
    Let,
    Library,
    Namespace,
    Package,
    Var
  };

  explicit DeclIntroducerState(DeclKind decl_kind) : kind(decl_kind) {}

  auto modifier_node_id(ModifierOrder order) -> Parse::NodeId {
    return ordered_modifier_node_ids[static_cast<int8_t>(order)];
  }
  auto set_modifier_node_id(ModifierOrder order, Parse::NodeId node_id)
      -> void {
    ordered_modifier_node_ids[static_cast<int8_t>(order)] = node_id;
  }

  DeclKind kind;

  // Nodes of modifiers on this declaration, in expected order. `Invalid` if no
  // modifier of that kind is present.
  Parse::NodeId
      ordered_modifier_node_ids[static_cast<int8_t>(ModifierOrder::Decl) + 1] =
          {Parse::NodeId::Invalid, Parse::NodeId::Invalid,
           Parse::NodeId::Invalid};

  // Invariant: contains just the modifiers represented by `saw_*_modifier`.
  KeywordModifierSet modifier_set;
};

// Stack of `DeclIntroducerState` values, representing all the declaration
// introducers we are currently nested within. Commonly size 0 or 1, as nested
// introducers are rare.
class DeclIntroducerStateStack {
 public:
  // Begins introducing a declaration of kind `k`.
  auto Push(DeclIntroducerState::DeclKind k) -> void { stack_.emplace_back(k); }

  // Gets the state of declaration at the top of the stack -- the innermost
  // declaration currently being processed.
  auto innermost() -> DeclIntroducerState& { return stack_.back(); }

  // Finishes introducing a declaration of kind `k` and returns the
  // produced state.
  auto Pop(DeclIntroducerState::DeclKind k) -> DeclIntroducerState {
    CARBON_CHECK(stack_.back().kind == k)
        << "Found: " << stack_.back().kind << " expected: " << k;
    return stack_.pop_back_val();
  }

 private:
  llvm::SmallVector<DeclIntroducerState> stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DECL_INTRODUCER_STATE_H_
