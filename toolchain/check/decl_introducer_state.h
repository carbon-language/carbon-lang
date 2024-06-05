// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DECL_INTRODUCER_STATE_H_
#define CARBON_TOOLCHAIN_CHECK_DECL_INTRODUCER_STATE_H_

#include "toolchain/check/keyword_modifier_set.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_ids.h"

namespace Carbon::Check {

// State stored for each declaration we are introducing: the kind of
// declaration and the keyword modifiers that apply to that declaration
// introducer.
struct DeclIntroducerState {
  auto modifier_node_id(ModifierOrder order) -> Parse::NodeId {
    return ordered_modifier_node_ids[static_cast<int8_t>(order)];
  }
  auto set_modifier_node_id(ModifierOrder order, Parse::NodeId node_id)
      -> void {
    ordered_modifier_node_ids[static_cast<int8_t>(order)] = node_id;
  }

  // The token kind of the introducer.
  Lex::TokenKind kind;

  // Nodes of modifiers on this declaration, in expected order. `Invalid` if no
  // modifier of that kind is present.
  Parse::NodeId
      ordered_modifier_node_ids[static_cast<int8_t>(ModifierOrder::Decl) + 1] =
          {Parse::NodeId::Invalid, Parse::NodeId::Invalid,
           Parse::NodeId::Invalid};

  // Invariant: contains just the modifiers represented by `saw_*_modifier`.
  KeywordModifierSet modifier_set = KeywordModifierSet();
};

// Stack of `DeclIntroducerState` values, representing all the declaration
// introducers we are currently nested within. Commonly size 0 or 1, as nested
// introducers are rare.
class DeclIntroducerStateStack {
 public:
  // Begins a declaration introducer `Kind`.
  template <Lex::TokenKind::RawEnumType Kind>
  auto Push() -> void {
    static_assert(IsDeclIntroducer(Kind));
    stack_.push_back({.kind = Lex::TokenKind::Make(Kind)});
  }

  // Gets the state of declaration at the top of the stack -- the innermost
  // declaration currently being processed.
  auto innermost() -> DeclIntroducerState& { return stack_.back(); }

  // Finishes a declaration introducer `Kind` and returns the produced state.
  template <Lex::TokenKind::RawEnumType Kind>
  auto Pop() -> DeclIntroducerState {
    static_assert(IsDeclIntroducer(Kind));
    CARBON_CHECK(stack_.back().kind == Kind)
        << "Found: " << stack_.back().kind
        << " expected: " << Lex::TokenKind::Make(Kind);
    return stack_.pop_back_val();
  }

 private:
  // Returns true if the token is a declaration introducer. Supports restricting
  // Push/Pop to only work with introducers.
  static constexpr auto IsDeclIntroducer(Lex::TokenKind::RawEnumType kind)
      -> bool {
    switch (kind) {
#define CARBON_TOKEN(...)
#define CARBON_DECL_INTRODUCER_TOKEN(Name, ...) \
  case Lex::TokenKind::Name:                    \
    return true;
#include "toolchain/lex/token_kind.def"
      default:
        return false;
    }
  }

  llvm::SmallVector<DeclIntroducerState> stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DECL_INTRODUCER_STATE_H_
