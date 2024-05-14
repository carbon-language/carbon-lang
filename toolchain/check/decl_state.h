// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DECL_STATE_H_
#define CARBON_TOOLCHAIN_CHECK_DECL_STATE_H_

#include "llvm/ADT/BitmaskEnum.h"
#include "toolchain/parse/node_ids.h"

namespace Carbon::Check {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// Represents a set of keyword modifiers, using a separate bit per modifier.
//
// We expect this to grow, so are using a bigger size than needed.
// NOLINTNEXTLINE(performance-enum-size)
enum class KeywordModifierSet : uint32_t {
  // At most one of these access modifiers allowed for a given declaration,
  // and if present it must be first:
  Private = 1 << 0,
  Protected = 1 << 1,

  // Extern is standalone.
  Extern = 1 << 2,

  // At most one of these declaration modifiers allowed for a given
  // declaration:
  Abstract = 1 << 3,
  Base = 1 << 4,
  Default = 1 << 5,
  Extend = 1 << 6,
  Final = 1 << 7,
  Impl = 1 << 8,
  Virtual = 1 << 9,

  // Sets of modifiers:
  Access = Private | Protected,
  Class = Abstract | Base,
  Method = Abstract | Impl | Virtual,
  ImplDecl = Extend | Final,
  Interface = Default | Final,
  Decl = Class | Method | ImplDecl | Interface,
  None = 0,

  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/Virtual)
};

inline constexpr auto operator!(KeywordModifierSet k) -> bool {
  return !static_cast<uint32_t>(k);
}

// The order of modifiers. Each of these corresponds to a group on
// KeywordModifierSet, and can be used as an array index.
enum class ModifierOrder : int8_t { Access, Extern, Decl, Last = Decl };

static_assert(!(KeywordModifierSet::Access & KeywordModifierSet::Extern) &&
                  !((KeywordModifierSet::Access | KeywordModifierSet::Extern) &
                    KeywordModifierSet::Decl),
              "Order-related sets must not overlap");

// State stored for each declaration we are currently in: the kind of
// declaration and the keyword modifiers that apply to that declaration.
struct DeclState {
  // The kind of declaration.
  enum DeclKind : int8_t {
    FileScope,
    Adapt,
    Alias,
    Base,
    Class,
    Constraint,
    Fn,
    Impl,
    Interface,
    Let,
    Namespace,
    Var
  };

  explicit DeclState(DeclKind decl_kind) : kind(decl_kind) {}

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
  KeywordModifierSet modifier_set = KeywordModifierSet::None;
};

// Stack of `DeclState` values, representing all the declarations we are
// currently nested within.
// Invariant: Bottom of the stack always has a "DeclState::FileScope" entry.
class DeclStateStack {
 public:
  DeclStateStack() { stack_.emplace_back(DeclState::FileScope); }

  // Enters a declaration of kind `k`.
  auto Push(DeclState::DeclKind k) -> void { stack_.emplace_back(k); }

  // Gets the state of declaration at the top of the stack -- the innermost
  // declaration currently being processed.
  auto innermost() -> DeclState& { return stack_.back(); }

  // Exits a declaration of kind `k`.
  auto Pop(DeclState::DeclKind k) -> void {
    CARBON_CHECK(stack_.back().kind == k)
        << "Found: " << stack_.back().kind << " expected: " << k;
    stack_.pop_back();
    CARBON_CHECK(!stack_.empty());
  }

 private:
  llvm::SmallVector<DeclState> stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DECL_STATE_H_
