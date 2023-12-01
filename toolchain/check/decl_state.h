// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DECL_STATE_H_
#define CARBON_TOOLCHAIN_CHECK_DECL_STATE_H_

namespace Carbon::Check {

// Represents a set of keyword modifiers, uses a separate bit per modifier.
class KeywordModifierSet {
 public:
  enum RawEnum {
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
  };

  KeywordModifierSet() : keywords_(static_cast<RawEnum>(0)) {}
  KeywordModifierSet(RawEnum k) : keywords_(k) {}
  auto Union(RawEnum to_set) const -> KeywordModifierSet {
    KeywordModifierSet ret;
    ret.keywords_ = static_cast<RawEnum>(keywords_ | to_set);
    return ret;
  }
  auto SetPrivate() const -> KeywordModifierSet { return Union(Private); }
  auto SetProtected() const -> KeywordModifierSet { return Union(Protected); }
  auto SetAbstract() const -> KeywordModifierSet { return Union(Abstract); }
  auto SetBase() const -> KeywordModifierSet { return Union(Base); }
  auto SetDefault() const -> KeywordModifierSet { return Union(Default); }
  auto SetFinal() const -> KeywordModifierSet { return Union(Final); }
  auto SetImpl() const -> KeywordModifierSet { return Union(Impl); }
  auto SetVirtual() const -> KeywordModifierSet { return Union(Virtual); }

  // Returns the set difference or relative complement `*this \ to_unset`.
  auto Minus(RawEnum to_unset) const -> KeywordModifierSet {
    return Intersect(static_cast<RawEnum>(~to_unset));
  }
  auto UnsetPrivate() const -> KeywordModifierSet { return Minus(Private); }
  auto UnsetProtected() const -> KeywordModifierSet { return Minus(Protected); }
  auto UnsetAbstract() const -> KeywordModifierSet { return Minus(Abstract); }
  auto UnsetBase() const -> KeywordModifierSet { return Minus(Base); }
  auto UnsetDefault() const -> KeywordModifierSet { return Minus(Default); }
  auto UnsetFinal() const -> KeywordModifierSet { return Minus(Final); }
  auto UnsetImpl() const -> KeywordModifierSet { return Minus(Impl); }
  auto UnsetVirtual() const -> KeywordModifierSet { return Minus(Virtual); }

  // Returns true if any modifier from `to_check` is in the set.
  auto Overlaps(RawEnum to_check) const -> bool { return keywords_ & to_check; }

  auto HasPrivate() const -> bool { return Overlaps(Private); }
  auto HasProtected() const -> bool { return Overlaps(Protected); }
  auto HasAbstract() const -> bool { return Overlaps(Abstract); }
  auto HasBase() const -> bool { return Overlaps(Base); }
  auto HasDefault() const -> bool { return Overlaps(Default); }
  auto HasFinal() const -> bool { return Overlaps(Final); }
  auto HasImpl() const -> bool { return Overlaps(Impl); }
  auto HasVirtual() const -> bool { return Overlaps(Virtual); }

  auto is_empty() const -> bool { return keywords_ == 0; }
  auto Intersect(RawEnum with) const -> KeywordModifierSet {
    KeywordModifierSet ret;
    ret.keywords_ = static_cast<RawEnum>(keywords_ & with);
    return ret;
  }

  auto GetRaw() const -> RawEnum { return keywords_; }

 private:
  RawEnum keywords_;
};

struct DeclState {
  // What kind of declaration
  // FIXME: `Fn` or `Function`?
  enum DeclKind { FileScope, Class, NamedConstraint, Fn, Interface, Let, Var };

  DeclState(DeclKind decl_kind, Parse::Node parse_node)
      : first_node(parse_node), kind(decl_kind) {}

  // Nodes of modifiers on this declaration
  Parse::Node saw_access_modifier = Parse::Node::Invalid;
  Parse::Node saw_decl_modifier = Parse::Node::Invalid;

  // Node corresponding to the first token of the declaration.
  Parse::Node first_node;

  // These fields are last because they are smaller.

  // Invariant: members of the set are `saw_access_modifier` and `saw_other_mod`
  // if valid.
  KeywordModifierSet found;

  DeclKind kind;
};

// Stack of `DeclStat` values, representing all the declarations currently
// nested within.
// Invariant: Bottom of the stack always has a "DeclState::FileScope" entry.
class DeclStateStack {
 public:
  DeclStateStack() {
    s_.emplace_back(DeclState::FileScope, Parse::Node::Invalid);
  }

  // Enter a declaration of kind `k`, with `parse_node` for the introducer
  // token.
  auto Push(DeclState::DeclKind k, Parse::Node parse_node) {
    s_.push_back(DeclState(k, parse_node));
  }

  // Access the most recently entered declaration.
  auto innermost() -> DeclState& { return s_.back(); }

  // Get the state for the declaration containing the innermost declaration.
  // Requires that the innermost declaration is not `FileScope`.
  auto containing() const -> const DeclState& {
    CARBON_CHECK(s_.size() >= 2);
    return s_[s_.size() - 2];
  }

  // Leave a declaration of kind `k`.
  auto Pop(DeclState::DeclKind k) {
    CARBON_CHECK(s_.back().kind == k);
    s_.pop_back();
    CARBON_CHECK(!s_.empty());
  }

 private:
  llvm::SmallVector<DeclState> s_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DECL_STATE_H_
