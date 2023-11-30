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
  auto Set(RawEnum to_set) const -> KeywordModifierSet {
    KeywordModifierSet ret;
    ret.keywords_ = static_cast<RawEnum>(keywords_ | to_set);
    return ret;
  }
  auto SetPrivate() const -> KeywordModifierSet { return Set(Private); }
  auto SetProtected() const -> KeywordModifierSet { return Set(Protected); }
  auto SetAbstract() const -> KeywordModifierSet { return Set(Abstract); }
  auto SetBase() const -> KeywordModifierSet { return Set(Base); }
  auto SetDefault() const -> KeywordModifierSet { return Set(Default); }
  auto SetFinal() const -> KeywordModifierSet { return Set(Final); }
  auto SetImpl() const -> KeywordModifierSet { return Set(Impl); }
  auto SetVirtual() const -> KeywordModifierSet { return Set(Virtual); }

  auto Has(RawEnum to_check) const -> bool { return keywords_ & to_check; }
  auto HasPrivate() const -> bool { return Has(Private); }
  auto HasProtected() const -> bool { return Has(Protected); }
  auto HasAbstract() const -> bool { return Has(Abstract); }
  auto HasBase() const -> bool { return Has(Base); }
  auto HasDefault() const -> bool { return Has(Default); }
  auto HasFinal() const -> bool { return Has(Final); }
  auto HasImpl() const -> bool { return Has(Impl); }
  auto HasVirtual() const -> bool { return Has(Virtual); }
  auto GetRaw() const -> RawEnum { return keywords_; }

 private:
  RawEnum keywords_;
};

struct DeclState {
  // FIXME: `Fn` or `Function`?
  enum DeclKind { FileScope, Class, NamedConstraint, Fn, Interface, Let, Var };

  DeclState(DeclKind decl_kind, Parse::Node parse_node)
      : first_node(parse_node), kind(decl_kind) {}
  DeclState() : DeclState(FileScope, Parse::Node::Invalid) {}

  Parse::Node saw_access_mod = Parse::Node::Invalid;
  Parse::Node saw_decl_mod = Parse::Node::Invalid;
  Parse::Node first_node;
  // Invariant: members of the set are `saw_access_mod` and `saw_other_mod`
  // if valid.
  KeywordModifierSet found;
  DeclKind kind;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_DECL_STATE_H_
