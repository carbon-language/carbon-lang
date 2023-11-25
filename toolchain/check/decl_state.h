// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_DECL_STATE_H_
#define CARBON_TOOLCHAIN_CHECK_DECL_STATE_H_

namespace Carbon::Check {

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
    Override = 1 << 6,
    Virtual = 1 << 7,

    // Sets of modifiers:
    Access = Private | Protected,
  };

  KeywordModifierSet() : keywords(static_cast<RawEnum>(0)) {}
  KeywordModifierSet(RawEnum k) : keywords(k) {}
  auto Set(RawEnum to_set) const -> KeywordModifierSet {
    KeywordModifierSet ret;
    ret.keywords = static_cast<RawEnum>(keywords | to_set);
    return ret;
  }
  auto SetPrivate() const -> KeywordModifierSet { return Set(Private); }
  auto SetProtected() const -> KeywordModifierSet { return Set(Protected); }
  auto SetAbstract() const -> KeywordModifierSet { return Set(Abstract); }
  auto SetBase() const -> KeywordModifierSet { return Set(Base); }
  auto SetDefault() const -> KeywordModifierSet { return Set(Default); }
  auto SetFinal() const -> KeywordModifierSet { return Set(Final); }
  auto SetOverride() const -> KeywordModifierSet { return Set(Override); }
  auto SetVirtual() const -> KeywordModifierSet { return Set(Virtual); }

  auto Has(unsigned to_Has) const -> bool { return keywords & to_Has; }
  auto HasPrivate() const -> bool { return Has(Private); }
  auto HasProtected() const -> bool { return Has(Protected); }
  auto HasAbstract() const -> bool { return Has(Abstract); }
  auto HasBase() const -> bool { return Has(Base); }
  auto HasDefault() const -> bool { return Has(Default); }
  auto HasFinal() const -> bool { return Has(Final); }
  auto HasOverride() const -> bool { return Has(Override); }
  auto HasVirtual() const -> bool { return Has(Virtual); }
  auto GetRaw() const -> RawEnum { return keywords; }

 private:
  RawEnum keywords;
};

struct DeclState {
  enum DeclKind { FileScope, Class, NamedConstraint, Fn, Interface, Let, Var };

  DeclState(DeclKind k, Parse::Node f) : first_node(f), kind(k) {}
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
