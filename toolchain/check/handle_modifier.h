// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_HANDLE_MODIFIER_H_
#define CARBON_TOOLCHAIN_CHECK_HANDLE_MODIFIER_H_

#include "toolchain/check/context.h"

namespace Carbon::Check {

class DeclModifierKeywords {
 public:
  enum DeclModifierKeywordsRaw {
    // At most one of these, and if present it must be first:
    Private = 1 << 0,
    Protected = 1 << 1,
    // At most one of these allowed for a given declaration:
    Abstract = 1 << 2,
    Base = 1 << 3,
    Default = 1 << 4,
    Final = 1 << 5,
    Override = 1 << 6,
    Virtual = 1 << 7
  };

  DeclModifierKeywords() : keywords(static_cast<DeclModifierKeywordsRaw>(0)) {}
  auto Set(DeclModifierKeywordsRaw to_set) const -> DeclModifierKeywords {
    DeclModifierKeywords ret;
    ret.keywords = static_cast<DeclModifierKeywordsRaw>(keywords | to_set);
    return ret;
  }
  auto SetPrivate() const -> DeclModifierKeywords { return Set(Private); }
  auto SetProtected() const -> DeclModifierKeywords { return Set(Protected); }
  auto SetAbstract() const -> DeclModifierKeywords { return Set(Abstract); }
  auto SetBase() const -> DeclModifierKeywords { return Set(Base); }
  auto SetDefault() const -> DeclModifierKeywords { return Set(Default); }
  auto SetFinal() const -> DeclModifierKeywords { return Set(Final); }
  auto SetOverride() const -> DeclModifierKeywords { return Set(Override); }
  auto SetVirtual() const -> DeclModifierKeywords { return Set(Virtual); }

  auto Has(unsigned to_Has) const -> bool { return keywords & to_Has; }
  auto HasPrivate() const -> bool { return Has(Private); }
  auto HasProtected() const -> bool { return Has(Protected); }
  auto HasAbstract() const -> bool { return Has(Abstract); }
  auto HasBase() const -> bool { return Has(Base); }
  auto HasDefault() const -> bool { return Has(Default); }
  auto HasFinal() const -> bool { return Has(Final); }
  auto HasOverride() const -> bool { return Has(Override); }
  auto HasVirtual() const -> bool { return Has(Virtual); }

 private:
  DeclModifierKeywordsRaw keywords;
};

// Pops any DeclModifierKeyword parse nodes from `context` and then the
// introducer node (using `pop_introducer`). Reports a diagnostic if they
// contain repeated modifiers, modifiers in the incorrect order, or modifiers
// not in `allowed`. Returns modifiers that were both found and allowed, and the
// parse node corresponding to the first token of the declaration.
auto ValidateModifiers(Context& context, DeclModifierKeywords allowed,
                       std::function<Parse::Node()> pop_introducer)
    -> std::pair<DeclModifierKeywords, Parse::Node>;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_HANDLE_MODIFIER_H_
