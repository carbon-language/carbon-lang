// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_HANDLE_MODIFIER_H_
#define CARBON_TOOLCHAIN_CHECK_HANDLE_MODIFIER_H_

#include "toolchain/check/context.h"

namespace Carbon::Check {

struct DeclModifierKeywords {
  // At most one of these, and if present they must be first:
  bool private_ = false;
  bool protected_ = false;

  // Only one of these allowed:
  bool abstract_ = false;
  bool base_ = false;
  bool default_ = false;
  bool final_ = false;
  bool override_ = false;
  bool virtual_ = false;
};

// Pops the DeclModifierKeyword parse nodes from context. Reports a diagnostic
// if they contain repeated modifiers, modifiers in the incorrect order, or
// modifiers not in `allowed`. Returns modifiers that were both found and
// allowed.
auto ValidateModifiers(Context& context, DeclModifierKeywords allowed)
    -> DeclModifierKeywords;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_HANDLE_MODIFIER_H_
