// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_MACROS_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_MACROS_H_

#include "toolchain/check/context.h"

namespace Carbon::Check {

// Tries to evaluate the given macro to a constant expression. Returns the
// evaluated expression on success or nullptr otherwise. Currently supports only
// object-like macros that evaluate to an integer constant.
// TODO: Add support for other literal types.
auto TryEvaluateMacroToConstant(Context& context, SemIR::LocId loc_id,
                                SemIR::NameId name_id,
                                clang::MacroInfo* macro_info) -> clang::Expr*;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_MACROS_H_
