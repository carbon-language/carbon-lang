// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_MODIFIERS_ALLOWED_H_
#define CARBON_TOOLCHAIN_CHECK_MODIFIERS_ALLOWED_H_

#include "toolchain/check/context.h"

namespace Carbon::Check {

// Reports a diagnostic (using `decl_name`) if access control modifiers on this
// are not allowed, and updates the declaration state in `context`.
auto CheckAccessModifiersOnDecl(Context& context, llvm::StringRef decl_name)
    -> void;

enum class ShouldReportContext { ExcludeContext, IncludeContext };
static constexpr auto IncludeContext = ShouldReportContext::IncludeContext;

// Reports a diagnostic (using `decl_name`) if declaration modifiers on this
// declaration are not in `allowed`. Assumes access modifiers were already
// handled using `CheckAccessModifiersOnDecl`. Returns modifiers that were both
// found and allowed (including access modifiers), and updates the declaration
// state in `context`. Pass `IncludeContext` to the last parameter to include
// the containing declaration as context in diagnostics.
auto ModifiersAllowedOnDecl(
    Context& context, KeywordModifierSet allowed, llvm::StringRef decl_name,
    ShouldReportContext report_context = ShouldReportContext::ExcludeContext)
    -> KeywordModifierSet;

// Like `ModifiersAllowedOnDecl` with `IncludeContext` except it uses a custom
// `context_string` to describe the context.
auto ModifiersAllowedOnDeclCustomContext(Context& context,
                                         KeywordModifierSet allowed,
                                         llvm::StringRef decl_name,
                                         llvm::StringRef context_string)
    -> KeywordModifierSet;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_MODIFIERS_ALLOWED_H_
