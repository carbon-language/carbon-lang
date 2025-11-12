// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/macros.h"

#include "clang/AST/ASTContext.h"
#include "clang/Sema/Sema.h"

namespace Carbon::Check {

auto TryEvaluateMacroToConstant(Context& context, SemIR::LocId loc_id,
                                SemIR::NameId name_id,
                                clang::MacroInfo* macro_info) -> clang::Expr* {
  auto name_str_opt = context.names().GetAsStringIfIdentifier(name_id);
  CARBON_CHECK(macro_info, "macro info missing");
  if (macro_info->getNumTokens() != 1) {
    context.TODO(loc_id,
                 llvm::formatv("Unsupported: macro with {0} replacement tokens",
                               macro_info->getNumTokens()));
    return nullptr;
  }
  const clang::Token& tok = macro_info->getReplacementToken(0);
  if (!tok.is(clang::tok::numeric_constant)) {
    context.TODO(loc_id,
                 "Unsupported: macro replacement token kind: " +
                     std::string(clang::tok::getTokenName(tok.getKind())));
    return nullptr;
  }
  clang::Sema& sema = context.clang_sema();
  clang::ExprResult result = sema.ActOnNumericConstant(tok);
  clang::Expr* result_expr = result.get();

  if (!result_expr || result.isInvalid()) {
    CARBON_DIAGNOSTIC(
        InCppMacroEvaluation, Error,
        "failed to evaluate macro Cpp.{0} to a valid constant expression",
        std::string);
    context.emitter().Emit(loc_id, InCppMacroEvaluation, (*name_str_opt).str());
    return nullptr;
  }

  return result_expr;
}

}  // namespace Carbon::Check
