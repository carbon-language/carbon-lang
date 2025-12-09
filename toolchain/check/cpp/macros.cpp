// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/macros.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"

namespace Carbon::Check {

auto TryEvaluateMacroToConstant(Context& context, SemIR::LocId loc_id,
                                SemIR::NameId name_id,
                                clang::MacroInfo* macro_info) -> clang::Expr* {
  auto name_str_opt = context.names().GetAsStringIfIdentifier(name_id);
  CARBON_CHECK(macro_info, "macro info missing");

  if (macro_info->getNumTokens() == 0) {
    context.TODO(loc_id, "Unsupported: macro with 0 replacement tokens");
    return nullptr;
  }

  clang::Sema& sema = context.clang_sema();
  clang::Preprocessor& preprocessor = sema.getPreprocessor();
  clang::Parser parser(preprocessor, sema, false);

  llvm::SmallVector<clang::Token> tokens(macro_info->tokens().begin(),
                                         macro_info->tokens().end());

  clang::Token current_token = parser.getCurToken();

  // Add eof token
  clang::Token eof;
  eof.startToken();
  eof.setKind(clang::tok::eof);
  eof.setLocation(current_token.getEndLoc());
  tokens.push_back(eof);

  tokens.push_back(current_token);

  preprocessor.EnterTokenStream(tokens, false, false);
  parser.ConsumeAnyToken(true);

  clang::ExprResult result = parser.ParseConstantExpression();
  clang::Expr* result_expr = result.get();

  bool success =
      !result.isInvalid() && parser.getCurToken().is(clang::tok::eof);

  if (!success) {
    parser.SkipUntil(clang::tok::eof);
    CARBON_DIAGNOSTIC(
        InCppMacroEvaluation, Error,
        "failed to evaluate macro Cpp.{0} to a valid constant expression",
        std::string);
    context.emitter().Emit(loc_id, InCppMacroEvaluation, (*name_str_opt).str());
    return nullptr;
  }

  clang::Expr::EvalResult evaluated_result;
  if (!result_expr->EvaluateAsInt(evaluated_result, sema.getASTContext())) {
    context.TODO(loc_id, "non-integer constant expression in macro.");
    return nullptr;
  }
  return clang::IntegerLiteral::Create(
      sema.getASTContext(), evaluated_result.Val.getInt(),
      result_expr->getType(), result_expr->getExprLoc());
}

}  // namespace Carbon::Check
