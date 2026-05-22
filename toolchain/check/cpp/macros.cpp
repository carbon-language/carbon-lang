// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/macros.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "common/check.h"
#include "toolchain/check/cpp/constant.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/literal.h"

namespace Carbon::Check {

// Maps a Clang literal expression to a Carbon constant.
static auto MapConstant(Context& context, SemIR::LocId loc_id,
                        clang::Expr* expr) -> SemIR::InstId {
  CARBON_CHECK(expr, "empty expression");

  if (auto* string_literal = dyn_cast<clang::StringLiteral>(expr)) {
    if (!string_literal->isOrdinary() && !string_literal->isUTF8()) {
      context.TODO(loc_id,
                   llvm::formatv("Unsupported: string literal type: {0}",
                                 expr->getType()));
      return SemIR::ErrorInst::InstId;
    }
    StringLiteralValueId string_id =
        context.string_literal_values().Add(string_literal->getString());
    auto inst_id =
        MakeStringLiteral(context, Parse::StringLiteralId::None, string_id);
    return inst_id;
  } else if (isa<clang::CXXNullPtrLiteralExpr>(expr)) {
    auto type_id = ImportCppType(context, loc_id, expr->getType()).type_id;
    return GetOrAddInst<SemIR::UninitializedValue>(context, SemIR::LocId::None,
                                                   {.type_id = type_id});
  }

  context.TODO(loc_id,
               llvm::formatv("Unsupported: C++ constant expression type: '{0}'",
                             expr->getType().getAsString()));
  return SemIR::ErrorInst::InstId;
}

auto TryEvaluateMacro(Context& context, SemIR::LocId loc_id,
                      SemIR::NameId name_id, clang::MacroInfo* macro_info)
    -> SemIR::InstId {
  auto name_str_opt = context.names().GetAsStringIfIdentifier(name_id);
  CARBON_CHECK(macro_info, "macro info missing");

  if (macro_info->getNumTokens() == 0) {
    context.TODO(loc_id, "Unsupported: macro with 0 replacement tokens");
    return SemIR::ErrorInst::InstId;
  }

  clang::Sema& sema = context.clang_sema();
  clang::Preprocessor& preprocessor = sema.getPreprocessor();
  auto& parser = context.cpp_context()->parser();

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

  preprocessor.EnterTokenStream(tokens, /*DisableMacroExpansion=*/false,
                                /*IsReinject=*/false);
  parser.ConsumeAnyToken(true);

  clang::ExprResult result = parser.ParseConstantExpression();

  clang::Expr* result_expr = result.get();

  bool success =
      !result.isInvalid() && parser.getCurToken().is(clang::tok::eof);

  if (!success) {
    parser.SkipUntil(clang::tok::eof);
    CARBON_DIAGNOSTIC(
        InCppMacroEvaluation, Error,
        "failed to parse macro Cpp.{0} to a valid constant expression",
        std::string);
    context.emitter().Emit(loc_id, InCppMacroEvaluation, (*name_str_opt).str());
    return SemIR::ErrorInst::InstId;
  }

  result_expr = result_expr->IgnoreParenImpCasts();

  if (isa<clang::StringLiteral>(result_expr) ||
      isa<clang::CXXNullPtrLiteralExpr>(result_expr)) {
    return MapConstant(context, loc_id, result_expr);
  }

  clang::Expr::EvalResult evaluated_result;
  if (!result_expr->EvaluateAsConstantExpr(evaluated_result,
                                           sema.getASTContext())) {
    CARBON_FATAL("failed to evaluate macro as constant expression");
  }

  auto const_id = MapAPValueToConstant(context, loc_id, evaluated_result.Val,
                                       result_expr->getType(),
                                       /*is_lvalue=*/result_expr->isGLValue());
  if (const_id == SemIR::ConstantId::NotConstant) {
    context.TODO(loc_id,
                 "Unsupported: macro evaluated to a constant of type: " +
                     result_expr->getType().getAsString());
    return SemIR::ErrorInst::InstId;
  }

  return context.constant_values().GetInstId(const_id);
}

}  // namespace Carbon::Check
