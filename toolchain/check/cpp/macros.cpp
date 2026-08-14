// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/macros.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "common/check.h"
#include "toolchain/check/cpp/constant.h"
#include "toolchain/check/cpp/generate_ast.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
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
                      clang::IdentifierInfo* identifier_info,
                      clang::MacroInfo* macro_info) -> SemIR::InstId {
  CARBON_CHECK(macro_info, "macro info missing");
  if (macro_info->getNumTokens() == 0) {
    context.TODO(loc_id, "Unsupported: macro with 0 replacement tokens");
    return SemIR::ErrorInst::InstId;
  }

  auto& ast_context = context.cpp_context()->ast_context();
  auto& parser = context.cpp_context()->parser();
  auto& preprocessor = context.cpp_context()->sema().getPreprocessor();

  // Enter the macro name, not the macro body, so we properly suppress recursive
  // expansion.
  clang::Token macro_name[1];
  macro_name[0].startToken();
  macro_name[0].setKind(clang::tok::identifier);
  macro_name[0].setIdentifierInfo(identifier_info);
  macro_name[0].setLocation(GetCppLocation(context, loc_id));
  preprocessor.EnterTokenStream(macro_name, /*DisableMacroExpansion=*/false,
                                /*IsReinject=*/false);

  CARBON_CHECK(parser.getCurToken().is(clang::tok::eof));
  parser.ConsumeToken();

  clang::ExprResult result = parser.ParseConstantExpression();
  clang::Expr* result_expr = result.get();

  // Consume any remaining tokens to advance the preprocessor and parser past
  // the end of the macro.
  bool success =
      !result.isInvalid() && parser.getCurToken().is(clang::tok::eof);
  while (!parser.getCurToken().is(clang::tok::eof)) {
    parser.ConsumeAnyToken(true);
  }

  if (!success) {
    CARBON_DIAGNOSTIC(
        InCppMacroEvaluation, Error,
        "failed to parse macro Cpp.{0} to a valid constant expression",
        std::string);
    context.emitter().Emit(loc_id, InCppMacroEvaluation,
                           identifier_info->getName().str());
    return SemIR::ErrorInst::InstId;
  }

  result_expr = result_expr->IgnoreParenImpCasts();

  if (isa<clang::StringLiteral>(result_expr) ||
      isa<clang::CXXNullPtrLiteralExpr>(result_expr)) {
    return MapConstant(context, loc_id, result_expr);
  }

  clang::Expr::EvalResult evaluated_result;
  if (!result_expr->EvaluateAsConstantExpr(evaluated_result, ast_context)) {
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
