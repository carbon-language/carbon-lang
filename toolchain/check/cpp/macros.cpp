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
#include "toolchain/check/member_access.h"
#include "toolchain/check/type_completion.h"

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

  clang::APValue ap_value = evaluated_result.Val;
  // TODO: Add support for other types.
  if (result_expr->isGLValue()) {
    const auto* value_decl =
        ap_value.getLValueBase().get<const clang::ValueDecl*>();

    if (!ap_value.hasLValuePath()) {
      context.TODO(loc_id, "Macro expanded to lvalue with no path");
      return SemIR::ErrorInst::InstId;
    }

    if (ap_value.isLValueOnePastTheEnd()) {
      context.TODO(loc_id, "Macro expanded to a one-past-the-end lvalue");
      return SemIR::ErrorInst::InstId;
    }

    auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(
        // TODO: can this const_cast be avoided?
        const_cast<clang::ValueDecl*>(value_decl));

    auto inst_id = ImportCppDecl(context, loc_id, key);
    if (ap_value.getLValuePath().size() == 0) {
      return inst_id;
    }

    // Import the base type so that its fields can be accessed.
    auto var_storage = context.insts().GetAs<SemIR::VarStorage>(inst_id);
    // TODO: currently an error isn't reachable here because incomplete
    // array types can't be imported. Once that changes, switch to
    // `RequireCompleteType` and handle the error.
    CompleteTypeOrCheckFail(context, var_storage.type_id);

    clang::QualType qual_type = ap_value.getLValueBase().getType();
    for (const auto& entry : ap_value.getLValuePath()) {
      if (qual_type->isArrayType()) {
        context.TODO(loc_id, "Macro expanded to array type");
      } else {
        const auto* decl =
            cast<clang::Decl>(entry.getAsBaseOrMember().getPointer());

        const auto* field_decl = dyn_cast<clang::FieldDecl>(decl);
        if (!field_decl) {
          context.TODO(loc_id, "Macro expanded to a base class subobject");
          return SemIR::ErrorInst::InstId;
        }

        auto field_inst_id =
            ImportCppDecl(context, loc_id,
                          SemIR::ClangDeclKey::ForNonFunctionDecl(
                              const_cast<clang::FieldDecl*>(field_decl)));

        if (field_inst_id == SemIR::ErrorInst::InstId) {
          context.TODO(loc_id,
                       "Unsupported field in macro expansion: " +
                           ap_value.getAsString(context.ast_context(),
                                                result_expr->getType()));
          return SemIR::ErrorInst::InstId;
        }

        const SemIR::FieldDecl& field_decl_inst =
            context.insts().GetAs<SemIR::FieldDecl>(field_inst_id);

        qual_type = field_decl->getType();
        inst_id = PerformMemberAccess(context, loc_id, inst_id,
                                      field_decl_inst.name_id);
      }
    }

    return inst_id;
  } else {
    auto const_id =
        MapAPValueToConstant(context, loc_id, ap_value, result_expr->getType());
    if (const_id == SemIR::ConstantId::NotConstant) {
      context.TODO(loc_id,
                   "Unsupported: macro evaluated to a constant of type: " +
                       result_expr->getType().getAsString());
      return SemIR::ErrorInst::InstId;
    }

    return context.constant_values().GetInstId(const_id);
  }
}

}  // namespace Carbon::Check
