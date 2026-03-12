// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/constant.h"

#include "toolchain/check/cpp/import.h"
#include "toolchain/check/eval.h"
#include "toolchain/diagnostics/format_providers.h"

namespace Carbon::Check {

auto MapAPValueToConstant(Context& context, SemIR::LocId loc_id,
                          const clang::APValue& ap_value, clang::QualType type)
    -> SemIR::ConstantId {
  SemIR::TypeId type_id = ImportCppType(context, loc_id, type).type_id;
  if (!type_id.has_value()) {
    return SemIR::ConstantId::NotConstant;
  }

  if (ap_value.isInt()) {
    if (type->isBooleanType()) {
      auto value = SemIR::BoolValue::From(!ap_value.getInt().isZero());
      return TryEvalInst(
          context, SemIR::BoolLiteral{.type_id = type_id, .value = value});
    } else {
      CARBON_CHECK(type->isIntegralOrEnumerationType());

      IntId int_id = context.ints().Add(ap_value.getInt());
      return TryEvalInst(context,
                         SemIR::IntValue{.type_id = type_id, .int_id = int_id});
    }
  } else if (ap_value.isFloat()) {
    FloatId float_id = context.floats().Add(ap_value.getFloat());
    return TryEvalInst(
        context, SemIR::FloatValue{.type_id = type_id, .float_id = float_id});
  } else {
    // TODO: support other types.
    return SemIR::ConstantId::NotConstant;
  }
}

static auto ConvertConstantToAPValue(Context& context,
                                     SemIR::InstId const_inst_id,
                                     clang::QualType param_type)
    -> std::optional<clang::APValue> {
  if (param_type->isIntegerType()) {
    if (auto int_value =
            context.insts().TryGetAs<SemIR::IntValue>(const_inst_id)) {
      const auto& ap_int = context.ints().GetAtWidth(
          int_value->int_id, context.ast_context().getIntWidth(param_type));

      auto aps_int =
          llvm::APSInt(ap_int, !param_type->isSignedIntegerOrEnumerationType())
              .extOrTrunc(context.ast_context().getIntWidth(param_type));

      return clang::APValue(aps_int);
    }
  }

  // TODO: support additional parameter types.
  return std::nullopt;
}

static auto ConvertArgToExpr(Context& context, SemIR::InstId arg_inst_id,
                             clang::QualType param_type) -> clang::Expr* {
  auto const_inst_id = context.constant_values().GetConstantInstId(arg_inst_id);
  if (!const_inst_id.has_value()) {
    return nullptr;
  }

  auto ap_value = ConvertConstantToAPValue(context, const_inst_id, param_type);
  if (!ap_value.has_value()) {
    return nullptr;
  }

  auto* opaque_value_expr = new (context.ast_context()) clang::OpaqueValueExpr(
      clang::SourceLocation(), param_type, clang::VK_PRValue);

  return clang::ConstantExpr::Create(context.ast_context(), opaque_value_expr,
                                     *ap_value);
}

auto EvalCppCall(Context& context, SemIR::LocId loc_id,
                 SemIR::ClangDeclId clang_decl_id, SemIR::InstBlockId args_id)
    -> SemIR::ConstantId {
  const auto& args = context.inst_blocks().Get(args_id);

  auto* decl = context.clang_decls().Get(clang_decl_id).GetAsKey().decl;

  auto* function_decl = cast<clang::FunctionDecl>(decl);

  // Create expr for the function declaration.
  auto* decl_ref_expr = clang::DeclRefExpr::Create(
      context.ast_context(), /*QualifierLoc=*/clang::NestedNameSpecifierLoc(),
      /*TemplateKWLoc=*/clang::SourceLocation(), function_decl,
      /*RefersToEnclosingVariableOrCapture=*/false,
      /*NameLoc=*/clang::SourceLocation(), function_decl->getType(),
      clang::VK_LValue);

  // Cast to a function pointer type.
  auto function_ptr_type =
      context.ast_context().getPointerType(function_decl->getType());
  auto* implicit_cast_expr = clang::ImplicitCastExpr::Create(
      context.ast_context(), function_ptr_type,
      clang::CK_FunctionToPointerDecay, decl_ref_expr, nullptr,
      clang::VK_PRValue, clang::FPOptionsOverride());

  // Convert the arguments to exprs.
  clang::SmallVector<clang::Expr*> arg_exprs;
  for (auto [arg_inst_id, parm_var_decl] :
       llvm::zip(args, function_decl->parameters())) {
    if (auto* arg_expr =
            ConvertArgToExpr(context, arg_inst_id, parm_var_decl->getType())) {
      arg_exprs.push_back(arg_expr);
    } else {
      return SemIR::ConstantId::NotConstant;
    }
  }

  // Create an expr to call the function.
  auto* call_expr = clang::CallExpr::Create(
      context.ast_context(), implicit_cast_expr, arg_exprs,
      function_decl->getCallResultType(), clang::VK_PRValue,
      /*RParenLoc=*/clang::SourceLocation(), clang::FPOptionsOverride());

  // Evaluate the expr as a constant and map that to Carbon constant.
  clang::Expr::EvalResult eval_result;
  if (!call_expr->EvaluateAsConstantExpr(eval_result, context.ast_context())) {
    // TODO: improve this diagnostic with information from `eval_result`.
    CARBON_DIAGNOSTIC(CppConstexprEval, Error,
                      "failed to evaluate {0:consteval|constexpr} function "
                      "call as a constant",
                      Diagnostics::BoolAsSelect);
    context.emitter().Emit(loc_id, CppConstexprEval,
                           function_decl->isConsteval());
    return SemIR::ErrorInst::ConstantId;
  }
  return MapAPValueToConstant(context, loc_id, eval_result.Val,
                              function_decl->getCallResultType());
}

}  // namespace Carbon::Check
