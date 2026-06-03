// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/constant.h"

#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/type_mapping.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/format_providers.h"

namespace Carbon::Check {

static auto MapLValueToConstant(Context& context, SemIR::LocId loc_id,
                                const clang::APValue& ap_value,
                                clang::QualType type) -> SemIR::ConstantId {
  CARBON_CHECK(ap_value.isLValue(), "not an LValue");

  const auto* value_decl =
      ap_value.getLValueBase().get<const clang::ValueDecl*>();

  if (!ap_value.hasLValuePath()) {
    context.TODO(loc_id, "lvalue has no path");
    return SemIR::ErrorInst::ConstantId;
  }

  if (ap_value.isLValueOnePastTheEnd()) {
    context.TODO(loc_id, "one-past-the-end lvalue");
    return SemIR::ErrorInst::ConstantId;
  }

  auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(
      // TODO: can this const_cast be avoided?
      const_cast<clang::ValueDecl*>(value_decl));

  auto inst_id = ImportCppDecl(context, loc_id, key);
  if (ap_value.getLValuePath().empty()) {
    return context.constant_values().Get(inst_id);
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
      context.TODO(loc_id, "lvalue path contains an array type");
    } else {
      const auto* decl =
          cast<clang::Decl>(entry.getAsBaseOrMember().getPointer());

      const auto* field_decl = dyn_cast<clang::FieldDecl>(decl);
      if (!field_decl) {
        context.TODO(loc_id, "lvalue path contains a base class subobject");
        return SemIR::ErrorInst::ConstantId;
      }

      auto field_inst_id =
          ImportCppDecl(context, loc_id,
                        SemIR::ClangDeclKey::ForNonFunctionDecl(
                            const_cast<clang::FieldDecl*>(field_decl)));

      if (field_inst_id == SemIR::ErrorInst::InstId) {
        context.TODO(loc_id,
                     "unsupported field in lvalue path: " +
                         ap_value.getAsString(context.ast_context(), type));
        return SemIR::ErrorInst::ConstantId;
      }

      const SemIR::FieldDecl& field_decl_inst =
          context.insts().GetAs<SemIR::FieldDecl>(field_inst_id);

      qual_type = field_decl->getType();
      inst_id = PerformMemberAccess(context, loc_id, inst_id,
                                    field_decl_inst.name_id);
    }
  }

  return context.constant_values().Get(inst_id);
}

auto MapAPValueToConstant(Context& context, SemIR::LocId loc_id,
                          const clang::APValue& ap_value, clang::QualType type,
                          bool is_lvalue) -> SemIR::ConstantId {
  SemIR::TypeId type_id = ImportCppType(context, loc_id, type).type_id;
  if (!type_id.has_value()) {
    return SemIR::ConstantId::NotConstant;
  }

  if (is_lvalue) {
    return MapLValueToConstant(context, loc_id, ap_value, type);
  } else if (type->isPointerType()) {
    auto const_id = MapLValueToConstant(context, loc_id, ap_value, type);
    auto inst_id = AddInst<SemIR::AddrOf>(
        context, loc_id,
        {.type_id = type_id,
         .lvalue_id = context.constant_values().GetInstId(const_id)});
    return context.constant_values().Get(inst_id);
  } else if (ap_value.isInt()) {
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
    context.TODO(loc_id, "unsupported conversion to constant from APValue " +
                             ap_value.getAsString(context.ast_context(), type));
    return SemIR::ErrorInst::ConstantId;
  }
}

static auto MapAPValueToConstantForConstexpr(Context& context,
                                             SemIR::LocId loc_id,
                                             const clang::APValue& ap_value,
                                             clang::QualType type)
    -> SemIR::ConstantId {
  bool is_lvalue = false;
  if (type->isReferenceType()) {
    is_lvalue = true;
    type = type.getNonReferenceType();
  }
  return MapAPValueToConstant(context, loc_id, ap_value, type, is_lvalue);
}

auto EvalCppVarDecl(Context& context, SemIR::LocId loc_id,
                    const clang::VarDecl* var_decl, SemIR::TypeId type_id)
    -> SemIR::ConstantId {
  // If the C++ global is constant, map it to a Carbon constant.
  if (var_decl->isUsableInConstantExpressions(context.ast_context())) {
    if (const auto* ap_value = var_decl->getEvaluatedValue()) {
      auto clang_type = MapToCppType(context, type_id);
      if (clang_type.isNull()) {
        context.TODO(loc_id, "failed to map C++ type to Carbon");
        return SemIR::ErrorInst::ConstantId;
      }

      return MapAPValueToConstantForConstexpr(context, loc_id, *ap_value,
                                              clang_type);
    }
  }

  return SemIR::ConstantId::NotConstant;
}

auto MapConstantToAPValue(Context& context, SemIR::InstId const_inst_id,
                          clang::QualType param_type)
    -> std::optional<clang::APValue> {
  if (param_type->isIntegerType()) {
    const bool is_signed = param_type->isSignedIntegerOrEnumerationType();
    if (auto int_value =
            context.insts().TryGetAs<SemIR::IntValue>(const_inst_id)) {
      const auto& ap_int = context.ints().Get(int_value->int_id);
      auto aps_int =
          llvm::APSInt(ap_int, !is_signed)
              .extOrTrunc(context.ast_context().getIntWidth(param_type));
      return clang::APValue(aps_int);
    } else if (auto bool_value = context.insts().TryGetAs<SemIR::BoolLiteral>(
                   const_inst_id)) {
      llvm::APInt ap_int(context.ast_context().getIntWidth(param_type),
                         bool_value->value.ToBool(), is_signed);
      auto aps_int =
          llvm::APSInt(ap_int, !is_signed)
              .extOrTrunc(context.ast_context().getIntWidth(param_type));
      return clang::APValue(aps_int);
    }
  } else if (param_type->isFloatingType()) {
    if (auto float_value =
            context.insts().TryGetAs<SemIR::FloatValue>(const_inst_id)) {
      const auto& ap_float = context.floats().Get(float_value->float_id);
      return clang::APValue(ap_float);
    }
  }

  // TODO: support additional parameter types.
  return std::nullopt;
}

static auto ConvertArgToExpr(Context& context, SemIR::InstId arg_inst_id,
                             clang::QualType param_type) -> clang::Expr* {
  if (auto temporary =
          context.insts().TryGetAs<SemIR::Temporary>(arg_inst_id)) {
    arg_inst_id = temporary->init_id;
  }

  auto const_inst_id = context.constant_values().GetConstantInstId(arg_inst_id);
  if (!const_inst_id.has_value()) {
    return nullptr;
  }

  auto ap_value = MapConstantToAPValue(context, const_inst_id, param_type);
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
      /*NameLoc=*/GetCppLocation(context, loc_id), function_decl->getType(),
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
      /*RParenLoc=*/GetCppLocation(context, loc_id),
      clang::FPOptionsOverride());

  // Evaluate the expr as a constant and map that to Carbon constant.
  clang::SmallVector<clang::PartialDiagnosticAt> notes;
  clang::Expr::EvalResult eval_result;
  eval_result.Diag = &notes;
  if (!call_expr->EvaluateAsConstantExpr(eval_result, context.ast_context())) {
    if (!function_decl->isConsteval()) {
      return SemIR::ConstantId::NotConstant;
    }
    context.clang_sema().Diag(call_expr->getBeginLoc(),
                              clang::diag::err_invalid_consteval_call)
        << function_decl << /*is consteval*/ true;
    for (const auto& note : notes) {
      context.clang_sema().Diag(note.first, note.second);
    }
    return SemIR::ErrorInst::ConstantId;
  }

  return MapAPValueToConstantForConstexpr(context, loc_id, eval_result.Val,
                                          function_decl->getCallResultType());
}

auto MaybeModifyCppThunkCallForConstEval(Context& context, SemIR::Call* call)
    -> void {
  clang::FunctionDecl* function_decl = nullptr;
  SemIR::InstId thunk_callee_inst_id = SemIR::InstId::None;

  // Check if the callee is a C++ thunk for a constexpr function. If so,
  // fill in `function_decl` and `thunk_callee_inst_id`.
  auto callee = SemIR::GetCallee(context.sem_ir(), call->callee_id);
  if (auto* callee_function = std::get_if<SemIR::CalleeFunction>(&callee)) {
    auto function = context.functions().Get(callee_function->function_id);

    thunk_callee_inst_id = function.cpp_thunk_callee();
    if (!thunk_callee_inst_id.has_value()) {
      return;
    }
    auto thunk_callee_function = context.functions().Get(
        context.insts()
            .GetAs<SemIR::FunctionDecl>(thunk_callee_inst_id)
            .function_id);

    function_decl = cast<clang::FunctionDecl>(
        context.clang_decls()
            .Get(context.clang_decls().Lookup(
                thunk_callee_function.first_decl_id()))
            .GetAsKey()
            .decl);

    if (!(function_decl->isConstexpr() || function_decl->isConsteval())) {
      return;
    }

    if (function_decl->isDefaulted()) {
      return;
    }
  } else {
    return;
  }

  auto thunk_args = context.inst_blocks().Get(call->args_id);

  // Get the new call arguments. This drops the return slot arg, if
  // present. It also remaps arguments that are a pointer in the thunk,
  // but a non-pointer in the callee.
  llvm::SmallVector<SemIR::InstId> new_args;
  for (auto [arg_inst_id, parm_var_decl] :
       llvm::zip(thunk_args, function_decl->parameters())) {
    auto parm_type = parm_var_decl->getType();
    auto new_arg_inst_id = arg_inst_id;

    // TODO: reuse the logic in `check/cpp/thunk.cpp` to determine
    // whether to dereference the argument.
    if (!parm_type->isPointerType()) {
      if (auto addr_of = context.insts().TryGetAs<SemIR::AddrOf>(arg_inst_id)) {
        new_arg_inst_id = addr_of->lvalue_id;
      }
    }

    new_args.push_back(new_arg_inst_id);
  }

  call->callee_id = thunk_callee_inst_id;
  call->args_id = context.inst_blocks().AddCanonical(new_args);
}

}  // namespace Carbon::Check
