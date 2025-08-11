// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp_thunk.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/Sema/Sema.h"
#include "toolchain/check/call.h"
#include "toolchain/check/context.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/literal.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Returns the C++ thunk mangled name given the callee function.
static auto GenerateThunkMangledName(
    clang::MangleContext& mangle_context,
    const clang::FunctionDecl& callee_function_decl) -> std::string {
  RawStringOstream mangled_name_stream;
  mangle_context.mangleName(clang::GlobalDecl(&callee_function_decl),
                            mangled_name_stream);
  mangled_name_stream << ".carbon_thunk";

  return mangled_name_stream.TakeStr();
}

// Returns true if a C++ thunk is required for the given type. A C++ thunk is
// required for any type except for void, pointer types and signed 32-bit and
// 64-bit integers.
static auto IsThunkRequiredForType(Context& context, SemIR::TypeId type_id)
    -> bool {
  if (!type_id.has_value() || type_id == SemIR::ErrorInst::TypeId) {
    return false;
  }

  type_id = context.types().GetUnqualifiedType(type_id);

  switch (context.types().GetAsInst(type_id).kind()) {
    case SemIR::PointerType::Kind: {
      return false;
    }

    case SemIR::ClassType::Kind: {
      if (!context.types().IsComplete(type_id)) {
        // Signed integers of 32 or 64 bits should be completed when imported.
        return true;
      }

      if (!context.types().IsSignedInt(type_id)) {
        return true;
      }

      llvm::APInt bit_width =
          context.ints().Get(context.types().GetIntTypeInfo(type_id).bit_width);
      return bit_width != 32 && bit_width != 64;
    }

    default:
      return true;
  }
}

auto IsCppThunkRequired(Context& context, const SemIR::Function& function)
    -> bool {
  if (!function.clang_decl_id.has_value()) {
    return false;
  }

  if (function.self_param_id.has_value()) {
    // TODO: Support member methods.
    return false;
  }

  SemIR::TypeId return_type_id =
      function.GetDeclaredReturnType(context.sem_ir());
  if (return_type_id.has_value()) {
    // TODO: Support non-void return values.
    return false;
  }

  bool thunk_required_for_param = false;
  for (auto param_id :
       context.inst_blocks().GetOrEmpty(function.call_params_id)) {
    if (param_id == SemIR::ErrorInst::InstId) {
      return false;
    }
    if (!thunk_required_for_param &&
        IsThunkRequiredForType(
            context,
            context.insts().GetAs<SemIR::AnyParam>(param_id).type_id)) {
      thunk_required_for_param = true;
    }
  }

  return thunk_required_for_param;
}

// Returns whether the type is a pointer or a signed int of 32 or 64 bits.
static auto IsSimpleAbiType(clang::ASTContext& ast_context,
                            clang::QualType type) -> bool {
  if (type->isPointerType()) {
    return true;
  }

  if (const auto* builtin_type = type->getAs<clang::BuiltinType>()) {
    if (builtin_type->isSignedInteger()) {
      uint64_t type_size = ast_context.getIntWidth(type);
      return type_size == 32 || type_size == 64;
    }
  }

  return false;
}

// Creates the thunk parameter types given the callee function. Also returns for
// each type whether it is different from the matching callee function parameter
// type.
static auto BuildThunkParameterTypes(
    clang::ASTContext& ast_context,
    const clang::FunctionDecl& callee_function_decl)
    -> std::tuple<llvm::SmallVector<clang::QualType>, llvm::SmallVector<bool>> {
  std::tuple<llvm::SmallVector<clang::QualType>, llvm::SmallVector<bool>>
      result;
  auto& [thunk_param_types, param_type_changed] = result;

  unsigned num_params = callee_function_decl.getNumParams();
  thunk_param_types.reserve(num_params);
  param_type_changed.reserve(num_params);

  for (const clang::ParmVarDecl* callee_param :
       callee_function_decl.parameters()) {
    clang::QualType param_type = callee_param->getType();
    bool is_simple_abi_type = IsSimpleAbiType(ast_context, param_type);
    if (!is_simple_abi_type) {
      clang::QualType pointer_type = ast_context.getPointerType(param_type);
      param_type = ast_context.getAttributedType(
          clang::NullabilityKind::NonNull, pointer_type, pointer_type);
    }
    param_type_changed.push_back(!is_simple_abi_type);
    thunk_param_types.push_back(param_type);
  }

  return result;
}

// Returns the thunk parameters using the callee function parameter identifiers.
static auto BuildThunkParameters(
    clang::ASTContext& ast_context,
    const clang::FunctionDecl& callee_function_decl,
    clang::FunctionDecl* thunk_function_decl)
    -> llvm::SmallVector<clang::ParmVarDecl*> {
  clang::SourceLocation clang_loc = callee_function_decl.getLocation();

  unsigned num_params = thunk_function_decl->getNumParams();
  CARBON_CHECK(callee_function_decl.getNumParams() == num_params);

  const auto* thunk_function_proto_type =
      thunk_function_decl->getFunctionType()->getAs<clang::FunctionProtoType>();

  llvm::SmallVector<clang::ParmVarDecl*> thunk_params;
  thunk_params.reserve(num_params);
  for (unsigned i = 0; i < num_params; ++i) {
    clang::ParmVarDecl* thunk_param = clang::ParmVarDecl::Create(
        ast_context, thunk_function_decl, clang_loc, clang_loc,
        callee_function_decl.getParamDecl(i)->getIdentifier(),
        thunk_function_proto_type->getParamType(i), nullptr, clang::SC_None,
        nullptr);
    thunk_params.push_back(thunk_param);
  }
  return thunk_params;
}

// Returns the thunk function declaration given the callee function and the
// thunk parameter types.
static auto CreateThunkFunctionDecl(
    Context& context, const clang::FunctionDecl& callee_function_decl,
    llvm::ArrayRef<clang::QualType> thunk_param_types) -> clang::FunctionDecl* {
  clang::ASTContext& ast_context = context.ast_context();
  clang::SourceLocation clang_loc = callee_function_decl.getLocation();

  clang::IdentifierInfo& identifier_info = ast_context.Idents.get(
      callee_function_decl.getNameAsString() + "__carbon_thunk");

  const auto* callee_function_type = callee_function_decl.getFunctionType()
                                         ->castAs<clang::FunctionProtoType>();

  // TODO: Check whether we need to modify `ExtParameterInfo` in `ExtProtoInfo`.
  clang::QualType thunk_function_type = ast_context.getFunctionType(
      callee_function_decl.getReturnType(), thunk_param_types,
      callee_function_type->getExtProtoInfo());

  clang::DeclContext* decl_context = ast_context.getTranslationUnitDecl();
  // TODO: Thunks should not have external linkage, consider using `SC_Static`.
  clang::FunctionDecl* thunk_function_decl = clang::FunctionDecl::Create(
      ast_context, decl_context, clang_loc, clang_loc,
      clang::DeclarationName(&identifier_info), thunk_function_type,
      /*TInfo=*/nullptr, clang::SC_Extern);
  decl_context->addDecl(thunk_function_decl);

  thunk_function_decl->setParams(BuildThunkParameters(
      ast_context, callee_function_decl, thunk_function_decl));

  // Set always_inline.
  thunk_function_decl->addAttr(
      clang::AlwaysInlineAttr::CreateImplicit(ast_context));

  // Set asm("<callee function mangled name>.carbon_thunk").
  thunk_function_decl->addAttr(clang::AsmLabelAttr::CreateImplicit(
      ast_context,
      GenerateThunkMangledName(*context.sem_ir().clang_mangle_context(),
                               callee_function_decl),
      clang_loc));

  return thunk_function_decl;
}

// Takes the thunk function parameters and for each one creates an arg for the
// callee function which is the thunk parameter or its address.
static auto BuildCalleeArgs(clang::Sema& sema,
                            clang::FunctionDecl* thunk_function_decl,
                            llvm::ArrayRef<bool> param_type_changed)
    -> llvm::SmallVector<clang::Expr*> {
  llvm::SmallVector<clang::Expr*> call_args;
  size_t num_params = thunk_function_decl->getNumParams();
  CARBON_CHECK(param_type_changed.size() == num_params);
  call_args.reserve(num_params);
  for (unsigned i = 0; i < num_params; ++i) {
    clang::ParmVarDecl* thunk_param = thunk_function_decl->getParamDecl(i);
    clang::SourceLocation clang_loc = thunk_param->getLocation();

    clang::Expr* call_arg = sema.BuildDeclRefExpr(
        thunk_param, thunk_param->getType(), clang::VK_LValue, clang_loc);
    if (param_type_changed[i]) {
      // TODO: Consider inserting a cast to an rvalue. Note that we currently
      // pass pointers to non-temporary objects as the argument when calling a
      // thunk, so we'll need to either change that or generate different thunks
      // depending on whether we're moving from each parameter.
      clang::ExprResult deref_result =
          sema.BuildUnaryOp(nullptr, clang_loc, clang::UO_Deref, call_arg);
      CARBON_CHECK(deref_result.isUsable());
      call_arg = deref_result.get();
    }
    call_args.push_back(call_arg);
  }

  return call_args;
}

// Builds the thunk function body which calls the callee function using the call
// args and returns the callee function return value. Returns nullptr on
// failure.
static auto BuildThunkBody(clang::Sema& sema,
                           clang::FunctionDecl* callee_function_decl,
                           llvm::MutableArrayRef<clang::Expr*> call_args)
    -> clang::Stmt* {
  clang::SourceLocation clang_loc = callee_function_decl->getLocation();

  clang::DeclRefExpr* callee_function_ref = sema.BuildDeclRefExpr(
      callee_function_decl, callee_function_decl->getType(), clang::VK_PRValue,
      clang_loc);

  clang::ExprResult call_result = sema.BuildCallExpr(
      nullptr, callee_function_ref, clang_loc, call_args, clang_loc);
  if (!call_result.isUsable()) {
    return nullptr;
  }
  clang::Expr* call = call_result.get();

  clang::StmtResult return_result = sema.BuildReturnStmt(clang_loc, call);
  CARBON_CHECK(return_result.isUsable());
  return return_result.get();
}

auto BuildCppThunk(Context& context, const SemIR::Function& callee_function)
    -> clang::FunctionDecl* {
  clang::FunctionDecl* callee_function_decl =
      context.sem_ir()
          .clang_decls()
          .Get(callee_function.clang_decl_id)
          .decl->getAsFunction();
  CARBON_CHECK(callee_function_decl);

  // Build the thunk function declaration.
  auto [thunk_param_types, param_type_changed] =
      BuildThunkParameterTypes(context.ast_context(), *callee_function_decl);
  clang::FunctionDecl* thunk_function_decl = CreateThunkFunctionDecl(
      context, *callee_function_decl, thunk_param_types);

  // Build the thunk function body.
  clang::Sema& sema = context.sem_ir().clang_ast_unit()->getSema();
  clang::Sema::ContextRAII context_raii(sema, thunk_function_decl);
  sema.ActOnStartOfFunctionDef(nullptr, thunk_function_decl);

  llvm::SmallVector<clang::Expr*> call_args =
      BuildCalleeArgs(sema, thunk_function_decl, param_type_changed);
  clang::Stmt* body = BuildThunkBody(sema, callee_function_decl, call_args);
  sema.ActOnFinishFunctionBody(thunk_function_decl, body);
  if (!body) {
    return nullptr;
  }

  return thunk_function_decl;
}

auto PerformCppThunkCall(Context& context, SemIR::LocId loc_id,
                         SemIR::FunctionId callee_function_id,
                         llvm::ArrayRef<SemIR::InstId> callee_arg_ids,
                         SemIR::InstId thunk_callee_id) -> SemIR::InstId {
  llvm::ArrayRef<SemIR::InstId> callee_function_params =
      context.inst_blocks().GetOrEmpty(
          context.functions().Get(callee_function_id).call_params_id);

  llvm::ArrayRef<SemIR::InstId> thunk_function_params =
      context.inst_blocks().GetOrEmpty(
          context.functions()
              .Get(GetCalleeFunction(context.sem_ir(), thunk_callee_id)
                       .function_id)
              .call_params_id);

  size_t num_params = callee_function_params.size();
  CARBON_CHECK(thunk_function_params.size() == num_params);
  CARBON_CHECK(callee_arg_ids.size() == num_params);
  llvm::SmallVector<SemIR::InstId> thunk_arg_ids;
  thunk_arg_ids.reserve(callee_arg_ids.size());
  for (size_t i = 0; i < callee_function_params.size(); ++i) {
    SemIR::TypeId callee_param_type_id =
        context.insts()
            .GetAs<SemIR::AnyParam>(callee_function_params[i])
            .type_id;
    SemIR::TypeId thunk_param_type_id =
        context.insts()
            .GetAs<SemIR::AnyParam>(thunk_function_params[i])
            .type_id;

    SemIR::InstId arg_id = callee_arg_ids[i];
    if (callee_param_type_id != thunk_param_type_id) {
      CARBON_CHECK(thunk_param_type_id ==
                   GetPointerType(context, context.types().GetInstId(
                                               callee_param_type_id)));

      arg_id = Convert(context, loc_id, arg_id,
                       {.kind = ConversionTarget::CppThunkRef,
                        .type_id = callee_param_type_id});
      arg_id = AddInst<SemIR::AddrOf>(
          context, loc_id,
          {.type_id = thunk_param_type_id, .lvalue_id = arg_id});
    }
    thunk_arg_ids.push_back(arg_id);
  }

  return PerformCall(context, loc_id, thunk_callee_id, thunk_arg_ids);
}

}  // namespace Carbon::Check
