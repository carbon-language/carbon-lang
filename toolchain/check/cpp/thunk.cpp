// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/thunk.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/TypeBase.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"
#include "toolchain/check/call.h"
#include "toolchain/check/context.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/cpp/context.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/literal.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Generate and return a function:
// `void* operator new(__SIZE_TYPE__, void*) noexcept`.
static auto GeneratePlacementNewFunctionDecl(clang::ASTContext& context)
    -> clang::FunctionDecl* {
  clang::QualType size_type = context.getSizeType();
  clang::QualType void_ptr_type = context.VoidPtrTy;

  auto ext_info = clang::FunctionProtoType::ExtProtoInfo();
  ext_info.ExceptionSpec.Type = clang::EST_BasicNoexcept;

  clang::QualType function_type = context.getFunctionType(
      void_ptr_type, {size_type, void_ptr_type}, ext_info);

  clang::DeclarationName name =
      context.DeclarationNames.getCXXOperatorName(clang::OO_New);

  clang::FunctionDecl* function_decl = clang::FunctionDecl::Create(
      context, context.getTranslationUnitDecl(), clang::SourceLocation(),
      clang::SourceLocation(), name, function_type,
      /*TInfo=*/nullptr, clang::SC_None);

  clang::ParmVarDecl* size_param = clang::ParmVarDecl::Create(
      context, function_decl, clang::SourceLocation(), clang::SourceLocation(),
      nullptr, size_type, nullptr, clang::SC_None, nullptr);
  clang::ParmVarDecl* ptr_param = clang::ParmVarDecl::Create(
      context, function_decl, clang::SourceLocation(), clang::SourceLocation(),
      nullptr, void_ptr_type, nullptr, clang::SC_None, nullptr);

  function_decl->setParams({size_param, ptr_param});
  CARBON_CHECK(function_decl->isReservedGlobalPlacementOperator());
  return function_decl;
}

// Returns the GlobalDecl to use to represent the given function declaration.
// TODO: Refactor with `Lower::CreateGlobalDecl`.
static auto GetGlobalDecl(const clang::FunctionDecl* decl)
    -> clang::GlobalDecl {
  if (const auto* ctor = dyn_cast<clang::CXXConstructorDecl>(decl)) {
    return clang::GlobalDecl(ctor, clang::CXXCtorType::Ctor_Complete);
  }
  if (const auto* dtor = dyn_cast<clang::CXXDestructorDecl>(decl)) {
    return clang::GlobalDecl(dtor, clang::CXXDtorType::Dtor_Complete);
  }
  return clang::GlobalDecl(decl);
}

// Returns the C++ thunk mangled name given the callee function.
static auto GenerateThunkMangledName(
    clang::MangleContext& mangle_context,
    const clang::FunctionDecl* callee_function_decl,
    const SemIR::ClangDeclSignature& signature) -> std::string {
  RawStringOstream mangled_name_stream;
  if (callee_function_decl != nullptr) {
    mangle_context.mangleName(GetGlobalDecl(callee_function_decl),
                              mangled_name_stream);
  }
  switch (signature.kind) {
    case SemIR::ClangDeclSignature::Normal:
      mangled_name_stream << ".carbon_thunk";
      break;
    case SemIR::ClangDeclSignature::TuplePattern:
      mangled_name_stream << ".carbon_thunk_tuple";
      break;
  }

  // Append passing modes.
  // TODO: Pick one "likely" set of passing modes for the function and omit the
  // suffix for that signature.
  mangled_name_stream << ".";
  auto append_mode = [&](SemIR::ClangDeclSignature::PassingMode mode) {
    switch (mode) {
      case SemIR::ClangDeclSignature::PassingMode::ByValue:
        mangled_name_stream << "_";
        break;
      case SemIR::ClangDeclSignature::PassingMode::ByVar:
        mangled_name_stream << "v";
        break;
      case SemIR::ClangDeclSignature::PassingMode::ByRef:
        mangled_name_stream << "r";
        break;
    }
  };

  // If there is no decl, the callee is a function pointer, which we treat as
  // the thunk's `self` parameter.
  if (callee_function_decl == nullptr ||
      IsObjectMemberFunction(*callee_function_decl)) {
    append_mode(signature.self_passing_mode);
  }
  for (auto mode : signature.passing_modes) {
    append_mode(mode);
  }

  return mangled_name_stream.TakeStr();
}

// Returns whether the Carbon lowering for a parameter or return of this type is
// known to match the C++ lowering.
static auto IsSimpleAbiType(clang::ASTContext& ast_context,
                            clang::QualType type, bool for_parameter) -> bool {
  if (type->isVoidType() || type->isPointerType()) {
    return true;
  }

  if (type->isReferenceType()) {
    if (for_parameter) {
      // A reference parameter has a simple ABI if it's a non-const lvalue
      // reference.  Otherwise, we map it to pass-by-value, and it's only simple
      // if the type uses a pointer value representation.
      //
      // TODO: Check whether the pointee type maps to a Carbon type that uses a
      // pointer value representation, and treat it as simple if so.
      return type->isLValueReferenceType() &&
             !type->getPointeeType().isConstQualified();
    }

    // A reference return type is always mapped to a Carbon pointer, which uses
    // the same ABI rule as a C++ reference.
    return true;
  }

  if (const auto* enum_decl = type->getAsEnumDecl()) {
    // An enum type has a simple ABI if its underlying type does.
    type = enum_decl->getIntegerType();
    if (type.isNull()) {
      return false;
    }
  }

  if (const auto* builtin_type = type->getAs<clang::BuiltinType>()) {
    if (builtin_type->isIntegerType()) {
      uint64_t type_size = ast_context.getIntWidth(type);
      return type_size == 32 || type_size == 64;
    }
  }

  return false;
}

CalleeFunctionInfo::CalleeFunctionInfo(Context* context,
                                       clang::FunctionDecl* decl,
                                       SemIR::ClangDeclSignatureId signature_id)
    : context(context),
      decl_or_pointer_type(decl),
      sem_ir_loc(AddImportIRInst(context->sem_ir(), decl->getLocation())),
      function_type(decl->getType()->getAs<clang::FunctionProtoType>()),
      signature_id(signature_id),
      signature(&context->clang_decl_signatures().Get(signature_id)),
      num_callee_params(signature->num_params +
                        decl->hasCXXExplicitFunctionObjectParameter()) {
  auto& ast_context = decl->getASTContext();
  if (IsObjectMemberFunction(*decl)) {
    const auto* method_decl = dyn_cast<clang::CXXMethodDecl>(decl);
    if (method_decl->isImplicitObjectMemberFunction()) {
      self_param_kind = SelfParamKind::ImplicitObjectParam;
    } else {
      self_param_kind = SelfParamKind::ExplicitObjectParam;
    }
  } else {
    self_param_kind = SelfParamKind::None;
  }
  has_simple_return_type = IsSimpleAbiType(ast_context, effective_return_type(),
                                           /*for_parameter=*/false);
}

CalleeFunctionInfo::CalleeFunctionInfo(Context* context,
                                       const clang::Type* function_pointer_type)
    : context(context),
      self_param_kind(SelfParamKind::FunctionPointer),
      decl_or_pointer_type(function_pointer_type),
      sem_ir_loc(SemIR::LocId::None),
      function_type(function_pointer_type->getPointeeType()
                        ->getAs<clang::FunctionProtoType>()),
      signature_id(SemIR::ClangDeclSignatureId::None),
      signature(nullptr),
      num_callee_params(function_type->getNumParams()) {
  SemIR::ClangDeclSignature local_signature;
  local_signature.kind = SemIR::ClangDeclSignature::Normal;
  local_signature.num_params =
      static_cast<int32_t>(function_type->getNumParams());
  local_signature.self_passing_mode =
      SemIR::ClangDeclSignature::PassingMode::ByValue;
  local_signature.passing_modes.assign(
      local_signature.num_params,
      SemIR::ClangDeclSignature::PassingMode::ByValue);
  signature_id =
      context->clang_decl_signatures().Add(std::move(local_signature));
  signature = &context->clang_decl_signatures().Get(signature_id);
  has_simple_return_type =
      IsSimpleAbiType(context->ast_context(), function_type->getReturnType(),
                      /*for_parameter=*/false);
}

auto CalleeFunctionInfo::decl_name() const -> clang::DeclarationName {
  if (auto* decl = decl_or_pointer_type.dyn_cast<clang::FunctionDecl*>()) {
    return decl->getDeclName();
  }
  return &context->ast_context().Idents.get("__invoke");
}

auto CalleeFunctionInfo::clang_location() const -> clang::SourceLocation {
  if (auto* decl = decl_or_pointer_type.dyn_cast<clang::FunctionDecl*>()) {
    return decl->getLocation();
  }
  return {};
}

auto CalleeFunctionInfo::self_param_type() const -> clang::QualType {
  if (auto* decl = decl_or_pointer_type.dyn_cast<clang::FunctionDecl*>()) {
    if (IsObjectMemberFunction(*decl)) {
      const auto* method_decl = cast<clang::CXXMethodDecl>(decl);
      return method_decl->getFunctionObjectParameterReferenceType();
    } else {
      return {};
    }
  } else if (const auto* type =
                 decl_or_pointer_type.dyn_cast<const clang::Type*>()) {
    return clang::QualType(type, 0);
  }
  CARBON_FATAL("Unreachable");
}

auto CalleeFunctionInfo::effective_return_type() const -> clang::QualType {
  if (auto* decl = decl_or_pointer_type.dyn_cast<clang::FunctionDecl*>()) {
    if (isa<clang::CXXConstructorDecl>(decl)) {
      const auto* method_decl = cast<clang::CXXMethodDecl>(decl);
      return method_decl->getASTContext().getCanonicalTagType(
          method_decl->getParent());
    }
  }
  return function_type->getReturnType();
}

auto CalleeFunctionInfo::GetCalleeParamIdentifier(int i) const
    -> clang::IdentifierInfo* {
  if (auto* decl = decl_or_pointer_type.dyn_cast<clang::FunctionDecl*>()) {
    return decl->getParamDecl(i)->getIdentifier();
  }
  return nullptr;
}

auto CalleeFunctionInfo::GetCalleeParamLocation(int i) const
    -> clang::SourceLocation {
  if (auto* decl = decl_or_pointer_type.dyn_cast<clang::FunctionDecl*>()) {
    return decl->getParamDecl(i)->getLocation();
  }
  return {};
}

auto IsCppThunkRequired(Context& context, const CalleeFunctionInfo& callee_info)
    -> bool {
  auto* decl = callee_info.decl();
  if (callee_info.signature->kind != SemIR::ClangDeclSignature::Normal ||
      callee_info.signature->num_params !=
          static_cast<int>(decl->getNumNonObjectParams())) {
    // We require a thunk if the number of parameters we want isn't all of them.
    // This happens if default arguments are in use, or (eventually) when
    // calling a varargs function.
    return true;
  }

  if (!callee_info.has_simple_return_type) {
    return true;
  }

  auto& ast_context = context.ast_context();
  auto self_param_type = callee_info.self_param_type();
  if (!self_param_type.isNull() &&
      (!IsSimpleAbiType(ast_context, self_param_type,
                        /*for_parameter=*/true) ||
       callee_info.signature->self_passing_mode ==
           SemIR::ClangDeclSignature::PassingMode::ByVar)) {
    return true;
  }

  const auto* function_type =
      decl->getType()->castAs<clang::FunctionProtoType>();
  for (int i : llvm::seq(decl->getNumParams())) {
    if (!IsSimpleAbiType(ast_context, function_type->getParamType(i),
                         /*for_parameter=*/true) ||
        callee_info.signature->GetPassingMode(i) ==
            SemIR::ClangDeclSignature::PassingMode::ByVar) {
      return true;
    }
  }

  return false;
}

// Given a pointer type, returns the corresponding _Nonnull-qualified pointer
// type.
static auto GetNonnullType(clang::ASTContext& ast_context,
                           clang::QualType pointer_type) -> clang::QualType {
  return ast_context.getAttributedType(clang::NullabilityKind::NonNull,
                                       pointer_type, pointer_type);
}

// Given a type, returns the corresponding _Nonnull-qualified pointer type,
// ignoring references.
static auto GetNonNullablePointerType(clang::ASTContext& ast_context,
                                      clang::QualType type) {
  return GetNonnullType(ast_context,
                        ast_context.getPointerType(type.getNonReferenceType()));
}

// Given the type of a callee parameter, returns the type to use for the
// corresponding thunk parameter.
static auto GetThunkParameterType(clang::ASTContext& ast_context,
                                  clang::QualType callee_type)
    -> clang::QualType {
  if (IsSimpleAbiType(ast_context, callee_type, /*for_parameter=*/true)) {
    return callee_type;
  }
  return GetNonNullablePointerType(ast_context, callee_type);
}

// Creates the thunk parameter types given the callee function.
static auto BuildThunkParameterTypes(clang::ASTContext& ast_context,
                                     CalleeFunctionInfo callee_info)
    -> llvm::SmallVector<clang::QualType> {
  llvm::SmallVector<clang::QualType> thunk_param_types;
  thunk_param_types.reserve(callee_info.num_thunk_params());
  if (callee_info.callee_param_to_carbon_param_offset() > 0) {
    thunk_param_types.push_back(callee_info.self_param_type());
  }

  for (int i : llvm::seq(callee_info.num_callee_params)) {
    thunk_param_types.push_back(GetThunkParameterType(
        ast_context, callee_info.function_type->getParamType(i)));
  }

  if (!callee_info.has_simple_return_type) {
    thunk_param_types.push_back(GetNonNullablePointerType(
        ast_context, callee_info.effective_return_type()));
  }

  CARBON_CHECK(thunk_param_types.size() == callee_info.num_thunk_params());
  return thunk_param_types;
}

// Returns the thunk parameters using the callee function parameter identifiers.
static auto BuildThunkParameters(clang::ASTContext& ast_context,
                                 CalleeFunctionInfo callee_info,
                                 clang::SourceLocation clang_loc,
                                 clang::FunctionDecl* thunk_function_decl)
    -> llvm::SmallVector<clang::ParmVarDecl*> {
  const auto* thunk_function_proto_type =
      thunk_function_decl->getType()->castAs<clang::FunctionProtoType>();

  llvm::SmallVector<clang::ParmVarDecl*> thunk_params;
  unsigned num_thunk_params = thunk_function_decl->getNumParams();
  thunk_params.reserve(num_thunk_params);

  if (callee_info.callee_param_to_carbon_param_offset() > 0) {
    clang::ParmVarDecl* thunk_param =
        clang::ParmVarDecl::Create(ast_context, thunk_function_decl, clang_loc,
                                   clang_loc, &ast_context.Idents.get("this"),
                                   thunk_function_proto_type->getParamType(0),
                                   nullptr, clang::SC_None, nullptr);
    thunk_params.push_back(thunk_param);
  }

  for (int i : llvm::seq(callee_info.num_callee_params)) {
    clang::ParmVarDecl* thunk_param = clang::ParmVarDecl::Create(
        ast_context, thunk_function_decl, clang_loc, clang_loc,
        callee_info.GetCalleeParamIdentifier(i),
        thunk_function_proto_type->getParamType(
            i + callee_info.callee_param_to_carbon_param_offset()),
        nullptr, clang::SC_None, nullptr);
    thunk_params.push_back(thunk_param);
  }

  if (!callee_info.has_simple_return_type) {
    int thunk_return_index = callee_info.num_callee_params +
                             callee_info.callee_param_to_carbon_param_offset();
    clang::ParmVarDecl* thunk_param = clang::ParmVarDecl::Create(
        ast_context, thunk_function_decl, clang_loc, clang_loc,
        &ast_context.Idents.get("return"),
        thunk_function_proto_type->getParamType(thunk_return_index), nullptr,
        clang::SC_None, nullptr);
    thunk_params.push_back(thunk_param);
  }

  CARBON_CHECK(thunk_params.size() == num_thunk_params);
  return thunk_params;
}

// Computes a name to use for a thunk, based on the name of the thunk's target.
// The actual name used isn't critical, since it doesn't show up much except in
// AST dumps and SemIR output, but we try to produce a valid C++ identifier.
static auto GetDeclNameForThunk(clang::ASTContext& ast_context,
                                clang::DeclarationName name)
    -> clang::DeclarationName {
  llvm::SmallString<64> thunk_name;
  switch (name.getNameKind()) {
    case clang::DeclarationName::NameKind::Identifier: {
      thunk_name = name.getAsIdentifierInfo()->getName();
      break;
    }
    case clang::DeclarationName::NameKind::CXXOperatorName: {
      thunk_name = "operator_";
      switch (name.getCXXOverloadedOperator()) {
        case clang::OO_None:
        case clang::NUM_OVERLOADED_OPERATORS:
          break;
#define OVERLOADED_OPERATOR(Name, Spelling, Token, Unary, Binary, MemberOnly) \
  case clang::OO_##Name:                                                      \
    thunk_name += #Name;                                                      \
    break;
#include "clang/Basic/OperatorKinds.def"
      }
      break;
    }
    default: {
      break;
    }
  }
  if (auto type = name.getCXXNameType(); !type.isNull()) {
    if (auto* class_decl = type->getAsCXXRecordDecl()) {
      thunk_name += class_decl->getName();
    }
  }
  thunk_name += "__carbon_thunk";
  return &ast_context.Idents.get(thunk_name);
}

// Returns the thunk function declaration given the callee function and the
// thunk parameter types.
static auto CreateThunkFunctionDecl(
    Context& context, CalleeFunctionInfo callee_info,
    clang::SourceLocation clang_loc,
    llvm::ArrayRef<clang::QualType> thunk_param_types) -> clang::FunctionDecl* {
  clang::ASTContext& ast_context = context.ast_context();
  clang::DeclarationName name =
      GetDeclNameForThunk(ast_context, callee_info.decl_name());

  auto ext_proto_info = clang::FunctionProtoType::ExtProtoInfo();
  clang::QualType thunk_function_type = ast_context.getFunctionType(
      callee_info.has_simple_return_type ? callee_info.effective_return_type()
                                         : ast_context.VoidTy,
      thunk_param_types, ext_proto_info);

  clang::DeclContext* decl_context = ast_context.getTranslationUnitDecl();
  clang::FunctionDecl* thunk_function_decl = clang::FunctionDecl::Create(
      ast_context, decl_context, clang_loc, clang_loc, name,
      thunk_function_type, /*TInfo=*/nullptr, clang::SC_None,
      /*UsesFPIntrin=*/false, /*isInlineSpecified=*/true);
  decl_context->addDecl(thunk_function_decl);

  thunk_function_decl->setParams(BuildThunkParameters(
      ast_context, callee_info, clang_loc, thunk_function_decl));

  // Force the thunk to be inlined and discarded.
  thunk_function_decl->addAttr(
      clang::AlwaysInlineAttr::CreateImplicit(ast_context));
  thunk_function_decl->addAttr(
      clang::InternalLinkageAttr::CreateImplicit(ast_context));

  // Set asm("<callee function mangled name>.carbon_thunk").
  thunk_function_decl->addAttr(clang::AsmLabelAttr::CreateImplicit(
      ast_context,
      GenerateThunkMangledName(context.cpp_context()->clang_mangle_context(),
                               callee_info.decl(), *callee_info.signature),
      clang_loc));

  // Set function declaration type source info.
  thunk_function_decl->setTypeSourceInfo(ast_context.getTrivialTypeSourceInfo(
      thunk_function_decl->getType(), clang_loc));

  return thunk_function_decl;
}

// Builds a reference to the given parameter thunk. If `type` is specified, that
// is the callee parameter type that's being held by the parameter, and
// conversions will be performed as necessary to recover a value of that type.
static auto BuildThunkParamRef(
    clang::Sema& sema, clang::FunctionDecl* thunk_function_decl,
    unsigned thunk_index, SemIR::ClangDeclSignature::PassingMode passing_mode,
    clang::QualType type = clang::QualType()) -> clang::Expr* {
  clang::ParmVarDecl* thunk_param =
      thunk_function_decl->getParamDecl(thunk_index);
  clang::SourceLocation clang_loc = thunk_param->getLocation();

  clang::Expr* call_arg = sema.BuildDeclRefExpr(
      thunk_param, thunk_param->getType().getNonReferenceType(),
      clang::VK_LValue, clang_loc);
  if (!type.isNull() && thunk_param->getType() != type) {
    clang::ExprResult deref_result =
        sema.BuildUnaryOp(nullptr, clang_loc, clang::UO_Deref, call_arg);
    CARBON_CHECK(deref_result.isUsable());
    call_arg = deref_result.get();
  }

  // Cast to an xvalue when using pass-by-`var` or when initializing an rvalue
  // reference (which might be passed by value if it's const-qualified).
  if (passing_mode == SemIR::ClangDeclSignature::PassingMode::ByVar ||
      (!type.isNull() && type->isRValueReferenceType())) {
    call_arg = clang::ImplicitCastExpr::Create(
        sema.getASTContext(), call_arg->getType(), clang::CK_NoOp, call_arg,
        nullptr, clang::ExprValueKind::VK_XValue, clang::FPOptionsOverride());
  }
  return call_arg;
}

// Builds a reference to the parameter thunk parameter corresponding to the
// given callee parameter index.
static auto BuildParamRefForCalleeArg(clang::Sema& sema,
                                      clang::FunctionDecl* thunk_function_decl,
                                      CalleeFunctionInfo callee_info,
                                      unsigned callee_index) -> clang::Expr* {
  unsigned thunk_index =
      callee_index + callee_info.callee_param_to_carbon_param_offset();
  return BuildThunkParamRef(
      sema, thunk_function_decl, thunk_index,
      callee_info.signature->GetPassingMode(callee_index),
      callee_info.function_type->getParamType(callee_index));
}

// Builds an argument list for the callee function by creating suitable uses of
// the corresponding thunk parameters.
static auto BuildCalleeArgs(clang::Sema& sema,
                            clang::FunctionDecl* thunk_function_decl,
                            CalleeFunctionInfo callee_info)
    -> llvm::SmallVector<clang::Expr*> {
  llvm::SmallVector<clang::Expr*> call_args;
  call_args.reserve(callee_info.num_callee_params -
                    callee_info.callee_arg_to_callee_param_offset());
  for (unsigned callee_index :
       llvm::seq(callee_info.callee_arg_to_callee_param_offset(),
                 callee_info.num_callee_params)) {
    call_args.push_back(BuildParamRefForCalleeArg(sema, thunk_function_decl,
                                                  callee_info, callee_index));
  }
  return call_args;
}

// Builds the thunk function body which calls the callee function using the call
// args and returns the callee function return value. Returns nullptr on
// failure.
static auto BuildThunkBody(CppContext& cpp_context, clang::Sema& sema,
                           clang::SourceLocation clang_loc,
                           clang::FunctionDecl* thunk_function_decl,
                           CalleeFunctionInfo callee_info)
    -> clang::StmtResult {
  // TODO: Consider building a CompoundStmt holding our created statement to
  // make our result more closely resemble a real C++ function.

  auto* callee_decl = callee_info.decl();
  // If the callee has an object parameter, build a member access expression as
  // the callee. Otherwise, build a regular reference to the function.
  clang::ExprResult callee;
  switch (callee_info.self_param_kind) {
    case CalleeFunctionInfo::SelfParamKind::ExplicitObjectParam:
    case CalleeFunctionInfo::SelfParamKind::ImplicitObjectParam: {
      clang::QualType object_param_type =
          cast<clang::CXXMethodDecl>(callee_decl)
              ->getFunctionObjectParameterReferenceType();
      auto* object_param_ref = BuildThunkParamRef(
          sema, thunk_function_decl, /*thunk_index=*/0,
          callee_info.signature->self_passing_mode, object_param_type);
      constexpr bool IsArrow = false;
      auto object =
          sema.PerformMemberExprBaseConversion(object_param_ref, IsArrow);
      if (object.isInvalid()) {
        return clang::StmtError();
      }
      callee = sema.BuildMemberExpr(
          object.get(), IsArrow, clang_loc, clang::NestedNameSpecifierLoc(),
          clang::SourceLocation(), callee_decl,
          clang::DeclAccessPair::make(callee_decl, clang::AS_public),
          /*HadMultipleCandidates=*/false,
          clang::DeclarationNameInfo(callee_decl->getDeclName(), clang_loc),
          sema.getASTContext().BoundMemberTy, clang::VK_PRValue,
          clang::OK_Ordinary);
      break;
    }
    case CalleeFunctionInfo::SelfParamKind::FunctionPointer:
      callee = BuildThunkParamRef(sema, thunk_function_decl, 0,
                                  callee_info.signature->self_passing_mode,
                                  callee_info.self_param_type());
      break;
    case CalleeFunctionInfo::SelfParamKind::None: {
      if (isa<clang::CXXConstructorDecl>(callee_decl)) {
        break;
      }
      callee = sema.BuildDeclRefExpr(callee_decl, callee_decl->getType(),
                                     clang::VK_PRValue, clang_loc);
      break;
    }
  }

  if (callee.isInvalid()) {
    return clang::StmtError();
  }

  // Build the argument list.
  llvm::SmallVector<clang::Expr*> call_args =
      BuildCalleeArgs(sema, thunk_function_decl, callee_info);

  clang::ExprResult call;
  if (auto info = callee_decl ? clang::getConstructorInfo(callee_decl)
                              : clang::ConstructorInfo{};
      info.Constructor) {
    // In C++, there are no direct calls to constructors, only initialization,
    // so we need to type-check and build the call ourselves.
    auto type = sema.Context.getCanonicalTagType(
        cast<clang::CXXRecordDecl>(callee_decl->getParent()));
    llvm::SmallVector<clang::Expr*> converted_args;
    converted_args.reserve(call_args.size());
    if (sema.CompleteConstructorCall(info.Constructor, type, call_args,
                                     clang_loc, converted_args)) {
      return clang::StmtError();
    }
    call = sema.BuildCXXConstructExpr(
        clang_loc, type, callee_decl, info.Constructor, converted_args, false,
        false, false, false, clang::CXXConstructionKind::Complete, clang_loc);
  } else {
    call = sema.BuildCallExpr(nullptr, callee.get(), clang_loc, call_args,
                              clang_loc);
  }
  if (!call.isUsable()) {
    return clang::StmtError();
  }

  if (callee_info.has_simple_return_type) {
    return sema.BuildReturnStmt(clang_loc, call.get());
  }

  int return_thunk_index = callee_info.num_callee_params +
                           callee_info.callee_param_to_carbon_param_offset();
  auto* return_object_addr =
      BuildThunkParamRef(sema, thunk_function_decl, return_thunk_index,
                         SemIR::ClangDeclSignature::PassingMode::ByValue);
  auto return_type = callee_info.effective_return_type().getNonReferenceType();
  auto* return_type_info =
      sema.Context.getTrivialTypeSourceInfo(return_type, clang_loc);

  auto* placement_new_decl = cpp_context.placement_new_decl();
  if (!placement_new_decl) {
    placement_new_decl = GeneratePlacementNewFunctionDecl(sema.getASTContext());
    cpp_context.set_placement_new_decl(placement_new_decl);
  }
  sema.MarkFunctionReferenced(clang_loc, placement_new_decl);
  clang::ImplicitAllocationParameters params(return_type,
                                             clang::TypeAwareAllocationMode::No,
                                             clang::AlignedAllocationMode::No);
  clang::SourceRange range(clang_loc, clang_loc);
  auto* placement_new = clang::CXXNewExpr::Create(
      sema.getASTContext(), /*IsGlobalNew*/ true, placement_new_decl,
      /*OperatorDelete*/ nullptr, params, /*UsualArrayDeleteWantsSize*/ false,
      {return_object_addr},
      /*TypeIdParens=*/clang::SourceRange(), /*ArraySize=*/std::nullopt,
      clang::CXXNewInitializationStyle::Parens, call.get(),
      sema.getASTContext().getPointerType(return_type), return_type_info, range,
      range);
  return sema.ActOnExprStmt(placement_new, /*DiscardedValue=*/true);
}

auto BuildCppThunk(Context& context, const CalleeFunctionInfo& callee_info)
    -> clang::FunctionDecl* {
  // Build the thunk function declaration.
  auto thunk_param_types =
      BuildThunkParameterTypes(context.ast_context(), callee_info);
  auto clang_loc = callee_info.clang_location();
  clang::FunctionDecl* thunk_function_decl = CreateThunkFunctionDecl(
      context, callee_info, clang_loc, thunk_param_types);

  // Build the thunk function body.
  clang::Sema& sema = context.clang_sema();
  clang::Sema::ContextRAII context_raii(sema, thunk_function_decl);
  sema.ActOnStartOfFunctionDef(nullptr, thunk_function_decl);
  clang::StmtResult body =
      BuildThunkBody(*context.cpp_context(), sema, clang_loc,
                     thunk_function_decl, callee_info);
  sema.ActOnFinishFunctionBody(thunk_function_decl, body.get());
  if (body.isInvalid()) {
    return nullptr;
  }

  context.clang_sema().getASTConsumer().HandleTopLevelDecl(
      clang::DeclGroupRef(thunk_function_decl));
  return thunk_function_decl;
}

auto PerformCppThunkCall(Context& context, SemIR::LocId loc_id,
                         SemIR::FunctionId callee_function_id,
                         llvm::ArrayRef<SemIR::InstId> callee_arg_ids,
                         SemIR::InstId thunk_callee_id) -> SemIR::InstId {
  auto& callee_function = context.functions().Get(callee_function_id);
  auto callee_function_params =
      context.inst_blocks().Get(callee_function.call_params_id);
  auto num_callee_return_params =
      callee_function.call_param_ranges.return_size();

  auto thunk_callee = GetCalleeAsFunction(context.sem_ir(), thunk_callee_id);
  auto& thunk_function = context.functions().Get(thunk_callee.function_id);
  auto thunk_function_params =
      context.inst_blocks().Get(thunk_function.call_params_id);
  auto num_thunk_return_params = thunk_function.call_param_ranges.return_size();

  CARBON_CHECK(
      num_callee_return_params <= 1 && num_thunk_return_params <= 1,
      "TODO: generalize this logic to support multiple return patterns.");

  // Whether we need to pass a return address to the thunk as a final argument.
  bool thunk_takes_return_address =
      num_callee_return_params > 0 && num_thunk_return_params == 0;

  // The number of arguments we should be acquiring in order to call the thunk.
  // This includes the return address parameters, if any.
  unsigned num_thunk_args =
      context.inst_blocks().Get(thunk_function.param_patterns_id).size();

  // The corresponding number of arguments that would be provided in a syntactic
  // call to the callee. This excludes the return slot.
  unsigned num_callee_args = num_thunk_args - thunk_takes_return_address;

  // Grab the return slot argument, if we were given one.
  auto return_slot_id = SemIR::InstId::None;
  if (callee_arg_ids.size() == num_callee_args + 1) {
    return_slot_id = callee_arg_ids.consume_back();
  }

  // If there are return slot patterns, drop the corresponding parameters.
  // TODO: The parameter should probably only be created if the return pattern
  // actually needs a return address to be passed in.
  thunk_function_params =
      thunk_function_params.drop_back(num_thunk_return_params);
  callee_function_params =
      callee_function_params.drop_back(num_callee_return_params);

  // We assume that the call parameters exactly match the parameter patterns for
  // both the thunk and the callee. This is guaranteed even when we generate a
  // tuple pattern wrapping the function parameters.
  CARBON_CHECK(num_callee_args == callee_function_params.size(), "{0} != {1}",
               num_callee_args, callee_function_params.size());
  CARBON_CHECK(num_callee_args == callee_arg_ids.size());
  CARBON_CHECK(num_thunk_args == thunk_function_params.size());

  // Build the thunk arguments by converting the callee arguments as needed.
  llvm::SmallVector<SemIR::InstId> thunk_arg_ids;
  thunk_arg_ids.reserve(num_thunk_args);
  for (auto [callee_param_inst_id, thunk_param_inst_id, callee_arg_id] :
       llvm::zip(callee_function_params, thunk_function_params,
                 callee_arg_ids)) {
    SemIR::TypeId callee_param_type_id =
        context.insts().GetAs<SemIR::AnyParam>(callee_param_inst_id).type_id;
    SemIR::TypeId thunk_param_type_id =
        context.insts().GetAs<SemIR::AnyParam>(thunk_param_inst_id).type_id;

    SemIR::InstId arg_id = callee_arg_id;
    if (callee_param_type_id != thunk_param_type_id) {
      arg_id = Convert(context, loc_id, arg_id,
                       {.kind = ConversionTarget::CppThunkRef,
                        .type_id = callee_param_type_id});
      arg_id = AddInst<SemIR::AddrOf>(
          context, loc_id,
          {.type_id = GetPointerType(
               context, context.types().GetTypeInstId(callee_param_type_id)),
           .lvalue_id = arg_id});
      arg_id =
          ConvertToValueOfType(context, loc_id, arg_id, thunk_param_type_id);
    }
    thunk_arg_ids.push_back(arg_id);
  }

  // Add an argument to hold the result of the call, if necessary.
  auto return_type_id = callee_function.GetDeclaredReturnType(context.sem_ir());
  if (thunk_takes_return_address) {
    // Create a temporary if the caller didn't provide a return slot.
    if (!return_slot_id.has_value()) {
      return_slot_id = AddInst<SemIR::TemporaryStorage>(
          context, loc_id, {.type_id = return_type_id});
    }

    auto arg_id = AddInst<SemIR::AddrOf>(
        context, loc_id,
        {.type_id = GetPointerType(
             context, context.types().GetTypeInstId(
                          context.insts().Get(return_slot_id).type_id())),
         .lvalue_id = return_slot_id});
    thunk_arg_ids.push_back(arg_id);
  } else if (return_slot_id.has_value()) {
    thunk_arg_ids.push_back(return_slot_id);
  }

  // Compute the return type of the call to the thunk.
  auto thunk_return_type_id =
      thunk_function.GetDeclaredReturnType(context.sem_ir());
  if (!thunk_return_type_id.has_value()) {
    CARBON_CHECK(thunk_takes_return_address || !return_type_id.has_value());
    thunk_return_type_id = GetTupleType(context, {});
  } else {
    CARBON_CHECK(thunk_return_type_id == return_type_id);
  }

  auto result_id = GetOrAddInst<SemIR::Call>(
      context, loc_id,
      {.type_id = thunk_return_type_id,
       .callee_id = thunk_callee_id,
       .args_id = context.inst_blocks().Add(thunk_arg_ids)});

  // Produce the result of the call, taking the value from the return storage.
  if (thunk_takes_return_address) {
    result_id = AddInst<SemIR::MarkInPlaceInit>(context, loc_id,
                                                {.type_id = return_type_id,
                                                 .src_id = result_id,
                                                 .dest_id = return_slot_id});
  }

  return result_id;
}

}  // namespace Carbon::Check
