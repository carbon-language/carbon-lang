// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/thunk.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
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

// Returns the GlobalDecl to use to represent the given function declaration.
// TODO: Refactor with `Lower::CreateGlobalDecl`.
static auto GetGlobalDecl(const clang::FunctionDecl* decl)
    -> clang::GlobalDecl {
  if (const auto* ctor = dyn_cast<clang::CXXConstructorDecl>(decl)) {
    return clang::GlobalDecl(ctor, clang::CXXCtorType::Ctor_Complete);
  }
  return clang::GlobalDecl(decl);
}

// Returns the C++ thunk mangled name given the callee function.
static auto GenerateThunkMangledName(
    clang::MangleContext& mangle_context,
    const clang::FunctionDecl& callee_function_decl) -> std::string {
  RawStringOstream mangled_name_stream;
  mangle_context.mangleName(GetGlobalDecl(&callee_function_decl),
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

      auto int_info = context.types().TryGetIntTypeInfo(type_id);
      if (!int_info || !int_info->bit_width.has_value()) {
        return true;
      }

      llvm::APInt bit_width = context.ints().Get(int_info->bit_width);
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

  // A thunk is required if any parameter or return type requires it. However,
  // we don't generate a thunk if any relevant type is erroneous.
  bool thunk_required = false;

  // We require a thunk if any parameter is of reference type, even if the
  // corresponding SemIR function has an acceptable parameter type.
  // TODO: We should be able to avoid thunks for reference parameters.
  const auto* decl = cast<clang::FunctionDecl>(
      context.clang_decls().Get(function.clang_decl_id).decl);
  for (auto* param : decl->parameters()) {
    if (param->getType()->isReferenceType()) {
      thunk_required = true;
    }
  }

  SemIR::TypeId return_type_id =
      function.GetDeclaredReturnType(context.sem_ir());
  if (return_type_id.has_value()) {
    if (return_type_id == SemIR::ErrorInst::TypeId) {
      return false;
    }
    thunk_required = IsThunkRequiredForType(context, return_type_id);
  }

  for (auto param_id :
       context.inst_blocks().GetOrEmpty(function.call_params_id)) {
    if (param_id == SemIR::ErrorInst::InstId) {
      return false;
    }
    if (!thunk_required &&
        IsThunkRequiredForType(
            context,
            context.insts().GetAs<SemIR::AnyParam>(param_id).type_id)) {
      thunk_required = true;
    }
  }

  return thunk_required;
}

// Returns whether the type is void, a pointer, or a signed int of 32 or 64
// bits.
static auto IsSimpleAbiType(clang::ASTContext& ast_context,
                            clang::QualType type) -> bool {
  if (type->isVoidType() || type->isPointerType()) {
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

namespace {
// Information about the callee of a thunk.
struct CalleeFunctionInfo {
  explicit CalleeFunctionInfo(clang::FunctionDecl* decl) : decl(decl) {
    auto& ast_context = decl->getASTContext();
    const auto* method_decl = dyn_cast<clang::CXXMethodDecl>(decl);
    bool is_ctor = isa<clang::CXXConstructorDecl>(decl);
    has_object_parameter = method_decl && !method_decl->isStatic() && !is_ctor;
    if (has_object_parameter && method_decl->isImplicitObjectMemberFunction()) {
      implicit_this_type = method_decl->getThisType();
    }
    effective_return_type =
        is_ctor ? ast_context.getCanonicalTagType(method_decl->getParent())
                : decl->getReturnType();
    has_simple_return_type =
        IsSimpleAbiType(ast_context, effective_return_type);
  }

  // Returns whether this callee has an implicit `this` parameter.
  auto has_implicit_object_parameter() const -> bool {
    return !implicit_this_type.isNull();
  }

  // Returns whether this callee has an explicit `this` parameter.
  auto has_explicit_object_parameter() const -> bool {
    return has_object_parameter && !has_implicit_object_parameter();
  }

  // Returns the number of parameters the thunk should have.
  auto num_thunk_params() const -> unsigned {
    return has_implicit_object_parameter() + decl->getNumParams() +
           !has_simple_return_type;
  }

  // Returns the thunk parameter index corresponding to a given callee parameter
  // index.
  auto GetThunkParamIndex(unsigned callee_param_index) const -> unsigned {
    return has_implicit_object_parameter() + callee_param_index;
  }

  // Returns the thunk parameter index corresponding to the parameter that holds
  // the address of the return value.
  auto GetThunkReturnParamIndex() const -> unsigned {
    CARBON_CHECK(!has_simple_return_type);
    return has_implicit_object_parameter() + decl->getNumParams();
  }

  // The callee function.
  clang::FunctionDecl* decl;

  // Whether the callee has an object parameter, which might be explicit or
  // implicit.
  bool has_object_parameter;

  // If the callee has an implicit object parameter, the corresponding `this`
  // type. Otherwise a null type.
  clang::QualType implicit_this_type;

  // The return type that the callee has when viewed from Carbon. This is the
  // C++ return type, except that constructors return the class type in Carbon
  // and return void in Clang's AST.
  clang::QualType effective_return_type;

  // Whether the callee has a simple return type, that we can return directly.
  // If not, we'll return through an out parameter instead.
  bool has_simple_return_type;
};
}  // namespace

// Given a pointer type, returns the corresponding _Nonnull-qualified pointer
// type.
static auto GetNonnullType(clang::ASTContext& ast_context,
                           clang::QualType pointer_type) -> clang::QualType {
  return ast_context.getAttributedType(clang::NullabilityKind::NonNull,
                                       pointer_type, pointer_type);
}

// Given the type of a callee parameter, returns the type to use for the
// corresponding thunk parameter.
static auto GetThunkParameterType(clang::ASTContext& ast_context,
                                  clang::QualType callee_type)
    -> clang::QualType {
  if (IsSimpleAbiType(ast_context, callee_type)) {
    return callee_type;
  }
  return GetNonnullType(ast_context, ast_context.getPointerType(
                                         callee_type.getNonReferenceType()));
}

// Creates the thunk parameter types given the callee function.
static auto BuildThunkParameterTypes(clang::ASTContext& ast_context,
                                     CalleeFunctionInfo callee_info)
    -> llvm::SmallVector<clang::QualType> {
  llvm::SmallVector<clang::QualType> thunk_param_types;
  thunk_param_types.reserve(callee_info.num_thunk_params());
  if (callee_info.has_implicit_object_parameter()) {
    thunk_param_types.push_back(
        GetNonnullType(ast_context, callee_info.implicit_this_type));
  }

  for (const clang::ParmVarDecl* callee_param :
       callee_info.decl->parameters()) {
    // TODO: We should use the type from the function signature, not the type of
    // the parameter here.
    thunk_param_types.push_back(
        GetThunkParameterType(ast_context, callee_param->getType()));
  }

  if (!callee_info.has_simple_return_type) {
    thunk_param_types.push_back(GetNonnullType(
        ast_context,
        ast_context.getPointerType(callee_info.effective_return_type)));
  }

  CARBON_CHECK(thunk_param_types.size() == callee_info.num_thunk_params());
  return thunk_param_types;
}

// Returns the thunk parameters using the callee function parameter identifiers.
static auto BuildThunkParameters(clang::ASTContext& ast_context,
                                 CalleeFunctionInfo callee_info,
                                 clang::FunctionDecl* thunk_function_decl)
    -> llvm::SmallVector<clang::ParmVarDecl*> {
  clang::SourceLocation clang_loc = callee_info.decl->getLocation();

  const auto* thunk_function_proto_type =
      thunk_function_decl->getFunctionType()->getAs<clang::FunctionProtoType>();

  llvm::SmallVector<clang::ParmVarDecl*> thunk_params;
  unsigned num_thunk_params = thunk_function_decl->getNumParams();
  thunk_params.reserve(num_thunk_params);

  if (callee_info.has_implicit_object_parameter()) {
    clang::ParmVarDecl* thunk_param =
        clang::ParmVarDecl::Create(ast_context, thunk_function_decl, clang_loc,
                                   clang_loc, &ast_context.Idents.get("this"),
                                   thunk_function_proto_type->getParamType(0),
                                   nullptr, clang::SC_None, nullptr);
    thunk_params.push_back(thunk_param);
  }

  for (unsigned i : llvm::seq(callee_info.decl->getNumParams())) {
    clang::ParmVarDecl* thunk_param = clang::ParmVarDecl::Create(
        ast_context, thunk_function_decl, clang_loc, clang_loc,
        callee_info.decl->getParamDecl(i)->getIdentifier(),
        thunk_function_proto_type->getParamType(
            callee_info.GetThunkParamIndex(i)),
        nullptr, clang::SC_None, nullptr);
    thunk_params.push_back(thunk_param);
  }

  if (!callee_info.has_simple_return_type) {
    clang::ParmVarDecl* thunk_param =
        clang::ParmVarDecl::Create(ast_context, thunk_function_decl, clang_loc,
                                   clang_loc, &ast_context.Idents.get("return"),
                                   thunk_function_proto_type->getParamType(
                                       callee_info.GetThunkReturnParamIndex()),
                                   nullptr, clang::SC_None, nullptr);
    thunk_params.push_back(thunk_param);
  }

  CARBON_CHECK(thunk_params.size() == num_thunk_params);
  return thunk_params;
}

// Returns the thunk function declaration given the callee function and the
// thunk parameter types.
static auto CreateThunkFunctionDecl(
    Context& context, CalleeFunctionInfo callee_info,
    llvm::ArrayRef<clang::QualType> thunk_param_types) -> clang::FunctionDecl* {
  clang::ASTContext& ast_context = context.ast_context();
  clang::SourceLocation clang_loc = callee_info.decl->getLocation();

  clang::IdentifierInfo& identifier_info = ast_context.Idents.get(
      callee_info.decl->getNameAsString() + "__carbon_thunk");

  auto ext_proto_info = clang::FunctionProtoType::ExtProtoInfo();
  clang::QualType thunk_function_type = ast_context.getFunctionType(
      callee_info.has_simple_return_type ? callee_info.effective_return_type
                                         : ast_context.VoidTy,
      thunk_param_types, ext_proto_info);

  clang::DeclContext* decl_context = ast_context.getTranslationUnitDecl();
  // TODO: Thunks should not have external linkage, consider using `SC_Static`.
  clang::FunctionDecl* thunk_function_decl = clang::FunctionDecl::Create(
      ast_context, decl_context, clang_loc, clang_loc,
      clang::DeclarationName(&identifier_info), thunk_function_type,
      /*TInfo=*/nullptr, clang::SC_Extern);
  decl_context->addDecl(thunk_function_decl);

  thunk_function_decl->setParams(
      BuildThunkParameters(ast_context, callee_info, thunk_function_decl));

  // Set always_inline.
  thunk_function_decl->addAttr(
      clang::AlwaysInlineAttr::CreateImplicit(ast_context));

  // Set asm("<callee function mangled name>.carbon_thunk").
  thunk_function_decl->addAttr(clang::AsmLabelAttr::CreateImplicit(
      ast_context,
      GenerateThunkMangledName(*context.sem_ir().clang_mangle_context(),
                               *callee_info.decl),
      clang_loc));

  // Set function declaration type source info.
  thunk_function_decl->setTypeSourceInfo(ast_context.getTrivialTypeSourceInfo(
      thunk_function_decl->getType(), clang_loc));

  return thunk_function_decl;
}

// Builds a reference to the given parameter thunk. If `type` is specified, that
// is the callee parameter type that's being held by the parameter, and
// conversions will be performed as necessary to recover a value of that type.
static auto BuildThunkParamRef(clang::Sema& sema,
                               clang::FunctionDecl* thunk_function_decl,
                               unsigned thunk_index,
                               clang::QualType type = clang::QualType())
    -> clang::Expr* {
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

    // Cast to an rvalue when initializing an rvalue reference. The validity of
    // the initialization of the reference should be validated by the caller of
    // the thunk.
    //
    // TODO: Consider inserting a cast to an rvalue in more cases. Note that we
    // currently pass pointers to non-temporary objects as the argument when
    // calling a thunk, so we'll need to either change that or generate
    // different thunks depending on whether we're moving from each parameter.
    if (type->isRValueReferenceType()) {
      deref_result = clang::ImplicitCastExpr::Create(
          sema.getASTContext(), deref_result.get()->getType(), clang::CK_NoOp,
          deref_result.get(), nullptr, clang::ExprValueKind::VK_XValue,
          clang::FPOptionsOverride());
    }
    call_arg = deref_result.get();
  }
  return call_arg;
}

// Builds a reference to the parameter thunk parameter corresponding to the
// given callee parameter index.
static auto BuildParamRefForCalleeArg(clang::Sema& sema,
                                      clang::FunctionDecl* thunk_function_decl,
                                      CalleeFunctionInfo callee_info,
                                      unsigned callee_index) -> clang::Expr* {
  unsigned thunk_index = callee_info.GetThunkParamIndex(callee_index);
  return BuildThunkParamRef(
      sema, thunk_function_decl, thunk_index,
      callee_info.decl->getParamDecl(callee_index)->getType());
}

// Builds an argument list for the callee function by creating suitable uses of
// the corresponding thunk parameters.
static auto BuildCalleeArgs(clang::Sema& sema,
                            clang::FunctionDecl* thunk_function_decl,
                            CalleeFunctionInfo callee_info)
    -> llvm::SmallVector<clang::Expr*> {
  llvm::SmallVector<clang::Expr*> call_args;
  // The object parameter is always passed as `self`, not in the callee argument
  // list, so the first argument corresponds to the second parameter if there is
  // an explicit object parameter and the first parameter otherwise.
  unsigned first_param = callee_info.has_explicit_object_parameter();
  unsigned num_params = callee_info.decl->getNumParams();
  call_args.reserve(num_params - first_param);
  for (unsigned callee_index : llvm::seq(first_param, num_params)) {
    call_args.push_back(BuildParamRefForCalleeArg(sema, thunk_function_decl,
                                                  callee_info, callee_index));
  }
  return call_args;
}

// Builds the thunk function body which calls the callee function using the call
// args and returns the callee function return value. Returns nullptr on
// failure.
static auto BuildThunkBody(clang::Sema& sema,
                           clang::FunctionDecl* thunk_function_decl,
                           CalleeFunctionInfo callee_info)
    -> clang::StmtResult {
  // TODO: Consider building a CompoundStmt holding our created statement to
  // make our result more closely resemble a real C++ function.

  clang::SourceLocation clang_loc = callee_info.decl->getLocation();

  // If the callee has an object parameter, build a member access expression as
  // the callee. Otherwise, build a regular reference to the function.
  clang::ExprResult callee;
  if (callee_info.has_object_parameter) {
    auto* object_param_ref =
        BuildThunkParamRef(sema, thunk_function_decl, 0,
                           callee_info.has_explicit_object_parameter()
                               ? callee_info.decl->getParamDecl(0)->getType()
                               : clang::QualType());
    bool is_arrow = callee_info.has_implicit_object_parameter();
    auto object =
        sema.PerformMemberExprBaseConversion(object_param_ref, is_arrow);
    if (object.isInvalid()) {
      return clang::StmtError();
    }
    callee = sema.BuildMemberExpr(
        object.get(), is_arrow, clang_loc, clang::NestedNameSpecifierLoc(),
        clang::SourceLocation(), callee_info.decl,
        clang::DeclAccessPair::make(callee_info.decl, clang::AS_public),
        /*HadMultipleCandidates=*/false, clang::DeclarationNameInfo(),
        sema.getASTContext().BoundMemberTy, clang::VK_PRValue,
        clang::OK_Ordinary);
  } else if (!isa<clang::CXXConstructorDecl>(callee_info.decl)) {
    callee =
        sema.BuildDeclRefExpr(callee_info.decl, callee_info.decl->getType(),
                              clang::VK_PRValue, clang_loc);
  }

  if (callee.isInvalid()) {
    return clang::StmtError();
  }

  // Build the argument list.
  llvm::SmallVector<clang::Expr*> call_args =
      BuildCalleeArgs(sema, thunk_function_decl, callee_info);

  clang::ExprResult call;
  if (auto info = clang::getConstructorInfo(callee_info.decl);
      info.Constructor) {
    // In C++, there are no direct calls to constructors, only initialization,
    // so we need to type-check and build the call ourselves.
    auto type = sema.Context.getCanonicalTagType(
        cast<clang::CXXRecordDecl>(callee_info.decl->getParent()));
    llvm::SmallVector<clang::Expr*> converted_args;
    converted_args.reserve(call_args.size());
    if (sema.CompleteConstructorCall(info.Constructor, type, call_args,
                                     clang_loc, converted_args)) {
      return clang::StmtError();
    }
    call = sema.BuildCXXConstructExpr(
        clang_loc, type, callee_info.decl, info.Constructor, converted_args,
        false, false, false, false, clang::CXXConstructionKind::Complete,
        clang_loc);
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

  auto* return_object_addr = BuildThunkParamRef(
      sema, thunk_function_decl, callee_info.GetThunkReturnParamIndex());
  auto return_type = callee_info.effective_return_type;
  auto* return_type_info =
      sema.Context.getTrivialTypeSourceInfo(return_type, clang_loc);
  auto placement_new = sema.BuildCXXNew(
      clang_loc, /*UseGlobal=*/true, clang_loc, {return_object_addr}, clang_loc,
      /*TypeIdParens=*/clang::SourceRange(), return_type, return_type_info,
      /*ArraySize=*/std::nullopt, clang_loc, call.get());
  return sema.ActOnExprStmt(placement_new, /*DiscardedValue=*/true);
}

auto BuildCppThunk(Context& context, const SemIR::Function& callee_function)
    -> clang::FunctionDecl* {
  clang::FunctionDecl* callee_function_decl =
      context.sem_ir()
          .clang_decls()
          .Get(callee_function.clang_decl_id)
          .decl->getAsFunction();
  CARBON_CHECK(callee_function_decl);

  CalleeFunctionInfo callee_info(callee_function_decl);

  // Build the thunk function declaration.
  auto thunk_param_types =
      BuildThunkParameterTypes(context.ast_context(), callee_info);
  clang::FunctionDecl* thunk_function_decl =
      CreateThunkFunctionDecl(context, callee_info, thunk_param_types);

  // Build the thunk function body.
  clang::Sema& sema = context.sem_ir().clang_ast_unit()->getSema();
  clang::Sema::ContextRAII context_raii(sema, thunk_function_decl);
  sema.ActOnStartOfFunctionDef(nullptr, thunk_function_decl);
  clang::StmtResult body =
      BuildThunkBody(sema, thunk_function_decl, callee_info);
  sema.ActOnFinishFunctionBody(thunk_function_decl, body.get());
  if (body.isInvalid()) {
    return nullptr;
  }

  return thunk_function_decl;
}

auto PerformCppThunkCall(Context& context, SemIR::LocId loc_id,
                         SemIR::FunctionId callee_function_id,
                         llvm::ArrayRef<SemIR::InstId> callee_arg_ids,
                         SemIR::InstId thunk_callee_id) -> SemIR::InstId {
  auto& callee_function = context.functions().Get(callee_function_id);
  auto callee_function_params =
      context.inst_blocks().Get(callee_function.call_params_id);

  auto thunk_callee = GetCalleeFunction(context.sem_ir(), thunk_callee_id);
  auto& thunk_function = context.functions().Get(thunk_callee.function_id);
  auto thunk_function_params =
      context.inst_blocks().Get(thunk_function.call_params_id);

  // Whether we need to pass a return address to the thunk as a final argument.
  bool thunk_takes_return_address =
      callee_function.return_slot_pattern_id.has_value() &&
      !thunk_function.return_slot_pattern_id.has_value();

  // The number of arguments we should be acquiring in order to call the thunk.
  // This includes the return address parameter, if any.
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

  // If there's a return slot pattern, drop the corresponding parameter.
  // TODO: The parameter should probably only be created if the return pattern
  // actually needs a return address to be passed in.
  if (thunk_function.return_slot_pattern_id.has_value()) {
    thunk_function_params.consume_back();
  }
  if (callee_function.return_slot_pattern_id.has_value()) {
    callee_function_params.consume_back();
  }

  // We assume that the call parameters exactly match the parameter patterns for
  // both the thunk and the callee. This is currently guaranteed because we only
  // create trivial *ParamPatterns when importing a C++ function.
  CARBON_CHECK(num_callee_args == callee_function_params.size());
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
               context, context.types().GetInstId(callee_param_type_id)),
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
             context, context.types().GetInstId(
                          context.insts().Get(return_slot_id).type_id())),
         .lvalue_id = return_slot_id});
    thunk_arg_ids.push_back(arg_id);
  }

  auto result_id = PerformCall(context, loc_id, thunk_callee_id, thunk_arg_ids);

  // Produce the result of the call, taking the value from the return storage.
  if (thunk_takes_return_address) {
    result_id = AddInst<SemIR::InPlaceInit>(context, loc_id,
                                            {.type_id = return_type_id,
                                             .src_id = result_id,
                                             .dest_id = return_slot_id});
  }

  return result_id;
}

}  // namespace Carbon::Check
