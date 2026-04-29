// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/export.h"

#include "llvm/Support/Casting.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/type_mapping.h"
#include "toolchain/check/function.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/thunk.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/mangler.h"

namespace Carbon::Check {

// If the given name scope was produced by importing a C++ declaration or has
// already been exported to C++, return the corresponding Clang decl context.
static auto GetClangDeclContextForScope(Context& context,
                                        SemIR::NameScopeId scope_id)
    -> clang::DeclContext* {
  if (!scope_id.has_value()) {
    return nullptr;
  }
  auto& scope = context.name_scopes().Get(scope_id);
  auto clang_decl_context_id = scope.clang_decl_context_id();
  if (!clang_decl_context_id.has_value()) {
    return nullptr;
  }
  auto* decl = context.clang_decls().Get(clang_decl_context_id).key.decl;
  return cast<clang::DeclContext>(decl);
}

auto ExportNameScopeToCpp(Context& context, SemIR::LocId loc_id,
                          SemIR::NameScopeId name_scope_id)
    -> clang::DeclContext* {
  llvm::SmallVector<SemIR::NameScopeId> name_scope_ids_to_create;

  // Walk through the parent scopes, looking for one that's already mapped into
  // C++. We already mapped the package scope to ::Carbon, so we must find one.
  clang::DeclContext* decl_context = nullptr;
  while (true) {
    // If this name scope was produced by importing a C++ declaration or has
    // already been exported to C++, return the corresponding Clang declaration.
    if (auto* existing_decl_context =
            GetClangDeclContextForScope(context, name_scope_id)) {
      decl_context = existing_decl_context;
      break;
    }

    // Otherwise, continue to the parent and create a scope for it first.
    name_scope_ids_to_create.push_back(name_scope_id);
    name_scope_id = context.name_scopes().Get(name_scope_id).parent_scope_id();

    // TODO: What should happen if there's an intervening function scope?
    CARBON_CHECK(
        name_scope_id.has_value(),
        "Reached the top level without finding a scope mapped into C++");
  }

  // Create the name scopes in order, starting from the outermost one.
  while (!name_scope_ids_to_create.empty()) {
    name_scope_id = name_scope_ids_to_create.pop_back_val();

    auto& name_scope = context.name_scopes().Get(name_scope_id);
    auto* identifier_info =
        GetClangIdentifierInfo(context, name_scope.name_id());
    if (!identifier_info) {
      // TODO: Handle keyword package names like `Cpp` and `Core`. These can
      // be named from C++ via an alias.
      context.TODO(loc_id, "interop with non-identifier package name");
      return nullptr;
    }

    auto inst = context.insts().Get(name_scope.inst_id());
    if (inst.Is<SemIR::Namespace>()) {
      // TODO: Provide a source location.
      auto* namespace_decl = clang::NamespaceDecl::Create(
          context.ast_context(), decl_context, false, clang::SourceLocation(),
          clang::SourceLocation(), identifier_info, nullptr, false);
      decl_context->addHiddenDecl(namespace_decl);
      decl_context = namespace_decl;
    } else if (inst.Is<SemIR::ClassDecl>()) {
      // TODO: Provide a source location.
      auto* record_decl = clang::CXXRecordDecl::Create(
          context.ast_context(), clang::TagTypeKind::Class, decl_context,
          clang::SourceLocation(), clang::SourceLocation(), identifier_info);
      // If this is a member class, set its access.
      if (isa<clang::CXXRecordDecl>(decl_context)) {
        // TODO: Map Carbon access to C++ access.
        record_decl->setAccess(clang::AS_public);
      }

      decl_context->addHiddenDecl(record_decl);
      decl_context = record_decl;
      decl_context->setHasExternalLexicalStorage();
    } else {
      context.TODO(loc_id, "non-class non-namespace name scope");
      return nullptr;
    }

    decl_context->setHasExternalVisibleStorage();

    auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(
        cast<clang::Decl>(decl_context));
    auto clang_decl_id = context.clang_decls().Add(
        {.key = key, .inst_id = name_scope.inst_id()});
    name_scope.set_clang_decl_context_id(clang_decl_id, /*is_cpp_scope=*/false);
  }

  return decl_context;
}

auto ExportClassToCpp(Context& context, SemIR::LocId loc_id,
                      SemIR::InstId class_inst_id, SemIR::ClassType class_type)
    -> clang::TagDecl* {
  // TODO: A lot of logic in this function is shared with ExportNameScopeToCpp.
  // This should be refactored.

  if (class_type.specific_id.has_value()) {
    context.TODO(loc_id, "interop with specific class");
    return nullptr;
  }

  const auto& class_info = context.classes().Get(class_type.class_id);

  // If this class was produced by importing a C++ declaration or has
  // already been exported to C++, return the corresponding Clang declaration.
  // That could either be a CXXRecordDecl or an EnumDecl.
  if (auto* decl_context =
          GetClangDeclContextForScope(context, class_info.scope_id)) {
    return cast<clang::TagDecl>(decl_context);
  }

  auto* identifier_info = GetClangIdentifierInfo(context, class_info.name_id);
  CARBON_CHECK(identifier_info, "non-identifier class name {0}",
               class_info.name_id);

  auto* decl_context =
      ExportNameScopeToCpp(context, loc_id, class_info.parent_scope_id);
  // TODO: Provide a source location.
  auto* record_decl = clang::CXXRecordDecl::Create(
      context.ast_context(), clang::TagTypeKind::Class, decl_context,
      clang::SourceLocation(), clang::SourceLocation(), identifier_info);
  // If this is a member class, set its access.
  if (isa<clang::CXXRecordDecl>(decl_context)) {
    // TODO: Map Carbon access to C++ access.
    record_decl->setAccess(clang::AS_public);
  }

  record_decl->setHasExternalLexicalStorage();
  record_decl->setHasExternalVisibleStorage();

  auto key =
      SemIR::ClangDeclKey::ForNonFunctionDecl(cast<clang::Decl>(record_decl));
  auto clang_decl_id =
      context.clang_decls().Add({.key = key, .inst_id = class_inst_id});
  if (class_info.scope_id.has_value()) {
    // TODO: Record the Carbon class -> clang declaration mapping for incomplete
    // classes too.
    context.name_scopes()
        .Get(class_info.scope_id)
        .set_clang_decl_context_id(clang_decl_id, /*is_cpp_scope=*/false);
  }
  return record_decl;
}

namespace {
struct FunctionInfo {
  struct Param {
    // Type of the parameter's scrutinee.
    SemIR::TypeId type_id;

    // Whether this is a `ref` param.
    bool is_ref;
  };

  explicit FunctionInfo(Context& context, SemIR::FunctionId function_id,
                        const SemIR::Function& function,
                        clang::DeclContext* decl_context)
      : function_id(function_id),
        function(function),
        decl_context(decl_context) {
    auto function_params =
        context.inst_blocks().Get(function.call_param_patterns_id);

    // Get the function's `self` parameter type, if present.
    if (function.call_param_ranges.implicit_size() > 0) {
      CARBON_CHECK(function.call_param_ranges.implicit_size() == 1);

      auto param_inst_id =
          function_params[function.call_param_ranges.implicit_begin().index];
      auto scrutinee_type_id = ExtractScrutineeType(
          context.sem_ir(), context.insts().Get(param_inst_id).type_id());
      self_type_id = scrutinee_type_id;
    }

    // Get the function's explicit parameters.
    function_params =
        function_params.drop_front(function.call_param_ranges.implicit_size());
    function_params =
        function_params.drop_back(function.call_param_ranges.return_size());
    for (auto param_inst_id : function_params) {
      explicit_params.push_back(
          {.type_id = ExtractScrutineeType(
               context.sem_ir(), context.insts().Get(param_inst_id).type_id()),
           .is_ref =
               context.insts().Is<SemIR::RefParamPattern>(param_inst_id)});
    }
  }

  // Get the `StorageClass` to use for `CXXMethodDecl`s.
  auto GetStorageClass() const -> clang::StorageClass {
    if (has_self()) {
      return clang::SC_None;
    } else {
      return clang::SC_Static;
    }
  }

  // Whether the function has a `self` parameter.
  auto has_self() const -> bool { return self_type_id != SemIR::TypeId::None; }

  SemIR::FunctionId function_id;
  const SemIR::Function& function;

  // Parent scope in the C++ AST where a C++ thunk for this function can
  // be created. If the function is a method, this will be a
  // `CXXRecordDecl`.
  clang::DeclContext* decl_context;

  // For each of the function's explicit parameters, the scrutinee type
  // and whether the parameter is a reference.
  llvm::SmallVector<Param> explicit_params;

  // Type of the function's `self` parameter, or `None` if the function
  // is not a method.
  SemIR::TypeId self_type_id = SemIR::TypeId::None;
};
}  // namespace

// Create a `clang::FunctionDecl` for the given Carbon function. This
// can be used to call the Carbon function from C++. The Carbon
// function's ABI must be compatible with C++.
//
// The resulting decl is used to allow a generated C++ function to call
// a generated Carbon function.
static auto BuildCppFunctionDeclForCarbonFn(Context& context,
                                            SemIR::LocId loc_id,
                                            SemIR::FunctionId function_id)
    -> clang::FunctionDecl* {
  auto clang_loc = GetCppLocation(context, loc_id);

  const SemIR::Function& function = context.functions().Get(function_id);
  FunctionInfo callee(context, function_id, function, nullptr);

  // Get parameters types.
  llvm::SmallVector<clang::QualType> cpp_param_types;
  if (callee.has_self()) {
    auto cpp_type = MapToCppType(context, callee.self_type_id);
    if (cpp_type.isNull()) {
      context.TODO(loc_id, "failed to map Carbon self type to C++");
      return nullptr;
    }
    cpp_type = context.ast_context().getLValueReferenceType(cpp_type);
    cpp_param_types.push_back(cpp_type);
  }
  for (auto param : callee.explicit_params) {
    auto cpp_type = MapToCppType(context, param.type_id);
    if (cpp_type.isNull()) {
      context.TODO(loc_id, "failed to map Carbon type to C++");
      return nullptr;
    }
    auto ref_type = context.ast_context().getLValueReferenceType(cpp_type);
    cpp_param_types.push_back(ref_type);
  }

  CARBON_CHECK(function.return_type_inst_id == SemIR::TypeInstId::None);
  auto cpp_return_type = context.ast_context().VoidTy;

  auto cpp_function_type = context.ast_context().getFunctionType(
      cpp_return_type, cpp_param_types,
      clang::FunctionProtoType::ExtProtoInfo());

  auto* identifier_info = GetClangIdentifierInfo(context, function.name_id);
  CARBON_CHECK(identifier_info, "function with non-identifier name {0}",
               function.name_id);

  clang::FunctionDecl* function_decl = clang::FunctionDecl::Create(
      context.ast_context(), context.ast_context().getTranslationUnitDecl(),
      /*StartLoc=*/clang_loc, /*NLoc=*/clang_loc, identifier_info,
      cpp_function_type, /*TInfo=*/nullptr, clang::SC_Extern);

  // Build parameter decls.
  llvm::SmallVector<clang::ParmVarDecl*> param_var_decls;
  for (auto [i, type] : llvm::enumerate(cpp_param_types)) {
    clang::ParmVarDecl* param = clang::ParmVarDecl::Create(
        context.ast_context(), function_decl, /*StartLoc=*/clang_loc,
        /*IdLoc=*/clang_loc, /*Id=*/nullptr, type, /*TInfo=*/nullptr,
        clang::SC_None, /*DefArg=*/nullptr);
    param_var_decls.push_back(param);
  }
  function_decl->setParams(param_var_decls);

  // Mangle the function name and attach it to the `FunctionDecl`.
  SemIR::Mangler m(context.sem_ir(), context.total_ir_count());
  std::string mangled_name = m.Mangle(function_id, SemIR::SpecificId::None);
  function_decl->addAttr(
      clang::AsmLabelAttr::Create(context.ast_context(), mangled_name));

  return function_decl;
}

// Create the declaration of the C++ thunk.
static auto BuildCppToCarbonThunkDecl(
    Context& context, SemIR::LocId loc_id, const FunctionInfo& target,
    clang::DeclarationName thunk_name,
    llvm::ArrayRef<clang::QualType> thunk_param_types) -> clang::FunctionDecl* {
  clang::ASTContext& ast_context = context.ast_context();

  auto clang_loc = GetCppLocation(context, loc_id);

  // Get the C++ return type (this corresponds to the return type of the
  // target Carbon function).
  clang::QualType cpp_return_type = context.ast_context().VoidTy;
  auto return_type_id = target.function.GetDeclaredReturnType(context.sem_ir());
  if (return_type_id != SemIR::TypeId::None) {
    cpp_return_type = MapToCppType(context, return_type_id);
    if (cpp_return_type.isNull()) {
      context.TODO(loc_id, "failed to map Carbon return type to C++ type");
      return nullptr;
    }
  }

  clang::DeclarationNameInfo name_info(thunk_name, clang_loc);

  auto ext_proto_info = clang::FunctionProtoType::ExtProtoInfo();
  clang::QualType thunk_function_type = ast_context.getFunctionType(
      cpp_return_type, thunk_param_types, ext_proto_info);

  auto* tinfo =
      ast_context.getTrivialTypeSourceInfo(thunk_function_type, clang_loc);

  bool uses_fp_intrin = false;
  bool inline_specified = true;
  auto constexpr_kind = clang::ConstexprSpecKind::Unspecified;
  auto trailing_requires_clause = clang::AssociatedConstraint();

  clang::FunctionDecl* thunk_function_decl = nullptr;
  if (auto* parent_class =
          dyn_cast<clang::CXXRecordDecl>(target.decl_context)) {
    thunk_function_decl = clang::CXXMethodDecl::Create(
        ast_context, parent_class, clang_loc, name_info, thunk_function_type,
        tinfo, target.GetStorageClass(), uses_fp_intrin, inline_specified,
        constexpr_kind, clang_loc, trailing_requires_clause);
    // TODO: Map Carbon access to C++ access.
    thunk_function_decl->setAccess(clang::AS_public);
  } else {
    thunk_function_decl = clang::FunctionDecl::Create(
        ast_context, target.decl_context, clang_loc, name_info,
        thunk_function_type, tinfo, clang::SC_None, uses_fp_intrin,
        inline_specified,
        /*hasWrittenPrototype=*/true, constexpr_kind, trailing_requires_clause);
  }
  target.decl_context->addHiddenDecl(thunk_function_decl);

  llvm::SmallVector<clang::ParmVarDecl*> param_var_decls;
  for (auto [i, type] : llvm::enumerate(thunk_param_types)) {
    clang::ParmVarDecl* thunk_param = clang::ParmVarDecl::Create(
        ast_context, thunk_function_decl, /*StartLoc=*/clang_loc,
        /*IdLoc=*/clang_loc, /*Id=*/nullptr, type,
        /*TInfo=*/nullptr, clang::SC_None, /*DefArg=*/nullptr);
    param_var_decls.push_back(thunk_param);
  }
  thunk_function_decl->setParams(param_var_decls);

  // Force the thunk to be inlined and discarded.
  thunk_function_decl->addAttr(
      clang::AlwaysInlineAttr::CreateImplicit(ast_context));
  thunk_function_decl->addAttr(
      clang::InternalLinkageAttr::CreateImplicit(ast_context));

  return thunk_function_decl;
}

// Create the body of a C++ thunk that calls a Carbon thunk. The
// arguments are passed by reference to the callee.
static auto BuildCppToCarbonThunkBody(clang::Sema& sema,
                                      const FunctionInfo& target,
                                      clang::FunctionDecl* function_decl,
                                      clang::FunctionDecl* callee_function_decl)
    -> clang::StmtResult {
  clang::SourceLocation clang_loc = function_decl->getLocation();

  llvm::SmallVector<clang::Stmt*> stmts;

  // Create return storage if the target function returns non-void.
  const bool has_return_value = !function_decl->getReturnType()->isVoidType();
  clang::VarDecl* return_storage_var_decl = nullptr;
  clang::ExprResult return_storage_expr;
  if (has_return_value) {
    auto& return_storage_ident =
        sema.getASTContext().Idents.get("return_storage");
    return_storage_var_decl =
        clang::VarDecl::Create(sema.getASTContext(), function_decl,
                               /*StartLoc=*/clang_loc,
                               /*IdLoc=*/clang_loc, &return_storage_ident,
                               function_decl->getReturnType(),
                               /*TInfo=*/nullptr, clang::SC_None);
    return_storage_var_decl->setNRVOVariable(true);
    return_storage_expr = sema.BuildDeclRefExpr(
        return_storage_var_decl, return_storage_var_decl->getType(),
        clang::VK_LValue, clang_loc);

    auto decl_group_ref = clang::DeclGroupRef(return_storage_var_decl);
    auto decl_stmt =
        sema.ActOnDeclStmt(clang::Sema::DeclGroupPtrTy::make(decl_group_ref),
                           clang_loc, clang_loc);
    stmts.push_back(decl_stmt.get());
  }

  clang::ExprResult callee = sema.BuildDeclRefExpr(
      callee_function_decl, callee_function_decl->getType(), clang::VK_PRValue,
      clang_loc);

  llvm::SmallVector<clang::Expr*> call_args;
  // For methods, pass the `this` pointer as the first argument to the callee.
  if (target.has_self()) {
    auto* parent_class = cast<clang::CXXRecordDecl>(target.decl_context);
    clang::QualType class_type =
        sema.getASTContext().getCanonicalTagType(parent_class);
    auto class_ptr_type = sema.getASTContext().getPointerType(class_type);
    auto* this_expr = sema.BuildCXXThisExpr(clang_loc, class_ptr_type,
                                            /*IsImplicit=*/true);
    this_expr = clang::UnaryOperator::Create(
        sema.getASTContext(), this_expr, clang::UO_Deref, class_type,
        clang::ExprValueKind::VK_LValue, clang::ExprObjectKind::OK_Ordinary,
        clang_loc, /*CanOverflow=*/false, clang::FPOptionsOverride());
    call_args.push_back(this_expr);
  }
  for (auto* param : function_decl->parameters()) {
    clang::Expr* call_arg =
        sema.BuildDeclRefExpr(param, param->getType().getNonReferenceType(),
                              clang::VK_LValue, clang_loc);
    call_args.push_back(call_arg);
  }

  // If the target function returns non-void, the Carbon thunk takes an
  // extra output parameter referencing the return storage.
  if (has_return_value) {
    call_args.push_back(return_storage_expr.get());
  }

  clang::ExprResult call = sema.BuildCallExpr(nullptr, callee.get(), clang_loc,
                                              call_args, clang_loc);
  CARBON_CHECK(call.isUsable());
  stmts.push_back(call.get());

  if (has_return_value) {
    auto* return_stmt = clang::ReturnStmt::Create(
        sema.getASTContext(), clang_loc, return_storage_expr.get(),
        return_storage_var_decl);
    stmts.push_back(return_stmt);
  }

  return clang::CompoundStmt::Create(sema.getASTContext(), stmts,
                                     clang::FPOptionsOverride(), clang_loc,
                                     clang_loc);
}

// Create a C++ thunk that calls the Carbon thunk. The C++ thunk's
// parameter types are mapped from the parameters of the target function
// with `MapToCppType`. (Note that the target function here is the
// callee of the Carbon thunk.)
static auto BuildCppToCarbonThunk(Context& context, SemIR::LocId loc_id,
                                  const FunctionInfo& target,
                                  llvm::StringRef thunk_name,
                                  clang::FunctionDecl* carbon_function_decl)
    -> clang::FunctionDecl* {
  auto& thunk_ident = context.ast_context().Idents.get(thunk_name);

  llvm::SmallVector<clang::QualType> param_types;
  for (auto param : target.explicit_params) {
    auto cpp_type = MapToCppType(context, param.type_id);
    if (cpp_type.isNull()) {
      context.TODO(loc_id, "failed to map C++ type to Carbon");
      return nullptr;
    }
    if (param.is_ref) {
      cpp_type = context.ast_context().getLValueReferenceType(cpp_type);
    }
    param_types.push_back(cpp_type);
  }

  auto* thunk_function_decl = BuildCppToCarbonThunkDecl(
      context, loc_id, target, &thunk_ident, param_types);

  // Build the thunk function body.
  clang::Sema& sema = context.clang_sema();
  clang::Sema::ContextRAII context_raii(sema, thunk_function_decl);
  sema.ActOnStartOfFunctionDef(nullptr, thunk_function_decl);
  clang::StmtResult body = BuildCppToCarbonThunkBody(
      sema, target, thunk_function_decl, carbon_function_decl);
  sema.ActOnFinishFunctionBody(thunk_function_decl, body.get());
  CARBON_CHECK(!body.isInvalid());

  context.clang_sema().getASTConsumer().HandleTopLevelDecl(
      clang::DeclGroupRef(thunk_function_decl));
  return thunk_function_decl;
}

// Create a Carbon thunk that calls `callee`. The thunk's parameters are
// all references to the callee parameter type.
static auto BuildCarbonToCarbonThunk(Context& context, SemIR::LocId loc_id,
                                     const FunctionInfo& target)
    -> SemIR::FunctionId {
  // Create the thunk's name.
  llvm::SmallString<64> thunk_name =
      context.names().GetFormatted(target.function.name_id);
  thunk_name += "__carbon_thunk";
  auto& ident = context.ast_context().Idents.get(thunk_name);
  auto thunk_name_id =
      SemIR::NameId::ForIdentifier(context.identifiers().Add(ident.getName()));

  // Get the thunk's parameters. These match the callee parameters, with
  // the addition of an output parameter for the callee's return value
  // (if it has one).
  llvm::SmallVector<SemIR::TypeId> thunk_param_type_ids;
  for (const auto& param : target.explicit_params) {
    thunk_param_type_ids.push_back(param.type_id);
  }
  auto callee_return_type_id =
      target.function.GetDeclaredReturnType(context.sem_ir());
  if (callee_return_type_id != SemIR::TypeId::None) {
    thunk_param_type_ids.push_back(callee_return_type_id);
  }

  auto carbon_thunk_function_id =
      MakeGeneratedFunctionDecl(
          context, loc_id,
          {.parent_scope_id = target.function.parent_scope_id,
           .name_id = thunk_name_id,
           .self_type_id = target.self_type_id,
           .param_type_ids = thunk_param_type_ids,
           .param_kind = ParamPatternKind::Ref})
          .second;

  BuildThunkDefinitionForExport(
      context, carbon_thunk_function_id, target.function_id,
      context.functions().Get(carbon_thunk_function_id).first_decl_id(),
      target.function.first_decl_id());

  return carbon_thunk_function_id;
}

auto ExportFunctionToCpp(Context& context, SemIR::LocId loc_id,
                         SemIR::FunctionId callee_function_id)
    -> clang::FunctionDecl* {
  const SemIR::Function& callee = context.functions().Get(callee_function_id);

  if (callee.generic_id.has_value()) {
    context.TODO(loc_id,
                 "unsupported: C++ calling a Carbon function with "
                 "generic parameters");
    return nullptr;
  }

  // Map the parent scope into the C++ AST.
  auto* decl_context =
      ExportNameScopeToCpp(context, loc_id, callee.parent_scope_id);
  if (!decl_context) {
    return nullptr;
  }

  FunctionInfo target_function_info(context, callee_function_id, callee,
                                    decl_context);

  // Create a Carbon thunk that calls the callee. The thunk's parameters
  // are all references so that the ABI is compatible with C++ callers.
  auto carbon_thunk_function_id =
      BuildCarbonToCarbonThunk(context, loc_id, target_function_info);

  // Create a `clang::FunctionDecl` that can be used to call the Carbon thunk.
  auto* carbon_function_decl = BuildCppFunctionDeclForCarbonFn(
      context, loc_id, carbon_thunk_function_id);
  if (!carbon_function_decl) {
    return nullptr;
  }

  // Create a C++ thunk that calls the Carbon thunk.
  return BuildCppToCarbonThunk(context, loc_id, target_function_info,
                               context.names().GetFormatted(callee.name_id),
                               carbon_function_decl);
}

}  // namespace Carbon::Check
