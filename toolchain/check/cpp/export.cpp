// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/export.h"

#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/type_mapping.h"
#include "toolchain/check/function.h"
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

  // Get parameters types.
  auto carbon_function_params =
      context.inst_blocks().Get(function.call_param_patterns_id);
  llvm::SmallVector<clang::QualType> cpp_param_types;
  for (auto param_inst_id : carbon_function_params) {
    auto scrutinee_type_id = ExtractScrutineeType(
        context.sem_ir(), context.insts().Get(param_inst_id).type_id());
    auto cpp_type = MapToCppType(context, scrutinee_type_id);
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

  auto* function_decl = clang::FunctionDecl::Create(
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
    Context& context, SemIR::LocId loc_id, clang::DeclContext* decl_context,
    clang::DeclarationName thunk_name,
    llvm::ArrayRef<clang::QualType> thunk_param_types,
    SemIR::TypeId return_type_id) -> clang::FunctionDecl* {
  clang::ASTContext& ast_context = context.ast_context();

  auto clang_loc = GetCppLocation(context, loc_id);

  // Get the C++ return type (this corresponds to the return type of the
  // target Carbon function).
  clang::QualType cpp_return_type = context.ast_context().VoidTy;
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

  const bool uses_fp_intrin = false;
  const bool inline_specified = true;
  const auto constexpr_kind = clang::ConstexprSpecKind::Unspecified;
  const auto trailing_requires_clause = clang::AssociatedConstraint();

  clang::FunctionDecl* thunk_function_decl = nullptr;
  if (auto* parent_class = dyn_cast<clang::CXXRecordDecl>(decl_context)) {
    // TODO: Support non-static methods.
    thunk_function_decl = clang::CXXMethodDecl::Create(
        ast_context, parent_class, clang_loc, name_info, thunk_function_type,
        tinfo, clang::SC_Static, uses_fp_intrin, inline_specified,
        constexpr_kind, clang_loc, trailing_requires_clause);
    // TODO: Map Carbon access to C++ access.
    thunk_function_decl->setAccess(clang::AS_public);
  } else {
    thunk_function_decl = clang::FunctionDecl::Create(
        ast_context, decl_context, clang_loc, name_info, thunk_function_type,
        tinfo, clang::SC_None, uses_fp_intrin, inline_specified,
        /*hasWrittenPrototype=*/true, constexpr_kind, trailing_requires_clause);
  }
  decl_context->addDecl(thunk_function_decl);

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
static auto BuildCppToCarbonThunk(
    Context& context, SemIR::LocId loc_id, clang::DeclContext* decl_context,
    llvm::StringRef base_name, clang::FunctionDecl* carbon_function_decl,
    llvm::ArrayRef<SemIR::TypeId> callee_param_type_ids,
    SemIR::TypeId return_type_id) -> clang::FunctionDecl* {
  // Create the thunk's name.
  llvm::SmallString<64> thunk_name = base_name;
  thunk_name += "__cpp_thunk";
  auto& thunk_ident = context.ast_context().Idents.get(thunk_name);

  llvm::SmallVector<clang::QualType> param_types;
  for (auto type_id : callee_param_type_ids) {
    auto cpp_type = MapToCppType(context, type_id);
    if (cpp_type.isNull()) {
      context.TODO(loc_id, "failed to map C++ type to Carbon");
      return nullptr;
    }
    param_types.push_back(cpp_type);
  }

  auto* thunk_function_decl = BuildCppToCarbonThunkDecl(
      context, loc_id, decl_context, &thunk_ident, param_types, return_type_id);

  // Build the thunk function body.
  clang::Sema& sema = context.clang_sema();
  clang::Sema::ContextRAII context_raii(sema, thunk_function_decl);
  sema.ActOnStartOfFunctionDef(nullptr, thunk_function_decl);
  clang::StmtResult body = BuildCppToCarbonThunkBody(sema, thunk_function_decl,
                                                     carbon_function_decl);
  sema.ActOnFinishFunctionBody(thunk_function_decl, body.get());
  CARBON_CHECK(!body.isInvalid());

  context.clang_sema().getASTConsumer().HandleTopLevelDecl(
      clang::DeclGroupRef(thunk_function_decl));
  return thunk_function_decl;
}

// Create a Carbon thunk that calls `callee`. The thunk's parameters are
// all references to the callee parameter type.
static auto BuildCarbonToCarbonThunk(
    Context& context, SemIR::LocId loc_id, SemIR::FunctionId callee_function_id,
    const SemIR::Function& callee,
    llvm::ArrayRef<SemIR::TypeId> callee_param_type_ids) -> SemIR::FunctionId {
  // Create the thunk's name.
  llvm::SmallString<64> thunk_name =
      context.names().GetFormatted(callee.name_id);
  thunk_name += "__carbon_thunk";
  auto& ident = context.ast_context().Idents.get(thunk_name);
  auto thunk_name_id =
      SemIR::NameId::ForIdentifier(context.identifiers().Add(ident.getName()));

  // Get the thunk's parameters. These match the callee parameters, with
  // the addition of an output parameter for the callee's return value
  // (if it has one).
  llvm::SmallVector<SemIR::TypeId> thunk_param_type_ids(callee_param_type_ids);
  auto callee_return_type_id = callee.GetDeclaredReturnType(context.sem_ir());
  if (callee_return_type_id != SemIR::TypeId::None) {
    thunk_param_type_ids.push_back(callee_return_type_id);
  }

  auto carbon_thunk_function_id =
      MakeGeneratedFunctionDecl(context, loc_id,
                                {.parent_scope_id = callee.parent_scope_id,
                                 .name_id = thunk_name_id,
                                 .param_type_ids = thunk_param_type_ids,
                                 .params_are_refs = true})
          .second;

  BuildThunkDefinitionForExport(
      context, carbon_thunk_function_id, callee_function_id,
      context.functions().Get(carbon_thunk_function_id).first_decl_id(),
      callee.first_decl_id());

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

  if (callee.call_param_ranges.implicit_size() != 0) {
    context.TODO(loc_id,
                 "unsupported: C++ calling a Carbon function with "
                 "an implicit parameter");
    return nullptr;
  }

  // Map the parent scope into the C++ AST.
  auto* decl_context =
      ExportNameScopeToCpp(context, loc_id, callee.parent_scope_id);
  if (!decl_context) {
    return nullptr;
  }

  // Get the parameter types of the Carbon function being
  // called. Exclude return params, if present.
  auto callee_function_params =
      context.inst_blocks().Get(callee.call_param_patterns_id);
  callee_function_params =
      callee_function_params.drop_back(callee.call_param_ranges.return_size());

  llvm::SmallVector<SemIR::TypeId> callee_param_type_ids;
  for (auto callee_param_inst_id : callee_function_params) {
    auto scrutinee_type_id = ExtractScrutineeType(
        context.sem_ir(), context.insts().Get(callee_param_inst_id).type_id());
    callee_param_type_ids.push_back(scrutinee_type_id);
  }

  // Create a Carbon thunk that calls the callee. The thunk's parameters
  // are all references so that the ABI is compatible with C++ callers.
  auto carbon_thunk_function_id = BuildCarbonToCarbonThunk(
      context, loc_id, callee_function_id, callee, callee_param_type_ids);

  // Create a `clang::FunctionDecl` that can be used to call the Carbon thunk.
  auto* carbon_function_decl = BuildCppFunctionDeclForCarbonFn(
      context, loc_id, carbon_thunk_function_id);
  if (!carbon_function_decl) {
    return nullptr;
  }

  // Create a C++ thunk that calls the Carbon thunk.
  return BuildCppToCarbonThunk(context, loc_id, decl_context,
                               context.names().GetFormatted(callee.name_id),
                               carbon_function_decl, callee_param_type_ids,
                               callee.GetDeclaredReturnType(context.sem_ir()));
}

}  // namespace Carbon::Check
