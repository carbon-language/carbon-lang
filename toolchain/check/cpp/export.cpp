// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/export.h"

#include <optional>
#include <string_view>

#include "clang/AST/ASTConsumer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/EnterExpressionEvaluationContext.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/Casting.h"
#include "toolchain/check/cpp/access.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/type_mapping.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/function.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/thunk.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/mangler.h"
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/typed_insts.h"
#include "toolchain/sem_ir/vtable.h"

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
  auto* decl = context.clang_decls().Get(clang_decl_context_id).decl();
  return cast<clang::DeclContext>(decl);
}

// Exports a Carbon class into C++ as a class in the given `decl_context`.
//
// This does not check for an existing export of the class, nor does it add
// the class to `clang_decls()`.
//
// Returns nullptr if the class could not be exported and an error was
// diagnosed.
static auto ExportClassToCppInDeclContext(Context& context,
                                          clang::DeclContext* decl_context,
                                          const SemIR::Class& class_info,
                                          const SemIR::SpecificId specific_id)
    -> clang::TagDecl* {
  SemIR::LocId loc_id(class_info.first_decl_id());

  if (specific_id.has_value()) {
    context.TODO(loc_id, "interop with specific class");
    return nullptr;
  }

  auto* identifier_info = GetClangIdentifierInfo(context, class_info.name_id);
  CARBON_CHECK(identifier_info, "non-identifier class name {0}",
               class_info.name_id);

  auto clang_loc = GetCppLocation(context, loc_id);
  auto* record_decl = clang::CXXRecordDecl::Create(
      context.ast_context(), clang::TagTypeKind::Class, decl_context, clang_loc,
      clang_loc, identifier_info);
  // If this is a member class, set its access.
  if (isa<clang::CXXRecordDecl>(decl_context)) {
    // TODO: Map Carbon access to C++ access.
    record_decl->setAccess(clang::AS_public);
  }

  decl_context->addHiddenDecl(record_decl);
  record_decl->setHasExternalLexicalStorage();
  record_decl->setHasExternalVisibleStorage();

  return record_decl;
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

    auto const_inst_id =
        context.constant_values().GetConstantInstId(name_scope.inst_id());
    if (context.insts().Is<SemIR::Namespace>(const_inst_id)) {
      auto* identifier_info =
          GetClangIdentifierInfo(context, name_scope.name_id());
      if (!identifier_info) {
        // TODO: Handle keyword package names like `Cpp` and `Core`. These can
        // be named from C++ via an alias.
        context.TODO(loc_id, "interop with non-identifier package name");
        return nullptr;
      }

      // TODO: Provide a source location.
      auto* namespace_decl = clang::NamespaceDecl::Create(
          context.ast_context(), decl_context, false, clang::SourceLocation(),
          clang::SourceLocation(), identifier_info, nullptr, false);
      decl_context->addHiddenDecl(namespace_decl);
      decl_context = namespace_decl;
    } else if (auto class_type =
                   context.insts().TryGetAs<SemIR::ClassType>(const_inst_id)) {
      const auto& class_info = context.classes().Get(class_type->class_id);
      decl_context = ExportClassToCppInDeclContext(
          context, decl_context, class_info, class_type->specific_id);
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

    // Complete the type here to avoid hitting a clang assert later when
    // adding methods.
    if (auto* record_decl = llvm::dyn_cast<clang::RecordDecl>(decl_context)) {
      context.ast_context().getExternalSource()->CompleteType(record_decl);
    }
  }

  return decl_context;
}

auto ExportClassToCpp(Context& context, SemIR::ClassType class_type)
    -> clang::TagDecl* {
  const auto& class_info = context.classes().Get(class_type.class_id);
  SemIR::LocId loc_id(class_info.first_decl_id());

  if (class_type.specific_id.has_value()) {
    context.TODO(loc_id, "interop with specific class");
    return nullptr;
  }

  // If this class was produced by importing a C++ declaration or has
  // already been exported to C++, return the corresponding Clang declaration.
  // That could either be a CXXRecordDecl or an EnumDecl.
  if (const auto* clang_decl =
          context.clang_decls().Lookup(class_info.first_decl_id())) {
    return cast<clang::TagDecl>(clang_decl->decl());
  }

  auto* decl_context =
      ExportNameScopeToCpp(context, loc_id, class_info.parent_scope_id);
  auto* record_decl = ExportClassToCppInDeclContext(
      context, decl_context, class_info, class_type.specific_id);

  auto key =
      SemIR::ClangDeclKey::ForNonFunctionDecl(cast<clang::Decl>(record_decl));
  auto clang_decl_id = context.clang_decls().Add(
      {.key = key, .inst_id = class_info.first_decl_id()});
  if (class_info.scope_id.has_value()) {
    // TODO: Record the Carbon class -> clang declaration mapping for incomplete
    // classes too.
    context.name_scopes()
        .Get(class_info.scope_id)
        .set_clang_decl_context_id(clang_decl_id, /*is_cpp_scope=*/false);
  }
  return record_decl;
}

// Export the bindings in a generic as a `clang::TemplateParameterList`.
static auto ExportGenericBindings(Context& context, SemIR::LocId loc_id,
                                  SemIR::GenericId generic_id,
                                  clang::DeclContext* decl_context)
    -> clang::TemplateParameterList* {
  auto clang_loc = GetCppLocation(context, loc_id);

  const auto& generic = context.generics().Get(generic_id);
  auto bindings = context.inst_blocks().Get(generic.bindings_id);
  llvm::SmallVector<clang::NamedDecl*> template_param_decls;

  // Create `clang::TemplateTypeParmDecl`s for each of the generic's bindings.
  //
  // TODO: handle the case where the generic is within an enclosing generic,
  // and only include the bindings introduced in the inner generic here. See
  // `fail_todo_enclosing_generic.carbon`.
  for (auto binding_inst_id : bindings) {
    binding_inst_id =
        context.constant_values().GetConstantInstId(binding_inst_id);
    auto symbolic_binding =
        context.insts().GetAs<SemIR::SymbolicBinding>(binding_inst_id);

    const auto& entity_name =
        context.entity_names().Get(symbolic_binding.entity_name_id);

    auto* param_ident = GetClangIdentifierInfo(context, entity_name.name_id);
    CARBON_CHECK(param_ident, "non-identifier param name {0}",
                 entity_name.name_id);

    if (symbolic_binding.type_id != SemIR::TypeType::TypeId &&
        !context.types().Is<SemIR::FacetType>(symbolic_binding.type_id)) {
      context.TODO(loc_id, "binding maps to a non-type template parameter");
      return nullptr;
    }

    auto* param_decl = clang::TemplateTypeParmDecl::Create(
        context.ast_context(), decl_context, /*KeyLoc=*/clang_loc,
        /*NameLoc=*/clang_loc,
        /*D=*/0, /*P=*/0, param_ident, /*Typename=*/true,
        /*ParameterPack=*/false);
    template_param_decls.push_back(param_decl);

    // Store a mapping between the generic parameter's `TypeInstId` and
    // the `clang::TemplateTypeParmDecl`.
    auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(param_decl);
    context.clang_decls().Add({.key = key, .inst_id = binding_inst_id});
  }

  return clang::TemplateParameterList::Create(context.ast_context(),
                                              /*TemplateLoc=*/clang_loc,
                                              /*LAngleLoc=*/clang_loc,
                                              template_param_decls,
                                              /*RAngleLoc=*/clang_loc,
                                              /*RequiresClause=*/nullptr);
}

/// Create a Specific for the given generic using the given template args.
///
/// Returns `SemIR::SpecificId::None` if an error occurs.
static auto MakeSpecificForTemplateArgs(
    Context& context, SemIR::LocId loc_id, SemIR::GenericId generic_id,
    llvm::ArrayRef<clang::TemplateArgument> template_args)
    -> SemIR::SpecificId {
  const auto& generic = context.generics().Get(generic_id);

  auto bindings = context.inst_blocks().Get(generic.bindings_id);
  CARBON_CHECK(bindings.size() == template_args.size());

  // Map the `clang::TemplateArgument`s into Carbon types suitable for
  // passing into `MakeSpecific`.
  llvm::SmallVector<SemIR::InstId> specific_arg_ids;
  for (auto [binding_inst_id, clang_template_arg] :
       llvm::zip(bindings, template_args)) {
    auto type_expr =
        ImportCppType(context, loc_id, clang_template_arg.getAsType());
    if (type_expr.type_id == SemIR::ErrorInst::TypeId) {
      return SemIR::SpecificId::None;
    }
    if (!type_expr.type_id.has_value()) {
      context.TODO(loc_id, "failed to import C++ type");
      return SemIR::SpecificId::None;
    }

    auto binding_const_inst_id =
        context.constant_values().GetConstantInstId(binding_inst_id);

    specific_arg_ids.push_back(ConvertToValueOfType(
        context, loc_id, type_expr.inst_id,
        context.insts().Get(binding_const_inst_id).type_id()));
  }

  return MakeSpecific(context, loc_id, generic_id, specific_arg_ids);
}

auto ExportGenericClassToCpp(Context& context, SemIR::InstId inst_id,
                             SemIR::GenericClassType generic_class_type)
    -> clang::ClassTemplateDecl* {
  // TODO
  (void)context;
  (void)inst_id;
  (void)generic_class_type;
  return nullptr;
}

static auto SetCppClassMemberAccess(const SemIR::NameScope& class_scope,
                                    SemIR::NameId member_name_id,
                                    clang::Decl* member) -> void {
  auto entry_id = class_scope.Lookup(member_name_id);
  CARBON_CHECK(entry_id.has_value());
  const auto& entry = class_scope.GetEntry(*entry_id);
  member->setAccess(MapToCppAccess(entry.result.access_kind()));
}

// Creates a `clang::FieldDecl` for a Carbon class field. Returns
// nullptr if an error occurs.
static auto CreateCppFieldDecl(Context& context,
                               const SemIR::NameScope& class_scope,
                               clang::CXXRecordDecl* record_decl,
                               SemIR::InstId field_inst_id,
                               const SemIR::FieldDecl& field_decl)
    -> clang::FieldDecl* {
  // Get the field's C++ type.
  auto unbound_element_type =
      context.types().GetAs<SemIR::UnboundElementType>(field_decl.type_id);
  auto cpp_type =
      MapToCppType(context, context.types().GetTypeIdForTypeInstId(
                                unbound_element_type.element_type_inst_id));
  if (cpp_type.isNull()) {
    context.TODO(field_inst_id, "failed to map Carbon type to C++");
    return nullptr;
  }

  // Get the field's C++ identifier.
  auto* identifier_info = GetClangIdentifierInfo(context, field_decl.name_id);
  CARBON_CHECK(identifier_info, "field with non-identifier name {0}",
               field_decl.name_id);

  // Create the `clang::FieldDecl`.
  auto clang_loc = GetCppLocation(context, SemIR::LocId(field_inst_id));
  auto* cpp_field_decl = clang::FieldDecl::Create(
      context.ast_context(), record_decl, /*StartLoc=*/clang_loc,
      /*IdLoc=*/clang_loc, identifier_info, cpp_type, /*TInfo=*/nullptr,
      /*BW=*/nullptr,
      /*Mutable=*/true, clang::ICIS_NoInit);

  SetCppClassMemberAccess(class_scope, field_decl.name_id, cpp_field_decl);

  record_decl->addHiddenDecl(cpp_field_decl);

  return cpp_field_decl;
}

// Create an invalid `clang::FieldDecl`. This is only used as an error marker
// to indicate that a Carbon field has already been unsuccessfully exported.
static auto CreateInvalidFieldDecl(Context& context,
                                   clang::DeclContext* decl_context)
    -> clang::FieldDecl* {
  clang::SourceLocation clang_loc;
  auto* identifier_info =
      context.clang_sema().getPreprocessor().getIdentifierInfo("invalid_field");
  auto cpp_type = context.ast_context().IntTy;
  auto* field_decl = clang::FieldDecl::Create(
      context.ast_context(), decl_context, /*StartLoc=*/clang_loc,
      /*IdLoc=*/clang_loc, identifier_info, cpp_type, /*TInfo=*/nullptr,
      /*BW=*/nullptr,
      /*Mutable=*/true, clang::ICIS_NoInit);
  field_decl->setInvalidDecl();
  return field_decl;
}

auto ExportAllFieldsToCpp(Context& context, SemIR::Class& class_info) -> void {
  const auto& class_scope = context.name_scopes().Get(class_info.scope_id);

  for (const auto& struct_field : class_info.GetStructTypeFields(
           context.sem_ir(), SemIR::SpecificId::None)) {
    auto class_field = LookupClassFieldByStructField(context.sem_ir(),
                                                     class_scope, struct_field);
    if (!class_field) {
      continue;
    }

    // Return early if the field is already exported. Since fields are always
    // exported as a group, this indicates all fields have been exported so
    // there's no need to continue to the rest.
    if (context.clang_decls().Lookup(class_field->inst_id)) {
      return;
    }

    // Map the parent scope into the C++ AST.
    auto* decl_context = ExportNameScopeToCpp(
        context, SemIR::LocId(class_field->inst_id), class_info.scope_id);
    if (!decl_context) {
      continue;
    }

    auto* cpp_field_decl = CreateCppFieldDecl(
        context, class_scope, cast<clang::CXXRecordDecl>(decl_context),
        class_field->inst_id, class_field->inst);

    // If the field cannot be exported, create an invalid `FieldDecl` to store
    // in `clang_decls`. This marks the field as unsuccessfully exported, so
    // that we know not to attempt export again (which could create duplicate
    // error diagnostics).
    if (!cpp_field_decl) {
      cpp_field_decl = CreateInvalidFieldDecl(context, decl_context);
    }

    // Create and store the `ClangDeclId`.
    auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(cpp_field_decl);
    context.clang_decls().Add({.key = key, .inst_id = class_field->inst_id});
  }
}

auto ExportFieldToCpp(Context& context, SemIR::InstId field_inst_id,
                      SemIR::FieldDecl field_decl) -> clang::FieldDecl* {
  // Get the `SemIR::Class` that contains the `field_decl`.
  auto unbound_element_type =
      context.types().GetAs<SemIR::UnboundElementType>(field_decl.type_id);
  SemIR::TypeId class_type_id = context.types().GetTypeIdForTypeInstId(
      unbound_element_type.class_type_inst_id);
  auto class_type = context.types().GetAs<SemIR::ClassType>(class_type_id);
  auto& class_info = context.classes().Get(class_type.class_id);

  // If the class's fields haven't already been exported, do so now.
  ExportAllFieldsToCpp(context, class_info);

  // Get the exported `clang::FieldDecl`.
  if (const auto* clang_decl = context.clang_decls().Lookup(field_inst_id)) {
    if (!clang_decl->decl()->isInvalidDecl()) {
      return cast<clang::FieldDecl>(clang_decl->decl());
    }
  }
  return nullptr;
}

namespace {
struct FunctionInfo {
  struct Param {
    Param(Context& context, SemIR::InstId param_inst_id)
        : pattern_inst_id(param_inst_id),
          type_id(ExtractScrutineeType(
              context.sem_ir(), context.insts().Get(param_inst_id).type_id())),
          kind(GetParamPatternKind(context, param_inst_id)) {}

    // The parameter's pattern type.
    SemIR::InstId pattern_inst_id;

    // Type of the parameter's scrutinee.
    SemIR::TypeId type_id;

    // Kind of the parameter pattern.
    ParamPatternKind kind;
  };

  explicit FunctionInfo(Context& context, SemIR::FunctionId function_id,
                        const SemIR::Function& function,
                        clang::DeclContext* decl_context,
                        bool export_as_constructor)
      : function_id(function_id),
        function(function),
        decl_context(decl_context),
        return_type_id(function.GetDeclaredReturnType(context.sem_ir())),
        export_as_constructor(export_as_constructor) {
    auto function_params =
        context.inst_blocks().Get(function.call_param_patterns_id);
    const auto& ranges = function.call_param_ranges;
    auto explicit_begin = ranges.explicit_begin().index;

    // Get the function's `self` parameter, if present. `self` is the first
    // explicit call parameter. (The lowered call parameters are leaf patterns
    // without binding names, so we rely on `self_param_id` and the positional
    // convention rather than inspecting the pattern.)
    if (function.self_param_id.has_value()) {
      CARBON_CHECK(explicit_begin != ranges.explicit_end().index);
      self_param = Param(context, function_params[explicit_begin]);
      ++explicit_begin;
    }

    // The remaining explicit parameters are the caller-provided arguments.
    for (auto i = explicit_begin; i != ranges.explicit_end().index; ++i) {
      explicit_params.push_back(Param(context, function_params[i]));
    }
  }

  // Get the `StorageClass` to use for `CXXMethodDecl`s.
  auto GetStorageClass() const -> clang::StorageClass {
    if (self_param) {
      return clang::SC_None;
    } else {
      return clang::SC_Static;
    }
  }

  // Get the `self` param type, or `None` if the function does not have
  // a `self` param.
  auto GetSelfTypeId() const -> SemIR::TypeId {
    if (self_param) {
      return self_param->type_id;
    }
    return SemIR::TypeId::None;
  }

  // Get the clang::DeclarationName of this function's C++ counterpart.
  auto GetCppName(Context& context) const -> clang::DeclarationName {
    if (export_as_constructor) {
      auto* record_decl = cast<clang::CXXRecordDecl>(decl_context);
      return context.ast_context().DeclarationNames.getCXXConstructorName(
          context.ast_context().getCanonicalTagType(record_decl));
    } else {
      return &context.ast_context().Idents.get(
          context.names().GetFormatted(function.name_id));
    }
  }

  SemIR::FunctionId function_id;
  const SemIR::Function& function;

  // Parent scope in the C++ AST where a C++ thunk for this function can
  // be created. If the function is a method or constructor, this will be a
  // `CXXRecordDecl`.
  clang::DeclContext* decl_context;

  // For each of the function's explicit parameters, the scrutinee type
  // and whether the parameter is a reference.
  llvm::SmallVector<Param> explicit_params;

  // Return type of the function.
  SemIR::TypeId return_type_id;

  // For methods, the type of `self` and whether it is a reference. If
  // the function does not have a `self` parameter, this is `nullopt`.
  std::optional<Param> self_param;

  // Whether this function should be exported as a C++ constructor.
  bool export_as_constructor;
};
}  // namespace

// Converts a Carbon parameter type to the parameter type that should be used
// for the C++ declaration of the Carbon -> Carbon thunk. This is always a
// reference type.
static auto MapToCppThunkParamType(Context& context, SemIR::TypeId type_id)
    -> clang::QualType {
  auto cpp_type = MapToCppType(context, type_id);
  if (cpp_type.isNull()) {
    return clang::QualType();
  }
  // The function exposed to C++ may have a `const&` parameter type for a value
  // parameter. Always use a const reference here so we accept the argument,
  // even though we might not need the `const`.
  return context.ast_context().getLValueReferenceType(
      context.ast_context().getConstType(cpp_type));
}

// Build FunctionInfo for an export of the given Carbon function. Exports the
// name scope if necessary.
static auto BuildFunctionInfo(Context& context, SemIR::LocId loc_id,
                              SemIR::FunctionId callee_function_id)
    -> std::optional<FunctionInfo> {
  const SemIR::Function& callee = context.functions().Get(callee_function_id);

  // Map the parent scope into the C++ AST.
  auto* decl_context =
      ExportNameScopeToCpp(context, loc_id, callee.parent_scope_id);
  if (!decl_context) {
    return std::nullopt;
  }

  bool export_as_constructor = false;
  const auto& parent_scope = context.name_scopes().Get(callee.parent_scope_id);
  if (auto class_decl =
          context.insts().TryGetAs<SemIR::ClassDecl>(parent_scope.inst_id())) {
    auto& class_info = context.classes().Get(class_decl->class_id);
    if (class_info.name_id == callee.name_id) {
      // If the function's name matches the name of the enclosing class,
      // we can't export it as an ordinary function, so from this point on
      // if we can't export it as a constructor we can't export it at all.
      //
      // TODO: figure out a way to provide good diagnostics in this situation.
      // Ideally we'd only diagnose if the user actually tries to call it,
      // because it's perfectly valid as Carbon code, but it's not clear how
      // to do that.
      //
      // TODO: some impl functions should also be exported as constructors
      // (e.g. `Core.Copy.Op`). Figure out how to avoid colliding with those
      // here.
      if (callee.self_param_id != SemIR::InstId::None) {
        return std::nullopt;
      }
      if (!context.insts().Is<SemIR::InitForm>(
              callee.GetDeclaredReturnForm(context.sem_ir()))) {
        return std::nullopt;
      }
      auto class_type_id =
          GetClassType(context, class_decl->class_id, SemIR::SpecificId::None);
      auto return_type_id =
          context.types().GetTypeIdForTypeInstId(callee.return_type_inst_id);
      if (class_type_id != return_type_id) {
        return std::nullopt;
      }
      // TODO figure out how to deal with explicit generic parameters.
      export_as_constructor = true;
    }
  }

  return FunctionInfo(context, callee_function_id, callee, decl_context,
                      export_as_constructor);
}

// Create a `clang::FunctionDecl` with the given parameter types and
// return type.
//
// The function's name will match the one referenced by `function_name_id`,
// and the function will be added to the given `decl_context`.
static auto BuildCppFunctionDecl(Context& context,
                                 clang::DeclContext* decl_context,
                                 SemIR::LocId loc_id,
                                 clang::DeclarationName declaration_name,
                                 clang::ArrayRef<clang::QualType> param_types,
                                 clang::QualType return_type,
                                 bool export_as_constructor) {
  auto clang_loc = GetCppLocation(context, loc_id);

  auto cpp_function_type = context.ast_context().getFunctionType(
      return_type, param_types, clang::FunctionProtoType::ExtProtoInfo());

  auto* tinfo = context.ast_context().getTrivialTypeSourceInfo(
      cpp_function_type, clang_loc);
  clang::FunctionDecl* function_decl;
  if (export_as_constructor) {
    auto* record_decl = cast<clang::CXXRecordDecl>(decl_context);
    function_decl = clang::CXXConstructorDecl::Create(
        context.ast_context(), record_decl, /*StartLoc=*/clang_loc,
        clang::DeclarationNameInfo{declaration_name, clang_loc},
        cpp_function_type, tinfo,
        clang::ExplicitSpecifier{nullptr,
                                 clang::ExplicitSpecKind::ResolvedTrue},
        /*UsesFPIntrin=*/false,
        /*isInline=*/false, /*isImplicitlyDeclared=*/false,
        clang::ConstexprSpecKind::Unspecified);
  } else {
    function_decl = clang::FunctionDecl::Create(
        context.ast_context(), decl_context,
        /*StartLoc=*/clang_loc, /*NLoc=*/clang_loc, declaration_name,
        cpp_function_type, tinfo, clang::SC_Extern);
  }

  // Build parameter decls.
  llvm::SmallVector<clang::ParmVarDecl*> param_var_decls;
  for (auto [i, type] : llvm::enumerate(param_types)) {
    auto* param_tinfo =
        context.ast_context().getTrivialTypeSourceInfo(type, clang_loc);
    clang::ParmVarDecl* param = clang::ParmVarDecl::Create(
        context.ast_context(), function_decl, /*StartLoc=*/clang_loc,
        /*IdLoc=*/clang_loc, /*Id=*/nullptr, type, param_tinfo, clang::SC_None,
        /*DefArg=*/nullptr);
    param_var_decls.push_back(param);
  }
  function_decl->setParams(param_var_decls);

  return function_decl;
}

// Create a `clang::FunctionDecl` for the given Carbon function. This
// can be used to call the Carbon function from C++. The Carbon
// function's ABI must be compatible with C++.
//
// The resulting decl is used to allow a generated C++ function to call
// a generated Carbon function.
static auto BuildCppFunctionDeclForNonGenericCarbonFn(Context& context,
                                                      SemIR::LocId loc_id,
                                                      FunctionInfo target)
    -> clang::FunctionDecl* {
  CARBON_CHECK(!target.function.generic_id.has_value());

  // Get parameters types.
  llvm::SmallVector<clang::QualType> cpp_param_types;
  if (target.self_param) {
    auto cpp_type = MapToCppThunkParamType(context, target.self_param->type_id);
    if (cpp_type.isNull()) {
      context.TODO(loc_id, "failed to map Carbon self type to C++");
      return nullptr;
    }
    cpp_param_types.push_back(cpp_type);
  }
  // For constructors, the first Carbon parameter is the object being
  // constructed, which is not explicitly declared in C++.
  llvm::ArrayRef<FunctionInfo::Param> params_to_map = target.explicit_params;
  if (target.export_as_constructor) {
    params_to_map = params_to_map.drop_front();
  }
  for (auto param : params_to_map) {
    auto cpp_type = MapToCppThunkParamType(context, param.type_id);
    if (cpp_type.isNull()) {
      context.TODO(loc_id, "failed to map Carbon type to C++");
      return nullptr;
    }
    cpp_param_types.push_back(cpp_type);
  }

  CARBON_CHECK(target.function.return_type_inst_id == SemIR::TypeInstId::None);
  auto cpp_return_type = context.ast_context().VoidTy;

  auto* decl_context = target.export_as_constructor
                           ? target.decl_context
                           : context.ast_context().getTranslationUnitDecl();
  auto* function_decl = BuildCppFunctionDecl(
      context, decl_context, loc_id, target.GetCppName(context),
      cpp_param_types, cpp_return_type, target.export_as_constructor);

  // Mangle the function name and attach it to the `FunctionDecl`.
  SemIR::Mangler m(context.sem_ir(), context.total_ir_count(),
                   context.mangle_string_fingerprint());
  std::string mangled_name =
      m.MangleWithPlatform(target.function_id, SemIR::SpecificId::None);
  function_decl->addAttr(
      clang::AsmLabelAttr::Create(context.ast_context(), mangled_name));

  return function_decl;
}

// Create a `clang::FunctionDecl` for the given generic Carbon function.
//
// The `clang::FunctionDecl` created here is only used as a function template
// decl. Only specializations of this function template decl are called
// directly, so the ABI of this function decl is irrelevant.
static auto BuildCppFunctionDeclForGenericCarbonFn(Context& context,
                                                   SemIR::LocId loc_id,
                                                   FunctionInfo callee)
    -> clang::FunctionDecl* {
  CARBON_CHECK(callee.function.generic_id.has_value());

  // Get parameters types.
  //
  // TODO: currently this matches the behavior of
  // BuildCppFunctionDeclForNonGenericCarbonFn, but for templates the ABI is
  // irrelevant, and the parameter should instead map to something that will
  // guide C++ template argument deduction into doing the right thing.
  llvm::SmallVector<clang::QualType> cpp_param_types;
  if (callee.self_param) {
    auto cpp_type = MapToCppThunkParamType(context, callee.self_param->type_id);
    if (cpp_type.isNull()) {
      context.TODO(loc_id, "failed to map Carbon self type to C++");
      return nullptr;
    }
    cpp_param_types.push_back(cpp_type);
  }
  for (auto param : callee.explicit_params) {
    auto cpp_type = MapToCppThunkParamType(context, param.type_id);
    if (cpp_type.isNull()) {
      context.TODO(loc_id, "failed to map Carbon type to C++");
      return nullptr;
    }
    cpp_param_types.push_back(cpp_type);
  }

  clang::QualType cpp_return_type = context.ast_context().VoidTy;
  if (callee.return_type_id.has_value()) {
    cpp_return_type = MapToCppType(context, callee.return_type_id);
    if (cpp_return_type.isNull()) {
      context.TODO(loc_id, "failed to map Carbon return type to C++");
      return nullptr;
    }
  }

  // TODO: provide the decl context corresponding to the Carbon generic
  // function.
  auto* decl_context = callee.export_as_constructor
                           ? callee.decl_context
                           : context.ast_context().getTranslationUnitDecl();
  return BuildCppFunctionDecl(context, decl_context, loc_id,
                              callee.GetCppName(context), cpp_param_types,
                              cpp_return_type, callee.export_as_constructor);
}

// Returns whether the given Carbon parameter should be passed as a C++ const
// reference.
static auto PassAsConstRef(Context& /*context*/,
                           const FunctionInfo::Param& param,
                           clang::QualType cpp_type) -> bool {
  // Use pass-by-const-ref for value parameters of array type.
  // TODO: Should we do this for value parameters of any type that uses a
  // pointer value representation?
  return param.kind == ParamPatternKind::Value && cpp_type->isArrayType();
}

// Converts a Carbon parameter type to the parameter type that should be exposed
// to C++ callers.
static auto MapToCppParamType(Context& context, SemIR::LocId loc_id,
                              const FunctionInfo::Param& param)
    -> clang::QualType {
  auto cpp_type = MapToCppType(context, param.type_id);
  if (cpp_type.isNull()) {
    return clang::QualType();
  }
  if (param.kind == Check::ParamPatternKind::Ref) {
    cpp_type = context.ast_context().getLValueReferenceType(cpp_type);
  } else if (PassAsConstRef(context, param, cpp_type)) {
    cpp_type = context.ast_context().getLValueReferenceType(
        context.ast_context().getConstType(cpp_type));
  } else if (cpp_type->isArrayType()) {
    // C++ doesn't support passing arrays by value.
    context.TODO(loc_id, "by-var array parameter");
    return clang::QualType();
  }
  return cpp_type;
}

// Returns the C++ function type (`clang::FunctionProtoType`) to use for a C++
// thunk calling a Carbon function.
static auto BuildCppToCarbonThunkFunctionType(Context& context,
                                              SemIR::LocId loc_id,
                                              const FunctionInfo& target)
    -> const clang::FunctionProtoType* {
  llvm::SmallVector<clang::QualType> thunk_param_types;
  thunk_param_types.reserve(target.explicit_params.size());
  for (auto param : target.explicit_params) {
    auto cpp_type = MapToCppParamType(context, loc_id, param);
    if (cpp_type.isNull()) {
      context.TODO(loc_id, "failed to map C++ type to Carbon");
      return nullptr;
    }
    thunk_param_types.push_back(cpp_type);
  }

  // Get the C++ return type (this corresponds to the return type of the
  // target Carbon function).
  clang::QualType cpp_return_type = context.ast_context().VoidTy;
  if (!target.export_as_constructor &&
      (target.return_type_id != SemIR::TypeId::None)) {
    cpp_return_type = MapToCppType(context, target.return_type_id);
    if (cpp_return_type.isNull()) {
      context.TODO(loc_id, "failed to map Carbon return type to C++ type");
      return nullptr;
    }
    if (cpp_return_type->isArrayType()) {
      // C++ doesn't support returning arrays by value.
      context.TODO(loc_id, "array return type");
      return nullptr;
    }
  }

  auto ext_proto_info = clang::FunctionProtoType::ExtProtoInfo();
  if (target.self_param) {
    if (target.self_param->kind == ParamPatternKind::Ref) {
      ext_proto_info.RefQualifier = clang::RQ_LValue;
    } else {
      // A method with `self` doesn't modify the object, so export it as
      // `const`. Unlike `ref self`, `self` doesn't require a reference
      // expression, so no ref-qualifier is added.
      ext_proto_info.TypeQuals.addConst();
    }
  }
  return context.ast_context()
      .getFunctionType(cpp_return_type, thunk_param_types, ext_proto_info)
      ->getAs<clang::FunctionProtoType>();
}

// Create the declaration of the C++ thunk.
static auto BuildCppToCarbonThunkDecl(Context& context, SemIR::LocId loc_id,
                                      const FunctionInfo& target,
                                      clang::DeclarationName thunk_name)
    -> clang::FunctionDecl* {
  clang::ASTContext& ast_context = context.ast_context();

  auto clang_loc = GetCppLocation(context, loc_id);

  // If the signature was imported from C++, use that declaration to form the
  // parameter types rather than (lossily) re-exporting the Carbon signature
  // back to C++.
  const clang::FunctionProtoType* thunk_function_type = nullptr;
  if (auto thunk_id = target.function.thunk_id(); thunk_id.has_value()) {
    const auto& thunk = context.thunks().Get(thunk_id);
    const auto& thunk_signature = context.functions().Get(thunk.signature_id);
    if (const auto* clang_decl =
            context.clang_decls().Lookup(thunk_signature.first_decl_id())) {
      thunk_function_type = cast<clang::FunctionDecl>(clang_decl->decl())
                                ->getType()
                                ->getAs<clang::FunctionProtoType>();
    }
  }
  if (!thunk_function_type) {
    thunk_function_type =
        BuildCppToCarbonThunkFunctionType(context, loc_id, target);
    if (!thunk_function_type) {
      return nullptr;
    }
  }

  clang::DeclarationNameInfo name_info(thunk_name, clang_loc);
  clang::QualType thunk_qual_type(thunk_function_type, 0);
  auto* tinfo =
      ast_context.getTrivialTypeSourceInfo(thunk_qual_type, clang_loc);

  bool uses_fp_intrin = false;
  bool inline_specified = true;
  auto constexpr_kind = clang::ConstexprSpecKind::Unspecified;
  auto trailing_requires_clause = clang::AssociatedConstraint();

  clang::FunctionDecl* thunk_function_decl = nullptr;
  if (auto* parent_class =
          dyn_cast<clang::CXXRecordDecl>(target.decl_context)) {
    if (target.export_as_constructor) {
      thunk_function_decl = clang::CXXConstructorDecl::Create(
          ast_context, parent_class, clang_loc, name_info, thunk_qual_type,
          tinfo,
          clang::ExplicitSpecifier{nullptr,
                                   clang::ExplicitSpecKind::ResolvedTrue},
          uses_fp_intrin, inline_specified, /* isImplicitlyDeclared= */ false,
          constexpr_kind);
    } else {
      thunk_function_decl = clang::CXXMethodDecl::Create(
          ast_context, parent_class, clang_loc, name_info, thunk_qual_type,
          tinfo, target.GetStorageClass(), uses_fp_intrin, inline_specified,
          constexpr_kind, clang_loc, trailing_requires_clause);
    }
    // TODO: Map Carbon access to C++ access.
    thunk_function_decl->setAccess(clang::AS_public);
    // Carbon overriders are non-virtual in C++; only the corresponding thunk is
    // virtual.
    thunk_function_decl->setVirtualAsWritten(
        target.function.virtual_modifier !=
            SemIR::Function::VirtualModifier::None &&
        target.function.virtual_modifier !=
            SemIR::Function::VirtualModifier::Override);
    if (target.function.virtual_modifier ==
        SemIR::Function::VirtualModifier::Abstract) {
      cast<clang::CXXMethodDecl>(thunk_function_decl)->setIsPureVirtual(true);
    }
  } else {
    thunk_function_decl = clang::FunctionDecl::Create(
        ast_context, target.decl_context, clang_loc, name_info, thunk_qual_type,
        tinfo, clang::SC_None, uses_fp_intrin, inline_specified,
        /*hasWrittenPrototype=*/true, constexpr_kind, trailing_requires_clause);
  }

  llvm::SmallVector<clang::ParmVarDecl*> param_var_decls;
  for (auto [i, type] : llvm::enumerate(thunk_function_type->param_types())) {
    clang::ParmVarDecl* thunk_param = clang::ParmVarDecl::Create(
        ast_context, thunk_function_decl, /*StartLoc=*/clang_loc,
        /*IdLoc=*/clang_loc, /*Id=*/nullptr, type,
        /*TInfo=*/nullptr, clang::SC_None, /*DefArg=*/nullptr);
    param_var_decls.push_back(thunk_param);
  }
  thunk_function_decl->setParams(param_var_decls);
  target.decl_context->addHiddenDecl(thunk_function_decl);

  // Force the thunk to be inlined and discarded.
  thunk_function_decl->addAttr(
      clang::AlwaysInlineAttr::CreateImplicit(ast_context));
  thunk_function_decl->addAttr(
      clang::InternalLinkageAttr::CreateImplicit(ast_context));

  return thunk_function_decl;
}

// Get an expr for accessing `this` in a method.
static auto GetThisArg(clang::Sema& sema, clang::SourceLocation clang_loc,
                       const clang::CXXMethodDecl* method_decl)
    -> clang::Expr* {
  // These pick up the method's `const` qualifier, if any.
  clang::QualType class_type = method_decl->getFunctionObjectParameterType();
  auto* this_expr = sema.BuildCXXThisExpr(clang_loc, method_decl->getThisType(),
                                          /*IsImplicit=*/true);
  return clang::UnaryOperator::Create(
      sema.getASTContext(), this_expr, clang::UO_Deref, class_type,
      clang::ExprValueKind::VK_LValue, clang::ExprObjectKind::OK_Ordinary,
      clang_loc, /*CanOverflow=*/false, clang::FPOptionsOverride());
}

// Create the body of a C++ thunk that calls a Carbon thunk. The
// arguments are passed by reference to the callee.
static auto BuildCppToCarbonThunkBody(Context& context,
                                      const FunctionInfo& target,
                                      clang::FunctionDecl* function_decl,
                                      clang::FunctionDecl* callee_function_decl)
    -> clang::StmtResult {
  clang::Sema& sema = context.clang_sema();
  clang::SourceLocation clang_loc = function_decl->getLocation();

  llvm::SmallVector<clang::Stmt*> stmts;

  // Create return storage if the target function returns non-void.
  const bool has_return_value = !function_decl->getReturnType()->isVoidType();
  clang::VarDecl* return_storage_var_decl = nullptr;
  clang::ExprResult return_storage_expr;
  if (has_return_value) {
    CARBON_CHECK(!target.export_as_constructor);
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

  llvm::SmallVector<clang::Expr*> call_args;
  // For methods, pass the `this` pointer as the first argument to the callee.
  if (target.self_param) {
    call_args.push_back(
        GetThisArg(sema, clang_loc, cast<clang::CXXMethodDecl>(function_decl)));
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

  if (target.export_as_constructor) {
    auto* class_decl = cast<clang::CXXRecordDecl>(target.decl_context);
    clang::QualType class_type =
        sema.getASTContext().getCanonicalTagType(class_decl);
    auto* callee_ctor_decl =
        llvm::cast<clang::CXXConstructorDecl>(callee_function_decl);
    llvm::SmallVector<clang::Expr*> converted_args;
    if (sema.CompleteConstructorCall(callee_ctor_decl, class_type, call_args,
                                     clang_loc, converted_args,
                                     /*AllowExplicit=*/true)) {
      CARBON_FATAL("CompleteConstructorCall failed");
    }
    auto call = sema.BuildCXXConstructExpr(
        clang_loc, class_type, callee_ctor_decl, /*Elidable=*/false,
        converted_args,
        /*HadMultipleCandidates=*/true, /*IsListInitialization=*/false,
        /*IsStdInitListInitialization=*/false,
        /*RequiresZeroInit=*/false, clang::CXXConstructionKind::Delegating,
        clang::SourceRange(clang_loc, clang_loc));
    auto* tinfo =
        context.ast_context().getTrivialTypeSourceInfo(class_type, clang_loc);
    auto* ctor_initializer =
        new (context.ast_context()) clang::CXXCtorInitializer(
            context.ast_context(), tinfo, clang_loc, call.get(), clang_loc);
    CARBON_CHECK(call.isUsable());
    sema.SetDelegatingInitializer(
        llvm::cast<clang::CXXConstructorDecl>(function_decl), ctor_initializer);
  } else {
    clang::ExprResult callee = sema.BuildDeclRefExpr(
        callee_function_decl, callee_function_decl->getType(),
        clang::VK_PRValue, clang_loc);
    clang::ExprResult call = sema.BuildCallExpr(
        nullptr, callee.get(), clang_loc, call_args, clang_loc);
    CARBON_CHECK(call.isUsable());
    stmts.push_back(call.get());

    if (has_return_value) {
      auto* return_stmt = clang::ReturnStmt::Create(
          sema.getASTContext(), clang_loc, return_storage_expr.get(),
          return_storage_var_decl);
      stmts.push_back(return_stmt);
    }
  }
  return clang::CompoundStmt::Create(sema.getASTContext(), stmts,
                                     clang::FPOptionsOverride(), clang_loc,
                                     clang_loc);
}

// Create a Carbon thunk that calls `callee`. The thunk's parameters are
// all references to the callee parameter type.
//
// `extra_name` will be appended to the thunk name. This is used to
// disambiguate the names of specialized function thunks.
static auto BuildCarbonToCarbonThunk(Context& context, SemIR::LocId loc_id,
                                     const FunctionInfo& target,
                                     std::string_view extra_name = "")
    -> FunctionInfo {
  // Create the thunk's name.
  llvm::SmallString<64> thunk_name =
      context.names().GetFormatted(target.function.name_id);
  thunk_name += "__carbon_thunk";
  thunk_name += extra_name;
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
  if (target.return_type_id != SemIR::TypeId::None) {
    thunk_param_type_ids.push_back(target.return_type_id);
  }

  // If this thunk will be exposed as a C++ constructor, we put the output
  // parameter first to match the Itanium constructor ABI.
  //
  // TODO: use `clang::CodeGen::CGCXXABI::HasThisReturn` to determine if the
  // constructor's `this` should be a return value instead of an output param.
  if (target.export_as_constructor) {
    CARBON_CHECK(target.return_type_id != SemIR::TypeId::None);
    std::rotate(thunk_param_type_ids.begin(), thunk_param_type_ids.end() - 1,
                thunk_param_type_ids.end());
  }

  auto carbon_thunk_function_id =
      MakeGeneratedFunctionDecl(
          context, loc_id,
          {.parent_scope_id = target.function.parent_scope_id,
           .name_id = thunk_name_id,
           .self_type_id = target.GetSelfTypeId(),
           .param_type_ids = thunk_param_type_ids,
           .param_kind = ParamPatternKind::Ref})
          .second;

  BuildThunkDefinitionForExport(
      context, carbon_thunk_function_id, target.function_id,
      context.functions().Get(carbon_thunk_function_id).first_decl_id(),
      target.function.first_decl_id(), target.export_as_constructor);

  return FunctionInfo(context, carbon_thunk_function_id,
                      context.functions().Get(carbon_thunk_function_id),
                      target.decl_context, target.export_as_constructor);
}

static auto ExportNonGenericFunctionDeclToCpp(Context& context,
                                              SemIR::LocId loc_id,
                                              const FunctionInfo& target)
    -> clang::FunctionDecl* {
  return BuildCppToCarbonThunkDecl(context, loc_id, target,
                                   target.GetCppName(context));
}

auto ExportVirtualFunctionDeclToCpp(Context& context, SemIR::LocId loc_id,
                                    clang::CXXRecordDecl* parent,
                                    SemIR::FunctionId function_id)
    -> clang::CXXMethodDecl* {
  FunctionInfo target(context, function_id,
                      context.functions().Get(function_id), parent,
                      /*export_as_constructor=*/false);
  return cast_or_null<clang::CXXMethodDecl>(
      ExportNonGenericFunctionDeclToCpp(context, loc_id, target));
}

static auto BuildCppToCarbonThunk(Context& context, SemIR::LocId loc_id,
                                  const FunctionInfo& target,
                                  clang::FunctionDecl* thunk_function_decl,
                                  std::string_view extra_name) -> void {
  // Create a Carbon thunk that calls the callee. The thunk's parameters
  // are all references so that the ABI is compatible with C++ callers.
  auto carbon_thunk_target =
      BuildCarbonToCarbonThunk(context, loc_id, target, extra_name);

  // Create a `clang::FunctionDecl` that can be used to call the Carbon thunk.
  auto* carbon_function_decl = BuildCppFunctionDeclForNonGenericCarbonFn(
      context, loc_id, carbon_thunk_target);
  if (!carbon_function_decl) {
    return;
  }

  // Build the thunk function body.
  clang::Sema& sema = context.clang_sema();
  clang::Sema::ContextRAII context_raii(sema, thunk_function_decl);
  // Ensure that the evaluation context is not `Unevaluated`, as that
  // would cause code generation to fail.
  clang::EnterExpressionEvaluationContext evaluated(
      sema, clang::Sema::ExpressionEvaluationContext::PotentiallyEvaluated);
  sema.ActOnStartOfFunctionDef(nullptr, thunk_function_decl);
  clang::StmtResult body = BuildCppToCarbonThunkBody(
      context, target, thunk_function_decl, carbon_function_decl);
  sema.ActOnFinishFunctionBody(thunk_function_decl, body.get());
  CARBON_CHECK(!body.isInvalid());

  context.clang_sema().getASTConsumer().HandleTopLevelDecl(
      clang::DeclGroupRef(thunk_function_decl));
}

auto DefineExportedVirtualFunction(Context& context, SemIR::LocId loc_id,
                                   SemIR::FunctionId callee_function_id,
                                   clang::CXXMethodDecl* method_decl) -> void {
  const SemIR::Function& callee = context.functions().Get(callee_function_id);
  FunctionInfo target_function_info(context, callee_function_id, callee,
                                    method_decl->getDeclContext(),
                                    /*export_as_constructor=*/false);
  BuildCppToCarbonThunk(context, loc_id, target_function_info, method_decl, "");
}

// Creates a `clang::FunctionDecl` that calls the Carbon function in
// `target`. The `extra_name` string is appended to the Carbon thunk's
// name.
//
// Returns nullptr if an error occurs.
auto ExportNonGenericFunctionToCpp(Context& context, SemIR::LocId loc_id,
                                   const FunctionInfo& target,
                                   std::string_view extra_name = "")
    -> clang::FunctionDecl* {
  auto* thunk_function_decl =
      ExportNonGenericFunctionDeclToCpp(context, loc_id, target);
  if (!thunk_function_decl) {
    return nullptr;
  }

  BuildCppToCarbonThunk(context, loc_id, target, thunk_function_decl,
                        extra_name);
  return thunk_function_decl;
}

auto ExportFunctionSpecializationToCpp(
    Context& context, clang::FunctionTemplateDecl* function_template_decl,
    llvm::ArrayRef<clang::TemplateArgument> template_args) -> bool {
  // Map from the `clang::FunctionTemplateDecl` to the Carbon `FunctionDecl`.
  auto clang_decl_id = context.clang_decls().LookupId(
      SemIR::ClangDeclKey(function_template_decl));
  if (clang_decl_id == SemIR::ClangDeclId::None) {
    return false;
  }
  SemIR::InstId inst_id = context.clang_decls().Get(clang_decl_id).inst_id;
  CARBON_CHECK(inst_id.has_value());
  auto target_function_decl =
      context.insts().GetAs<SemIR::FunctionDecl>(inst_id);
  auto target_function =
      context.functions().Get(target_function_decl.function_id);

  auto* decl_context = function_template_decl->getDeclContext();
  FunctionInfo target(context, target_function_decl.function_id,
                      target_function, decl_context,
                      llvm::isa<clang::CXXConstructorDecl>(
                          function_template_decl->getTemplatedDecl()));
  SemIR::LocId loc_id(target.function.first_decl_id());

  // Create a specific, and use that to convert return type and
  // parameters with symbolic types to concrete types.
  auto specific_id = MakeSpecificForTemplateArgs(
      context, loc_id, target.function.generic_id, template_args);
  if (specific_id == SemIR::SpecificId::None) {
    return false;
  }
  // This name is appended to the thunk name to disambiguate between
  // specializations.
  SemIR::Mangler m(context.sem_ir(), context.total_ir_count(),
                   context.mangle_string_fingerprint());
  auto extra_name = m.MangleSpecificId(specific_id);
  target.return_type_id =
      target.function.GetDeclaredReturnType(context.sem_ir(), specific_id);
  for (auto& param : target.explicit_params) {
    param.type_id =
        GetScrutineeTypeInSpecific(context, param.pattern_inst_id, specific_id);
  }

  // Build the thunks. Mark the C++ thunk as a template specialization.
  auto* function_decl =
      ExportNonGenericFunctionToCpp(context, loc_id, target, extra_name);
  if (!function_decl) {
    return false;
  }
  auto* template_arg_list = clang::TemplateArgumentList::CreateCopy(
      context.ast_context(), template_args);
  function_decl->setFunctionTemplateSpecialization(
      function_template_decl, template_arg_list,
      /*InsertPos=*/nullptr, clang::TSK_ExplicitSpecialization,
      /*TemplateArgsAsWritten=*/nullptr,
      /*PointOfInstantiation=*/clang::SourceLocation());

  return true;
}

// Creates a `clang::FunctionTemplateDecl` for a generic Carbon function.
//
// Returns nullptr if an error occurs.
static auto ExportGenericFunctionToCpp(Context& context, SemIR::LocId loc_id,
                                       const FunctionInfo& callee)
    -> clang::FunctionTemplateDecl* {
  auto clang_loc = GetCppLocation(context, loc_id);

  auto* template_param_list = ExportGenericBindings(
      context, loc_id, callee.function.generic_id, callee.decl_context);
  if (!template_param_list) {
    return nullptr;
  }

  auto* function_decl =
      BuildCppFunctionDeclForGenericCarbonFn(context, loc_id, callee);
  if (!function_decl) {
    return nullptr;
  }

  auto* template_decl = clang::FunctionTemplateDecl::Create(
      context.ast_context(), callee.decl_context, clang_loc,
      function_decl->getDeclName(), template_param_list, function_decl);
  function_decl->setDescribedFunctionTemplate(template_decl);

  return template_decl;
}

auto ExportFunctionToCpp(Context& context, SemIR::LocId loc_id,
                         SemIR::FunctionId callee_function_id)
    -> clang::NamedDecl* {
  auto target = BuildFunctionInfo(context, loc_id, callee_function_id);
  if (!target) {
    return nullptr;
  }

  if (target->function.generic_id.has_value()) {
    if (target->export_as_constructor || target->self_param.has_value()) {
      context.TODO(loc_id, "support exporting generic member functions");
      return nullptr;
    }
    return ExportGenericFunctionToCpp(context, loc_id, *target);
  }

  return ExportNonGenericFunctionToCpp(context, loc_id, *target);
}

// Returns whether the given class has any abstract methods.
static auto HasAnyAbstractMethods(Context& context,
                                  const SemIR::Class& class_info,
                                  SemIR::SpecificId class_specific_id) -> bool {
  if (class_info.vtable_decl_id == SemIR::InstId::None) {
    return false;
  }

  LoadImportRef(context, class_info.vtable_decl_id);
  auto vtable_decl_const_id = GetConstantValueInSpecific(
      context.sem_ir(), class_specific_id, class_info.vtable_decl_id);
  if (vtable_decl_const_id == SemIR::ErrorInst::ConstantId) {
    return false;
  }
  auto vtable_id = context.constant_values()
                       .GetInstAs<SemIR::VtableDecl>(vtable_decl_const_id)
                       .vtable_id;
  const auto& vtable = context.vtables().Get(vtable_id);
  for (auto virtual_fn_id :
       context.inst_blocks().Get(vtable.virtual_functions_id)) {
    auto virtual_fn = DecomposeVirtualFunction(context.sem_ir(), virtual_fn_id,
                                               class_specific_id);
    if (context.functions().Get(virtual_fn.function_id).virtual_modifier ==
        SemIR::Function::VirtualModifier::Abstract) {
      return true;
    }
  }
  return false;
}

auto ExportDestructorToCpp(Context& context, const SemIR::Class& class_info,
                           clang::CXXRecordDecl* record_decl)
    -> clang::CXXDestructorDecl* {
  SemIR::LocId loc_id(class_info.first_decl_id());
  auto clang_loc = record_decl->getLocation();

  // TODO: Add support for exporting specific classes.
  const auto specific_id = SemIR::SpecificId::None;

  // Create C++ destructor decl.
  auto class_type = context.ast_context().getCanonicalTagType(record_decl);
  auto name =
      context.ast_context().DeclarationNames.getCXXDestructorName(class_type);
  clang::DeclarationNameInfo name_info(name, clang_loc);
  clang::QualType type = context.ast_context().getFunctionType(
      context.ast_context().VoidTy, llvm::ArrayRef<clang::QualType>(),
      clang::FunctionProtoType::ExtProtoInfo().withExceptionSpec(
          clang::EST_BasicNoexcept));
  auto* cpp_destructor_decl = clang::CXXDestructorDecl::Create(
      context.ast_context(), record_decl,
      /*StartLoc=*/clang_loc, name_info, type, /*TInfo=*/nullptr,
      /*UsesFPIntrin=*/false, /*isInline=*/true, /*isImplicitlyDeclared=*/true,
      clang::ConstexprSpecKind::Unspecified);
  cpp_destructor_decl->setAccess(clang::AS_public);

  clang::Sema& sema = context.clang_sema();

  // Find and register any base class virtual destructors that this destructor
  // overrides. This marks the destructor as implicitly virtual if needed.
  sema.AddOverriddenMethods(record_decl, cpp_destructor_decl);

  // If the class is abstract and has no abstract methods, we need to mark the
  // destructor as pure virtual.
  if (class_info.inheritance_kind == SemIR::Class::InheritanceKind::Abstract &&
      !HasAnyAbstractMethods(context, class_info, specific_id)) {
    if (cpp_destructor_decl->isVirtual()) {
      cpp_destructor_decl->setIsPureVirtual(true);
    } else {
      context.TODO(class_info.definition_id,
                   "exporting abstract class with no abstract methods and "
                   "non-virtual destructor to C++");
    }
  }

  // Create Carbon thunk that destroys the object, and get a C++
  // function decl for calling it.
  // TODO: Once we support exporting specific classes, export the specific
  // destructor here rather than a generic one.
  auto thunk_function_id = BuildDestroyThunk(context, loc_id, class_info);
  FunctionInfo thunk_target(context, thunk_function_id,
                            context.functions().Get(thunk_function_id),
                            record_decl, /*export_as_constructor=*/false);
  auto* cpp_function_decl =
      BuildCppFunctionDeclForNonGenericCarbonFn(context, loc_id, thunk_target);
  if (!cpp_function_decl) {
    return nullptr;
  }

  // Build the destructor body.
  clang::Sema::ContextRAII context_raii(sema, cpp_destructor_decl);
  sema.ActOnStartOfFunctionDef(nullptr, cpp_destructor_decl);

  // Create a clang call expr to call the Carbon thunk.
  clang::ExprResult callee =
      sema.BuildDeclRefExpr(cpp_function_decl, cpp_function_decl->getType(),
                            clang::VK_PRValue, clang_loc);
  llvm::SmallVector<clang::Expr*> call_args;
  call_args.push_back(GetThisArg(sema, clang_loc, cpp_destructor_decl));
  clang::ExprResult call = sema.BuildCallExpr(nullptr, callee.get(), clang_loc,
                                              call_args, clang_loc);

  sema.ActOnFinishFunctionBody(cpp_destructor_decl, call.get());

  return cpp_destructor_decl;
}

auto ExportVarToCpp(Context& context, SemIR::InstId inst_id,
                    SemIR::VarStorage var_storage) -> clang::VarDecl* {
  // Check if the variable was already exported and return the existing
  // `VarDecl` if so. Note that the `pattern_id` is used as the key
  // rather than the `InstId` for the `VarStorage`.
  if (const auto* clang_decl =
          context.clang_decls().Lookup(var_storage.pattern_id)) {
    return cast<clang::VarDecl>(clang_decl->decl());
  }

  // Look up the entity name and check the scope.
  auto entity_name_id = GetFirstBindingNameFromPatternId(
      context.sem_ir(), var_storage.pattern_id);
  const auto& entity_name = context.entity_names().Get(entity_name_id);
  const auto& name_scope =
      context.name_scopes().Get(entity_name.parent_scope_id);
  auto scope_inst = context.insts().Get(name_scope.inst_id());
  CARBON_CHECK(scope_inst.Is<SemIR::Namespace>() ||
               scope_inst.Is<SemIR::ClassDecl>());

  // Map the parent scope into the C++ AST.
  SemIR::LocId loc_id(inst_id);
  auto* decl_context =
      ExportNameScopeToCpp(context, loc_id, entity_name.parent_scope_id);
  if (!decl_context) {
    return nullptr;
  }

  // Map the type.
  auto cpp_type = MapToCppType(context, var_storage.type_id);
  if (cpp_type.isNull()) {
    context.TODO(loc_id, "failed to map Carbon type to C++");
    return nullptr;
  }

  // Create the `clang::VarDecl` and add it to `clang_decls()`.
  auto clang_loc = GetCppLocation(context, loc_id);
  auto* identifier_info = GetClangIdentifierInfo(context, entity_name.name_id);
  auto* var_decl = clang::VarDecl::Create(
      context.ast_context(), decl_context,
      /*StartLoc=*/clang_loc, /*IdLoc=*/clang_loc, identifier_info, cpp_type,
      /*TInfo=*/nullptr, clang::SC_Extern);
  context.clang_decls().Add(
      {.key = SemIR::ClangDeclKey::ForNonFunctionDecl(var_decl),
       .inst_id = var_storage.pattern_id,
       .var_storage_inst_id = inst_id});

  if (scope_inst.Is<SemIR::ClassDecl>()) {
    SetCppClassMemberAccess(name_scope, entity_name.name_id, var_decl);
  }

  // Set the Carbon mangled variable name.
  // TODO: do we need to apply the platform mangling, like we do for exported
  // functions?
  SemIR::Mangler m(context.sem_ir(), context.total_ir_count(),
                   context.mangle_string_fingerprint());
  std::string mangled_name = m.MangleGlobalVariable(var_storage.pattern_id);
  var_decl->addAttr(
      clang::AsmLabelAttr::Create(context.ast_context(), mangled_name));

  return var_decl;
}

}  // namespace Carbon::Check
