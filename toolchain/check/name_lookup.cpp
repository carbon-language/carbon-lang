// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/name_lookup.h"

#include "toolchain/check/generic.h"
#include "toolchain/check/import.h"
#include "toolchain/check/import_cpp.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/name_scope.h"

namespace Carbon::Check {

auto AddNameToLookup(Context& context, SemIR::NameId name_id,
                     SemIR::InstId target_id, ScopeIndex scope_index) -> void {
  if (auto existing = context.scope_stack().LookupOrAddName(name_id, target_id,
                                                            scope_index);
      existing.has_value()) {
    // TODO: Add coverage to this use case and use the location of the name
    // instead of the target.
    DiagnoseDuplicateName(context, name_id, target_id, existing);
  }
}

auto LookupNameInDecl(Context& context, SemIR::LocId loc_id,
                      SemIR::NameId name_id, SemIR::NameScopeId scope_id,
                      ScopeIndex scope_index) -> SemIR::ScopeLookupResult {
  if (!scope_id.has_value()) {
    // Look for a name in the specified scope or a scope nested within it only.
    // There are two cases where the name would be in an outer scope:
    //
    //  - The name is the sole component of the declared name:
    //
    //    class A;
    //    fn F() {
    //      class A;
    //    }
    //
    //    In this case, the inner A is not the same class as the outer A, so
    //    lookup should not find the outer A.
    //
    //  - The name is a qualifier of some larger declared name:
    //
    //    class A { class B; }
    //    fn F() {
    //      class A.B {}
    //    }
    //
    //    In this case, we're not in the correct scope to define a member of
    //    class A, so we should reject, and we achieve this by not finding the
    //    name A from the outer scope.
    //
    // There is also one case where the name would be in an inner scope:
    //
    //  - The name is redeclared by a parameter of the same entity:
    //
    //    fn F() {
    //      class C(C:! type);
    //    }
    //
    //    In this case, the class C is not a redeclaration of its parameter, but
    //    we find the parameter in order to diagnose a redeclaration error.
    return SemIR::ScopeLookupResult::MakeWrappedLookupResult(
        context.scope_stack().LookupInLexicalScopesWithin(name_id, scope_index),
        SemIR::AccessKind::Public);
  } else {
    // We do not look into `extend`ed scopes here. A qualified name in a
    // declaration must specify the exact scope in which the name was originally
    // introduced:
    //
    //    base class A { fn F(); }
    //    class B { extend base: A; }
    //
    //    // Error, no `F` in `B`.
    //    fn B.F() {}
    return LookupNameInExactScope(context, loc_id, name_id, scope_id,
                                  context.name_scopes().Get(scope_id),
                                  /*is_being_declared=*/true);
  }
}

auto LookupUnqualifiedName(Context& context, Parse::NodeId node_id,
                           SemIR::NameId name_id, bool required)
    -> LookupResult {
  // TODO: Check for shadowed lookup results.

  // Find the results from ancestor lexical scopes. These will be combined with
  // results from non-lexical scopes such as namespaces and classes.
  auto [lexical_result, non_lexical_scopes] =
      context.scope_stack().LookupInLexicalScopes(name_id);

  // Walk the non-lexical scopes and perform lookups into each of them.
  for (auto [index, lookup_scope_id, specific_id] :
       llvm::reverse(non_lexical_scopes)) {
    if (auto non_lexical_result =
            LookupQualifiedName(context, node_id, name_id,
                                LookupScope{.name_scope_id = lookup_scope_id,
                                            .specific_id = specific_id},
                                /*required=*/false);
        non_lexical_result.scope_result.is_found()) {
      // In an interface definition, replace associated entity `M` with
      // `Self.M` (where the `Self` is the `Self` of the interface).
      const auto& scope = context.name_scopes().Get(lookup_scope_id);
      if (scope.is_interface_definition()) {
        SemIR::InstId target_inst_id =
            non_lexical_result.scope_result.target_inst_id();
        if (auto assoc_type =
                context.types().TryGetAs<SemIR::AssociatedEntityType>(
                    context.insts().Get(target_inst_id).type_id())) {
          auto interface_decl =
              context.insts().GetAs<SemIR::InterfaceDecl>(scope.inst_id());
          const auto& interface =
              context.interfaces().Get(interface_decl.interface_id);
          SemIR::InstId result_inst_id =
              GetAssociatedValue(context, node_id, interface.self_param_id,
                                 target_inst_id, assoc_type->interface_type_id);
          non_lexical_result.scope_result = SemIR::ScopeLookupResult::MakeFound(
              result_inst_id, non_lexical_result.scope_result.access_kind());
        }
      }
      return non_lexical_result;
    }
  }

  if (lexical_result == SemIR::InstId::InitTombstone) {
    CARBON_DIAGNOSTIC(UsedBeforeInitialization, Error,
                      "`{0}` used before initialization", SemIR::NameId);
    context.emitter().Emit(node_id, UsedBeforeInitialization, name_id);
    return {.specific_id = SemIR::SpecificId::None,
            .scope_result = SemIR::ScopeLookupResult::MakeError()};
  }

  if (lexical_result.has_value()) {
    // A lexical scope never needs an associated specific. If there's a
    // lexically enclosing generic, then it also encloses the point of use of
    // the name.
    return {.specific_id = SemIR::SpecificId::None,
            .scope_result = SemIR::ScopeLookupResult::MakeFound(
                lexical_result, SemIR::AccessKind::Public)};
  }

  // We didn't find anything at all.
  if (required) {
    DiagnoseNameNotFound(context, node_id, name_id);
  }

  return {.specific_id = SemIR::SpecificId::None,
          .scope_result = SemIR::ScopeLookupResult::MakeError()};
}

auto LookupNameInExactScope(Context& context, SemIR::LocId loc_id,
                            SemIR::NameId name_id, SemIR::NameScopeId scope_id,
                            SemIR::NameScope& scope, bool is_being_declared)
    -> SemIR::ScopeLookupResult {
  if (auto entry_id = is_being_declared
                          ? scope.Lookup(name_id)
                          : scope.LookupOrPoison(loc_id, name_id)) {
    auto lookup_result = scope.GetEntry(*entry_id).result;
    if (!lookup_result.is_poisoned()) {
      LoadImportRef(context, lookup_result.target_inst_id());
    }
    return lookup_result;
  }

  if (!scope.import_ir_scopes().empty()) {
    // TODO: Enforce other access modifiers for imports.
    return SemIR::ScopeLookupResult::MakeWrappedLookupResult(
        ImportNameFromOtherPackage(context, loc_id, scope_id,
                                   scope.import_ir_scopes(), name_id),
        SemIR::AccessKind::Public);
  }

  if (scope.is_cpp_scope()) {
    SemIR::InstId imported_inst_id =
        ImportNameFromCpp(context, loc_id, scope_id, name_id);
    if (imported_inst_id.has_value()) {
      SemIR::ScopeLookupResult result = SemIR::ScopeLookupResult::MakeFound(
          imported_inst_id, SemIR::AccessKind::Public);
      // `ImportNameFromCpp()` can invalidate `scope`, so we do a scope lookup.
      context.name_scopes().Get(scope_id).AddRequired(
          {.name_id = name_id, .result = result});
      return result;
    }
  }

  return SemIR::ScopeLookupResult::MakeNotFound();
}

// Prints diagnostics on invalid qualified name access.
static auto DiagnoseInvalidQualifiedNameAccess(Context& context, SemIRLoc loc,
                                               SemIR::InstId scope_result_id,
                                               SemIR::NameId name_id,
                                               SemIR::AccessKind access_kind,
                                               bool is_parent_access,
                                               AccessInfo access_info) -> void {
  auto class_type = context.insts().TryGetAs<SemIR::ClassType>(
      context.constant_values().GetInstId(access_info.constant_id));
  if (!class_type) {
    return;
  }

  // TODO: Support scoped entities other than just classes.
  const auto& class_info = context.classes().Get(class_type->class_id);

  auto parent_type_id = class_info.self_type_id;

  if (access_kind == SemIR::AccessKind::Private && is_parent_access) {
    if (auto base_type_id =
            class_info.GetBaseType(context.sem_ir(), class_type->specific_id);
        base_type_id.has_value()) {
      parent_type_id = base_type_id;
    } else if (auto adapted_type_id = class_info.GetAdaptedType(
                   context.sem_ir(), class_type->specific_id);
               adapted_type_id.has_value()) {
      parent_type_id = adapted_type_id;
    } else {
      CARBON_FATAL("Expected parent for parent access");
    }
  }

  CARBON_DIAGNOSTIC(
      ClassInvalidMemberAccess, Error,
      "cannot access {0:private|protected} member `{1}` of type {2}",
      BoolAsSelect, SemIR::NameId, SemIR::TypeId);
  CARBON_DIAGNOSTIC(ClassMemberDeclaration, Note, "declared here");
  context.emitter()
      .Build(loc, ClassInvalidMemberAccess,
             access_kind == SemIR::AccessKind::Private, name_id, parent_type_id)
      .Note(scope_result_id, ClassMemberDeclaration)
      .Emit();
}

// Returns whether the access is prohibited by the access modifiers.
static auto IsAccessProhibited(std::optional<AccessInfo> access_info,
                               SemIR::AccessKind access_kind,
                               bool is_parent_access) -> bool {
  if (!access_info) {
    return false;
  }

  switch (access_kind) {
    case SemIR::AccessKind::Public:
      return false;
    case SemIR::AccessKind::Protected:
      return access_info->highest_allowed_access == SemIR::AccessKind::Public;
    case SemIR::AccessKind::Private:
      return access_info->highest_allowed_access !=
                 SemIR::AccessKind::Private ||
             is_parent_access;
  }
}

// Information regarding a prohibited access.
struct ProhibitedAccessInfo {
  // The resulting inst of the lookup.
  SemIR::InstId scope_result_id;
  // The access kind of the lookup.
  SemIR::AccessKind access_kind;
  // If the lookup is from an extended scope. For example, if this is a base
  // class member access from a class that extends it.
  bool is_parent_access;
};

auto AppendLookupScopesForConstant(Context& context, SemIR::LocId loc_id,
                                   SemIR::ConstantId base_const_id,
                                   llvm::SmallVector<LookupScope>* scopes)
    -> bool {
  auto base_id = context.constant_values().GetInstId(base_const_id);
  auto base = context.insts().Get(base_id);

  if (auto base_as_facet_access_type = base.TryAs<SemIR::FacetAccessType>()) {
    // Move from the symbolic facet value up in typish-ness to its FacetType to
    // find a lookup scope.
    auto facet_type_type_id =
        context.insts()
            .Get(base_as_facet_access_type->facet_value_inst_id)
            .type_id();
    base_const_id = context.types().GetConstantId(facet_type_type_id);
    base_id = context.constant_values().GetInstId(base_const_id);
    base = context.insts().Get(base_id);
  }

  if (auto base_as_namespace = base.TryAs<SemIR::Namespace>()) {
    scopes->push_back(
        LookupScope{.name_scope_id = base_as_namespace->name_scope_id,
                    .specific_id = SemIR::SpecificId::None});
    return true;
  }
  if (auto base_as_class = base.TryAs<SemIR::ClassType>()) {
    RequireCompleteType(
        context, context.types().GetTypeIdForTypeConstantId(base_const_id),
        loc_id, [&] {
          CARBON_DIAGNOSTIC(QualifiedExprInIncompleteClassScope, Error,
                            "member access into incomplete class {0}",
                            InstIdAsType);
          return context.emitter().Build(
              loc_id, QualifiedExprInIncompleteClassScope, base_id);
        });
    auto& class_info = context.classes().Get(base_as_class->class_id);
    scopes->push_back(LookupScope{.name_scope_id = class_info.scope_id,
                                  .specific_id = base_as_class->specific_id});
    return true;
  }
  if (auto base_as_facet_type = base.TryAs<SemIR::FacetType>()) {
    auto complete_id = RequireCompleteFacetType(
        context, context.types().GetTypeIdForTypeConstantId(base_const_id),
        loc_id, *base_as_facet_type, [&] {
          CARBON_DIAGNOSTIC(QualifiedExprInIncompleteFacetTypeScope, Error,
                            "member access into incomplete facet type {0}",
                            InstIdAsType);
          return context.emitter().Build(
              loc_id, QualifiedExprInIncompleteFacetTypeScope, base_id);
        });
    if (complete_id.has_value()) {
      const auto& resolved = context.complete_facet_types().Get(complete_id);
      for (const auto& interface : resolved.required_interfaces) {
        auto& interface_info = context.interfaces().Get(interface.interface_id);
        scopes->push_back({.name_scope_id = interface_info.scope_id,
                           .specific_id = interface.specific_id});
      }
    } else {
      // Lookup into this scope should fail without producing an error since
      // `RequireCompleteFacetType` has already issued a diagnostic.
      scopes->push_back(LookupScope{.name_scope_id = SemIR::NameScopeId::None,
                                    .specific_id = SemIR::SpecificId::None});
    }
    return true;
  }
  if (base_const_id == SemIR::ErrorInst::SingletonConstantId) {
    // Lookup into this scope should fail without producing an error.
    scopes->push_back(LookupScope{.name_scope_id = SemIR::NameScopeId::None,
                                  .specific_id = SemIR::SpecificId::None});
    return true;
  }
  // TODO: Per the design, if `base_id` is any kind of type, then lookup should
  // treat it as a name scope, even if it doesn't have members. For example,
  // `(i32*).X` should fail because there's no name `X` in `i32*`, not because
  // there's no name `X` in `type`.
  return false;
}

// Prints a diagnostic for a missing qualified name.
static auto DiagnoseMemberNameNotFound(
    Context& context, SemIRLoc loc, SemIR::NameId name_id,
    llvm::ArrayRef<LookupScope> lookup_scopes) -> void {
  if (lookup_scopes.size() == 1 &&
      lookup_scopes.front().name_scope_id.has_value()) {
    auto specific_id = lookup_scopes.front().specific_id;
    auto scope_inst_id = specific_id.has_value()
                             ? GetInstForSpecific(context, specific_id)
                             : context.name_scopes()
                                   .Get(lookup_scopes.front().name_scope_id)
                                   .inst_id();
    CARBON_DIAGNOSTIC(MemberNameNotFoundInScope, Error,
                      "member name `{0}` not found in {1}", SemIR::NameId,
                      InstIdAsType);
    context.emitter().Emit(loc, MemberNameNotFoundInScope, name_id,
                           scope_inst_id);
    return;
  }

  CARBON_DIAGNOSTIC(MemberNameNotFound, Error, "member name `{0}` not found",
                    SemIR::NameId);
  context.emitter().Emit(loc, MemberNameNotFound, name_id);
}

auto LookupQualifiedName(Context& context, SemIR::LocId loc_id,
                         SemIR::NameId name_id,
                         llvm::ArrayRef<LookupScope> lookup_scopes,
                         bool required, std::optional<AccessInfo> access_info)
    -> LookupResult {
  llvm::SmallVector<LookupScope> scopes(lookup_scopes);

  // TODO: Support reporting of multiple prohibited access.
  llvm::SmallVector<ProhibitedAccessInfo> prohibited_accesses;

  LookupResult result = {
      .specific_id = SemIR::SpecificId::None,
      .scope_result = SemIR::ScopeLookupResult::MakeNotFound()};
  bool has_error = false;
  bool is_parent_access = false;

  // Walk this scope and, if nothing is found here, the scopes it extends.
  while (!scopes.empty()) {
    auto [scope_id, specific_id] = scopes.pop_back_val();
    if (!scope_id.has_value()) {
      has_error = true;
      continue;
    }
    auto& name_scope = context.name_scopes().Get(scope_id);
    has_error |= name_scope.has_error();

    const SemIR::ScopeLookupResult scope_result =
        LookupNameInExactScope(context, loc_id, name_id, scope_id, name_scope);
    SemIR::AccessKind access_kind = scope_result.access_kind();

    auto is_access_prohibited =
        IsAccessProhibited(access_info, access_kind, is_parent_access);

    // Keep track of prohibited accesses, this will be useful for reporting
    // multiple prohibited accesses if we can't find a suitable lookup.
    if (is_access_prohibited) {
      prohibited_accesses.push_back({
          .scope_result_id = scope_result.target_inst_id(),
          .access_kind = access_kind,
          .is_parent_access = is_parent_access,
      });
    }

    if (!scope_result.is_found() || is_access_prohibited) {
      // If nothing is found in this scope or if we encountered an invalid
      // access, look in its extended scopes.
      const auto& extended = name_scope.extended_scopes();
      scopes.reserve(scopes.size() + extended.size());
      for (auto extended_id : llvm::reverse(extended)) {
        // Substitute into the constant describing the extended scope to
        // determine its corresponding specific.
        CARBON_CHECK(extended_id.has_value());
        LoadImportRef(context, extended_id);
        SemIR::ConstantId const_id = GetConstantValueInSpecific(
            context.sem_ir(), specific_id, extended_id);

        DiagnosticAnnotationScope annotate_diagnostics(
            &context.emitter(), [&](auto& builder) {
              CARBON_DIAGNOSTIC(FromExtendHere, Note,
                                "declared as an extended scope here");
              builder.Note(extended_id, FromExtendHere);
            });
        if (!AppendLookupScopesForConstant(context, loc_id, const_id,
                                           &scopes)) {
          // TODO: Handle case where we have a symbolic type and instead should
          // look in its type.
        }
      }
      is_parent_access |= !extended.empty();
      continue;
    }

    // If this is our second lookup result, diagnose an ambiguity.
    if (result.scope_result.is_found()) {
      CARBON_DIAGNOSTIC(
          NameAmbiguousDueToExtend, Error,
          "ambiguous use of name `{0}` found in multiple extended scopes",
          SemIR::NameId);
      context.emitter().Emit(loc_id, NameAmbiguousDueToExtend, name_id);
      // TODO: Add notes pointing to the scopes.
      return {.specific_id = SemIR::SpecificId::None,
              .scope_result = SemIR::ScopeLookupResult::MakeError()};
    }

    result.scope_result = scope_result;
    result.specific_id = specific_id;
  }

  if (required && !result.scope_result.is_found()) {
    if (!has_error) {
      if (prohibited_accesses.empty()) {
        DiagnoseMemberNameNotFound(context, loc_id, name_id, lookup_scopes);
      } else {
        //  TODO: We should report multiple prohibited accesses in case we don't
        //  find a valid lookup. Reporting the last one should suffice for now.
        auto [scope_result_id, access_kind, is_parent_access] =
            prohibited_accesses.back();

        // Note, `access_info` is guaranteed to have a value here, since
        // `prohibited_accesses` is non-empty.
        DiagnoseInvalidQualifiedNameAccess(context, loc_id, scope_result_id,
                                           name_id, access_kind,
                                           is_parent_access, *access_info);
      }
    }

    CARBON_CHECK(!result.scope_result.is_poisoned());
    return {.specific_id = SemIR::SpecificId::None,
            .scope_result = SemIR::ScopeLookupResult::MakeError()};
  }

  return result;
}

// Returns the scope of the Core package, or `None` if it's not found.
//
// TODO: Consider tracking the Core package in SemIR so we don't need to use
// name lookup to find it.
static auto GetCorePackage(Context& context, SemIR::LocId loc_id,
                           llvm::StringRef name) -> SemIR::NameScopeId {
  auto packaging = context.parse_tree().packaging_decl();
  if (packaging && packaging->names.package_id == PackageNameId::Core) {
    return SemIR::NameScopeId::Package;
  }
  auto core_name_id = SemIR::NameId::Core;

  // Look up `package.Core`.
  auto core_scope_result = LookupNameInExactScope(
      context, loc_id, core_name_id, SemIR::NameScopeId::Package,
      context.name_scopes().Get(SemIR::NameScopeId::Package));
  if (core_scope_result.is_found()) {
    // We expect it to be a namespace.
    if (auto namespace_inst = context.insts().TryGetAs<SemIR::Namespace>(
            core_scope_result.target_inst_id())) {
      // TODO: Decide whether to allow the case where `Core` is not a package.
      return namespace_inst->name_scope_id;
    }
  }

  CARBON_DIAGNOSTIC(
      CoreNotFound, Error,
      "`Core.{0}` implicitly referenced here, but package `Core` not found",
      std::string);
  context.emitter().Emit(loc_id, CoreNotFound, name.str());
  return SemIR::NameScopeId::None;
}

auto LookupNameInCore(Context& context, SemIR::LocId loc_id,
                      llvm::StringRef name) -> SemIR::InstId {
  auto core_package_id = GetCorePackage(context, loc_id, name);
  if (!core_package_id.has_value()) {
    return SemIR::ErrorInst::SingletonInstId;
  }

  auto name_id = SemIR::NameId::ForIdentifier(context.identifiers().Add(name));
  auto scope_result =
      LookupNameInExactScope(context, loc_id, name_id, core_package_id,
                             context.name_scopes().Get(core_package_id));
  if (!scope_result.is_found()) {
    CARBON_DIAGNOSTIC(
        CoreNameNotFound, Error,
        "name `Core.{0}` implicitly referenced here, but not found",
        SemIR::NameId);
    context.emitter().Emit(loc_id, CoreNameNotFound, name_id);
    return SemIR::ErrorInst::SingletonInstId;
  }

  // Look through import_refs and aliases.
  return context.constant_values().GetConstantInstId(
      scope_result.target_inst_id());
}

auto DiagnoseDuplicateName(Context& context, SemIR::NameId name_id,
                           SemIRLoc dup_def, SemIRLoc prev_def) -> void {
  CARBON_DIAGNOSTIC(NameDeclDuplicate, Error,
                    "duplicate name `{0}` being declared in the same scope",
                    SemIR::NameId);
  CARBON_DIAGNOSTIC(NameDeclPrevious, Note, "name is previously declared here");
  context.emitter()
      .Build(dup_def, NameDeclDuplicate, name_id)
      .Note(prev_def, NameDeclPrevious)
      .Emit();
}

auto DiagnosePoisonedName(Context& context, SemIR::NameId name_id,
                          SemIR::LocId poisoning_loc_id,
                          SemIR::LocId decl_name_loc_id) -> void {
  CARBON_CHECK(poisoning_loc_id.has_value(),
               "Trying to diagnose poisoned name with no poisoning location");
  CARBON_DIAGNOSTIC(NameUseBeforeDecl, Error,
                    "name `{0}` used before it was declared", SemIR::NameId);
  CARBON_DIAGNOSTIC(NameUseBeforeDeclNote, Note, "declared here");
  context.emitter()
      .Build(poisoning_loc_id, NameUseBeforeDecl, name_id)
      .Note(decl_name_loc_id, NameUseBeforeDeclNote)
      .Emit();
}

auto DiagnoseNameNotFound(Context& context, SemIRLoc loc, SemIR::NameId name_id)
    -> void {
  CARBON_DIAGNOSTIC(NameNotFound, Error, "name `{0}` not found", SemIR::NameId);
  context.emitter().Emit(loc, NameNotFound, name_id);
}

}  // namespace Carbon::Check
