// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/name_lookup.h"

#include <optional>

#include "common/raw_string_ostream.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/subst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/name_scope.h"

namespace Carbon::Check {

auto AddNameToLookup(Context& context, SemIR::NameId name_id,
                     SemIR::InstId target_id, ScopeIndex scope_index) -> void {
  if (auto existing = context.scope_stack().LookupOrAddName(
          name_id, target_id, scope_index, IsCurrentPositionReachable(context));
      existing.has_value()) {
    // TODO: Add coverage to this use case and use the location of the name
    // instead of the target.
    DiagnoseDuplicateName(context, name_id, SemIR::LocId(target_id),
                          SemIR::LocId(existing));
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
    // In this case, the class C is not a redeclaration of its parameter, but
    // we find the parameter in order to diagnose a redeclaration error.
    return SemIR::ScopeLookupResult::MakeWrappedLookupResult(
        context.scope_stack().LookupInLexicalScopesWithin(
            name_id, scope_index, /*use_loc_id=*/SemIR::LocId::None,
            /*is_reachable=*/true),
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

auto LookupUnqualifiedName(Context& context, SemIR::LocId loc_id,
                           SemIR::NameId name_id, bool required)
    -> LookupResult {
  // TODO: Check for shadowed lookup results.

  // Find the results from ancestor lexical scopes. These will be combined with
  // results from non-lexical scopes such as namespaces and classes.
  auto [lexical_result, non_lexical_scopes] =
      context.scope_stack().LookupInLexicalScopes(
          name_id, loc_id, IsCurrentPositionReachable(context));

  // Walk the non-lexical scopes and perform lookups into each of them.
  for (auto [index, lookup_scope_id, specific_id] :
       llvm::reverse(non_lexical_scopes)) {
    if (auto non_lexical_result = LookupQualifiedName(
            context, loc_id, name_id,
            LookupScope{.name_scope_id = lookup_scope_id,
                        .specific_id = specific_id,
                        // A non-lexical lookup does not know what `Self` will
                        // be; it remains symbolic if needed.
                        .self_const_id = SemIR::ConstantId::None},
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
                    SemIR::GetTypeOfInstInSpecific(
                        context.sem_ir(), non_lexical_result.specific_id,
                        target_inst_id))) {
          auto interface_decl =
              context.insts().GetAs<SemIR::InterfaceWithSelfDecl>(
                  scope.inst_id());
          const auto& interface =
              context.interfaces().Get(interface_decl.interface_id);
          SemIR::InstId result_inst_id = GetAssociatedValue(
              context, loc_id, interface.self_param_id,
              SemIR::GetConstantValueInSpecific(context.sem_ir(),
                                                non_lexical_result.specific_id,
                                                target_inst_id),
              assoc_type->GetSpecificInterface());
          non_lexical_result = {
              .specific_id = SemIR::SpecificId::None,
              .scope_result = SemIR::ScopeLookupResult::MakeFound(
                  result_inst_id,
                  non_lexical_result.scope_result.access_kind())};
        }
      }
      return non_lexical_result;
    }
  }

  if (lexical_result == SemIR::InstId::InitTombstone) {
    CARBON_DIAGNOSTIC(UsedBeforeInitialization, Error,
                      "`{0}` used before initialization", SemIR::NameId);
    context.emitter().Emit(loc_id, UsedBeforeInitialization, name_id);
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
    DiagnoseNameNotFound(context, loc_id, name_id);
  }

  // TODO: Should this return MakeNotFound if `required` is false, so that
  // `is_found()` would be false?
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
    return ImportNameFromCpp(context, loc_id, scope_id, name_id);
  }

  return SemIR::ScopeLookupResult::MakeNotFound();
}

// Prints diagnostics on invalid qualified name access.
static auto DiagnoseInvalidQualifiedNameAccess(
    Context& context, SemIR::LocId loc_id, SemIR::LocId member_loc_id,
    SemIR::NameId name_id, SemIR::AccessKind access_kind, bool is_parent_access,
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
      Diagnostics::BoolAsSelect, SemIR::NameId, SemIR::TypeId);
  CARBON_DIAGNOSTIC(ClassMemberDeclaration, Note, "declared here");
  context.emitter()
      .Build(loc_id, ClassInvalidMemberAccess,
             access_kind == SemIR::AccessKind::Private, name_id, parent_type_id)
      .Note(member_loc_id, ClassMemberDeclaration)
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

auto CheckAccess(Context& context, SemIR::LocId loc_id,
                 SemIR::LocId member_loc_id, SemIR::NameId name_id,
                 SemIR::AccessKind access_kind, bool is_parent_access,
                 AccessInfo access_info) -> void {
  if (IsAccessProhibited(access_info, access_kind, is_parent_access)) {
    DiagnoseInvalidQualifiedNameAccess(context, loc_id, member_loc_id, name_id,
                                       access_kind, is_parent_access,
                                       access_info);
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

static auto GetSelfFacetForInterfaceFromLookupSelfType(
    Context& context, const SemIR::GenericId generic_with_self_id,
    SemIR::ConstantId self_type_const_id) -> SemIR::ConstantId {
  if (!self_type_const_id.has_value()) {
    // In a lookup into a non-lexical scope, there is no self-type from the
    // lookup for the interface-with-self specific. So the self-type we use is
    // the abstract symbolic Self from the self specific of the
    // interface-with-self.
    auto self_specific_args_id = context.specifics().GetArgsOrEmpty(
        context.generics().GetSelfSpecific(generic_with_self_id));
    auto self_specific_args = context.inst_blocks().Get(self_specific_args_id);
    return context.constant_values().Get(self_specific_args.back());
  }

  if (context.insts().Is<SemIR::FacetType>(
          context.constant_values().GetInstId(self_type_const_id))) {
    // We are looking directly in a facet type, like `I.F` for an interface `I`,
    // which means there is no self-type from the lookup for the
    // interface-with-self specific. So the self-type we use is the abstract
    // symbolic Self from the self specific of the interface-with-self.
    auto self_specific_args_id = context.specifics().GetArgsOrEmpty(
        context.generics().GetSelfSpecific(generic_with_self_id));
    auto self_specific_args = context.inst_blocks().Get(self_specific_args_id);
    return context.constant_values().Get(self_specific_args.back());
  }

  // Extended name lookup into a type, like `x.F`, can find a facet
  // type extended scope from the type of `x`. The type of `x` maybe a
  // facet converted to a type, so drop the `as type` conversion if
  // so.
  auto canonical_facet_or_type =
      GetCanonicalFacetOrTypeValue(context, self_type_const_id);

  auto type_of_canonical_facet_or_type =
      context.insts()
          .Get(context.constant_values().GetInstId(canonical_facet_or_type))
          .type_id();
  if (type_of_canonical_facet_or_type == SemIR::TypeType::TypeId) {
    // If we still have a type, turn it into a facet for use in the
    // interface-with-self specific.
    return GetConstantFacetValueForType(
        context, context.types().GetAsTypeInstId(
                     context.constant_values().GetInstId(self_type_const_id)));
  }

  // We have a facet for the self-type (or perhaps an ErrorInst), which we can
  // use directly in the interface-with-self specific.
  return canonical_facet_or_type;
}

auto AppendLookupScopesForConstant(Context& context, SemIR::LocId loc_id,
                                   SemIR::ConstantId lookup_const_id,
                                   SemIR::ConstantId self_type_const_id,
                                   llvm::SmallVector<LookupScope>* scopes)
    -> bool {
  auto lookup_inst_id = context.constant_values().GetInstId(lookup_const_id);
  auto lookup = context.insts().Get(lookup_inst_id);

  if (auto ns = lookup.TryAs<SemIR::Namespace>()) {
    scopes->push_back(LookupScope{.name_scope_id = ns->name_scope_id,
                                  .specific_id = SemIR::SpecificId::None,
                                  .self_const_id = SemIR::ConstantId::None});
    return true;
  }
  if (auto class_ty = lookup.TryAs<SemIR::ClassType>()) {
    // TODO: Allow name lookup into classes that are being defined even if they
    // are not complete.
    RequireCompleteType(
        context, context.types().GetTypeIdForTypeConstantId(lookup_const_id),
        loc_id, [&](auto& builder) {
          CARBON_DIAGNOSTIC(QualifiedExprInIncompleteClassScope, Context,
                            "member access into incomplete class {0}",
                            InstIdAsType);
          builder.Context(loc_id, QualifiedExprInIncompleteClassScope,
                          lookup_inst_id);
        });
    auto& class_info = context.classes().Get(class_ty->class_id);
    scopes->push_back(LookupScope{.name_scope_id = class_info.scope_id,
                                  .specific_id = class_ty->specific_id,
                                  .self_const_id = self_type_const_id});
    return true;
  }
  // Extended scopes may point to a FacetType.
  if (auto facet_type = lookup.TryAs<SemIR::FacetType>()) {
    // TODO: Allow name lookup into facet types that are being defined even if
    // they are not complete.
    if (RequireCompleteType(
            context,
            context.types().GetTypeIdForTypeConstantId(lookup_const_id), loc_id,
            [&](auto& builder) {
              CARBON_DIAGNOSTIC(
                  QualifiedExprInIncompleteFacetTypeScope, Context,
                  "member access into incomplete facet type {0}", InstIdAsType);
              builder.Context(loc_id, QualifiedExprInIncompleteFacetTypeScope,
                              lookup_inst_id);
            })) {
      auto facet_type_info =
          context.facet_types().Get(facet_type->facet_type_id);
      // Name lookup into "extend" constraints but not "self impls" constraints.
      for (const auto& extend : facet_type_info.extend_constraints) {
        auto& interface = context.interfaces().Get(extend.interface_id);

        // We need to build the inner interface-with-self specific. To do that
        // we need to determine the self facet value to use.
        auto self_facet = GetSelfFacetForInterfaceFromLookupSelfType(
            context, interface.generic_with_self_id, self_type_const_id);
        auto interface_with_self_specific_id = MakeSpecificWithInnerSelf(
            context, loc_id, interface.generic_id,
            interface.generic_with_self_id, extend.specific_id, self_facet);

        scopes->push_back({.name_scope_id = interface.scope_with_self_id,
                           .specific_id = interface_with_self_specific_id,
                           .self_const_id = self_type_const_id});
      }
      for (const auto& extend : facet_type_info.extend_named_constraints) {
        auto& constraint =
            context.named_constraints().Get(extend.named_constraint_id);

        // We need to build the inner constraint-with-self specific. To do that
        // we need to determine the self facet value to use.
        auto self_facet = GetSelfFacetForInterfaceFromLookupSelfType(
            context, constraint.generic_with_self_id, self_type_const_id);
        auto constraint_with_self_specific_id = MakeSpecificWithInnerSelf(
            context, loc_id, constraint.generic_id,
            constraint.generic_with_self_id, extend.specific_id, self_facet);

        scopes->push_back({.name_scope_id = constraint.scope_with_self_id,
                           .specific_id = constraint_with_self_specific_id,
                           .self_const_id = self_type_const_id});
      }
    } else {
      // Lookup into this scope should fail without producing an error since
      // `RequireCompleteFacetType` has already issued a diagnostic.
      scopes->push_back(LookupScope{.name_scope_id = SemIR::NameScopeId::None,
                                    .specific_id = SemIR::SpecificId::None,
                                    .self_const_id = SemIR::ConstantId::None});
    }
    return true;
  }
  if (lookup_const_id == SemIR::ErrorInst::ConstantId) {
    // Lookup into this scope should fail without producing an error.
    scopes->push_back(LookupScope{.name_scope_id = SemIR::NameScopeId::None,
                                  .specific_id = SemIR::SpecificId::None,
                                  .self_const_id = SemIR::ConstantId::None});
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
    Context& context, SemIR::LocId loc_id, SemIR::NameId name_id,
    llvm::ArrayRef<LookupScope> lookup_scopes) -> void {
  if (lookup_scopes.size() == 1 &&
      lookup_scopes.front().name_scope_id.has_value()) {
    if (auto specific_id = lookup_scopes.front().specific_id;
        specific_id.has_value()) {
      CARBON_DIAGNOSTIC(MemberNameNotFoundInSpecificScope, Error,
                        "member name `{0}` not found in {1}", SemIR::NameId,
                        SemIR::SpecificId);
      context.emitter().Emit(loc_id, MemberNameNotFoundInSpecificScope, name_id,
                             specific_id);
    } else {
      auto scope_inst_id = context.name_scopes()
                               .Get(lookup_scopes.front().name_scope_id)
                               .inst_id();
      CARBON_DIAGNOSTIC(MemberNameNotFoundInInstScope, Error,
                        "member name `{0}` not found in {1}", SemIR::NameId,
                        InstIdAsType);
      context.emitter().Emit(loc_id, MemberNameNotFoundInInstScope, name_id,
                             scope_inst_id);
    }
    return;
  }

  CARBON_DIAGNOSTIC(MemberNameNotFound, Error, "member name `{0}` not found",
                    SemIR::NameId);
  context.emitter().Emit(loc_id, MemberNameNotFound, name_id);
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
  auto parent_const_id = SemIR::ConstantId::None;
  bool has_error = false;
  bool is_parent_access = false;

  // Walk this scope and, if nothing is found here, the scopes it extends.
  while (!scopes.empty()) {
    auto [scope_id, specific_id, self_const_id] = scopes.pop_back_val();
    if (!scope_id.has_value()) {
      has_error = true;
      continue;
    }
    auto& name_scope = context.name_scopes().Get(scope_id);
    has_error |= name_scope.has_error();

    const SemIR::ScopeLookupResult scope_result =
        LookupNameInExactScope(context, loc_id, name_id, scope_id, name_scope);
    SemIR::AccessKind access_kind = scope_result.access_kind();

    if (is_parent_access && scope_result.is_found() &&
        !access_info.has_value()) {
      access_info =
          AccessInfo{.constant_id = parent_const_id,
                     .highest_allowed_access = SemIR::AccessKind::Protected};
    }

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
        if (!AppendLookupScopesForConstant(context, loc_id, const_id,
                                           self_const_id, &scopes)) {
          // TODO: Handle case where we have a symbolic type and instead should
          // look in its type.
        }
      }
      is_parent_access |= !extended.empty();
      parent_const_id = context.constant_values().Get(name_scope.inst_id());
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

  if ((!prohibited_accesses.empty() || required) &&
      !result.scope_result.is_found()) {
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
        DiagnoseInvalidQualifiedNameAccess(
            context, loc_id, SemIR::LocId(scope_result_id), name_id,
            access_kind, is_parent_access, *access_info);
      }
    }

    CARBON_CHECK(!result.scope_result.is_poisoned());
    return {.specific_id = SemIR::SpecificId::None,
            .scope_result = SemIR::ScopeLookupResult::MakeError()};
  }

  return result;
}

// Returns a `Core.<qualifiers>` name for diagnostics.
static auto GetCoreQualifiedName(llvm::ArrayRef<CoreIdentifier> qualifiers)
    -> std::string {
  RawStringOstream str;
  str << "Core";
  for (auto qualifier : qualifiers) {
    str << "." << qualifier;
  }
  return str.TakeStr();
}

// Returns the scope of the Core package, or `None` if it's not found.
//
// TODO: Consider tracking the Core package in SemIR so we don't need to use
// name lookup to find it.
static auto GetCorePackage(Context& context, SemIR::LocId loc_id,
                           llvm::ArrayRef<CoreIdentifier> qualifiers)
    -> SemIR::NameScopeId {
  if (context.name_scopes().IsCorePackage(SemIR::NameScopeId::Package)) {
    return SemIR::NameScopeId::Package;
  }

  // Look up `package.Core`.
  auto core_scope_result = LookupNameInExactScope(
      context, loc_id, SemIR::NameId::Core, SemIR::NameScopeId::Package,
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
      "`{0}` implicitly referenced here, but package `Core` not found",
      std::string);
  context.emitter().Emit(loc_id, CoreNotFound,
                         GetCoreQualifiedName(qualifiers));
  return SemIR::NameScopeId::None;
}

auto LookupNameInCore(Context& context, SemIR::LocId loc_id,
                      llvm::ArrayRef<CoreIdentifier> qualifiers)
    -> SemIR::InstId {
  CARBON_CHECK(!qualifiers.empty());

  auto core_package_id = GetCorePackage(context, loc_id, qualifiers);
  if (!core_package_id.has_value()) {
    return SemIR::ErrorInst::InstId;
  }

  auto inst_id = SemIR::InstId::None;
  for (auto qualifier : qualifiers) {
    auto name_id = context.core_identifiers().AddNameId(qualifier);

    auto scope_id = SemIR::NameScopeId::None;
    if (inst_id.has_value()) {
      auto namespace_inst = context.insts().TryGetAs<SemIR::Namespace>(inst_id);
      if (namespace_inst) {
        scope_id = namespace_inst->name_scope_id;
      }
    } else {
      scope_id = core_package_id;
    }

    auto scope_result =
        scope_id.has_value()
            ? LookupNameInExactScope(context, loc_id, name_id, scope_id,
                                     context.name_scopes().Get(scope_id))
            : SemIR::ScopeLookupResult::MakeNotFound();
    if (!scope_result.is_found()) {
      CARBON_DIAGNOSTIC(CoreNameNotFound, Error,
                        "name `{0}` implicitly referenced here, but not found",
                        std::string);
      context.emitter().Emit(loc_id, CoreNameNotFound,
                             GetCoreQualifiedName(qualifiers));
      return SemIR::ErrorInst::InstId;
    }

    // Look through import_refs and aliases.
    inst_id = context.constant_values().GetConstantInstId(
        scope_result.target_inst_id());
  }

  return inst_id;
}

auto DiagnoseDuplicateName(Context& context, SemIR::NameId name_id,
                           SemIR::LocId dup_def, SemIR::LocId prev_def)
    -> void {
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

auto DiagnoseNameNotFound(Context& context, SemIR::LocId loc_id,
                          SemIR::NameId name_id) -> void {
  CARBON_DIAGNOSTIC(NameNotFound, Error, "name `{0}` not found", SemIR::NameId);
  context.emitter().Emit(loc_id, NameNotFound, name_id);
}

}  // namespace Carbon::Check
