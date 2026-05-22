// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/period_self.h"
#include "toolchain/check/subst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/named_constraint.h"
#include "toolchain/sem_ir/specific_named_constraint.h"
#include "toolchain/sem_ir/type_iterator.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::RequireIntroducerId node_id)
    -> bool {
  // Require decls are always generic, since everything in an `interface` or
  // `constraint` is generic over `Self`.
  StartGenericDecl(context);

  // Create an instruction block to hold the instructions created for the type
  // and constraint.
  context.inst_block_stack().Push();

  // Optional modifiers follow.
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Require>();

  auto scope_id = context.scope_stack().PeekNameScopeId();
  auto scope_inst_id = context.name_scopes().Get(scope_id).inst_id();
  auto scope_inst = context.insts().Get(scope_inst_id);
  if (!scope_inst.Is<SemIR::InterfaceWithSelfDecl>() &&
      !scope_inst.Is<SemIR::NamedConstraintWithSelfDecl>()) {
    CARBON_DIAGNOSTIC(
        RequireInWrongScope, Error,
        "`require` can only be used in an `interface` or `constraint`");
    context.emitter().Emit(node_id, RequireInWrongScope);
    scope_inst_id = SemIR::ErrorInst::InstId;
  }

  context.node_stack().Push(node_id, scope_inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::RequireDefaultSelfImplsId node_id)
    -> bool {
  auto scope_inst_id =
      context.node_stack().Peek<Parse::NodeKind::RequireIntroducer>();
  if (scope_inst_id == SemIR::ErrorInst::InstId) {
    context.node_stack().Push(node_id, SemIR::ErrorInst::TypeInstId);
    return true;
  }

  auto lookup_result =
      LookupUnqualifiedName(context, node_id, SemIR::NameId::SelfType,
                            /*required=*/true);
  auto self_inst_id = lookup_result.scope_result.target_inst_id();
  auto self_type_id = context.insts().Get(self_inst_id).type_id();
  if (self_type_id == SemIR::ErrorInst::TypeId) {
    context.node_stack().Push(node_id, SemIR::ErrorInst::TypeInstId);
    return true;
  }

  CARBON_CHECK(context.types().Is<SemIR::FacetType>(self_type_id));
  // TODO: We could simplify with a call to ExprAsType, like below?
  auto self_facet_as_type = AddTypeInst<SemIR::FacetAccessType>(
      context, node_id,
      {.type_id = SemIR::TypeType::TypeId,
       .facet_value_inst_id = self_inst_id});
  context.node_stack().Push(node_id, self_facet_as_type);
  return true;
}

auto HandleParseNode(Context& context, Parse::RequireTypeImplsId node_id)
    -> bool {
  auto [self_node_id, self_inst_id] = context.node_stack().PopExprWithNodeId();
  auto self_type = ExprAsType(context, self_node_id, self_inst_id);

  const auto& introducer = context.decl_introducer_state_stack().innermost();
  if (introducer.modifier_set.HasAnyOf(KeywordModifierSet::Extend)) {
    if (self_type.type_id != SemIR::ErrorInst::TypeId) {
      CARBON_DIAGNOSTIC(RequireImplsExtendWithExplicitSelf, Error,
                        "`extend require impls` with explicit type");
      // TODO: If the explicit self-type matches a lookup of NameId::SelfType,
      // add a note to the diagnostic: "remove the explicit `Self` type here",
      // and continue without an ErrorInst. See ExtendImplSelfAsDefault.
      context.emitter().Emit(self_node_id, RequireImplsExtendWithExplicitSelf);
    }
    self_type.inst_id = SemIR::ErrorInst::TypeInstId;
  }

  context.node_stack().Push(node_id, self_type.inst_id);
  return true;
}

static auto TypeStructureReferencesSelf(
    Context& context, SemIR::LocId loc_id, SemIR::ConstantId const_id,
    const SemIR::IdentifiedFacetType& identified_facet_type) -> bool {
  auto find_self = [&](SemIR::TypeIterator& type_iter) -> bool {
    while (true) {
      auto step = type_iter.Next();
      if (step.Is<SemIR::TypeIterator::Step::Done>()) {
        break;
      }
      CARBON_KIND_SWITCH(step.any) {
        case CARBON_KIND(SemIR::TypeIterator::Step::Error _): {
          // Don't generate more diagnostics.
          return true;
        }
        case CARBON_KIND(SemIR::TypeIterator::Step::SymbolicType symbolic): {
          if (context.entity_names().Get(symbolic.entity_name_id).name_id ==
              SemIR::NameId::SelfType) {
            return true;
          }
          break;
        }
        default:
          break;
      }
    }
    return false;
  };

  {
    SemIR::TypeIterator type_iter(&context.sem_ir());
    type_iter.Add(context.constant_values().GetInstId(const_id));
    if (find_self(type_iter)) {
      return true;
    }
  }

  if (identified_facet_type.required_impls().empty()) {
    CARBON_DIAGNOSTIC(
        RequireImplsMissingSelfEmptyFacetType, Error,
        "no `Self` reference found in `require` declaration; `Self` must "
        "appear in the self-type or as a generic argument for each required "
        "interface, but no interfaces were found");
    context.emitter().Emit(loc_id, RequireImplsMissingSelfEmptyFacetType);
    return false;
  }

  bool interfaces_all_reference_self = true;
  for (auto [_, specific_interface] : identified_facet_type.required_impls()) {
    SemIR::TypeIterator type_iter(&context.sem_ir());
    type_iter.Add(specific_interface);
    if (!find_self(type_iter)) {
      // TODO: The IdentifiedFacetType loses the location (since it's
      // canonical), but it would be nice to somehow point this diagnostic at
      // the particular interface in the facet type that is missing `Self`.
      CARBON_DIAGNOSTIC(
          RequireImplsMissingSelf, Error,
          "no `Self` reference found in `require` declaration; `Self` must "
          "appear in the self-type or as a generic argument for each required "
          "interface, but found interface `{0}` without a `Self` argument",
          SemIR::SpecificInterface);
      context.emitter().Emit(loc_id, RequireImplsMissingSelf,
                             specific_interface);
      interfaces_all_reference_self = false;
    }
  }
  return interfaces_all_reference_self;
}

struct ValidateRequireResult {
  const SemIR::IdentifiedFacetType* identified_facet_type;
};

// Returns nullopt if a diagnostic has been emitted and the `require` decl is
// not valid.
static auto ValidateRequire(Context& context, SemIR::LocId full_require_loc_id,
                            SemIR::LocId constraint_loc_id,
                            SemIR::InstId self_inst_id,
                            SemIR::InstId constraint_inst_id,
                            SemIR::InstId scope_inst_id)
    -> std::optional<ValidateRequireResult> {
  auto self_type_id = context.types().GetTypeIdForTypeInstId(self_inst_id);
  auto constraint_type_id =
      context.types().TryGetTypeIdForTypeInstId(constraint_inst_id);

  if (self_type_id == SemIR::ErrorInst::TypeId ||
      constraint_type_id == SemIR::ErrorInst::TypeId ||
      scope_inst_id == SemIR::ErrorInst::InstId) {
    // An error was already diagnosed, don't diagnose another. We can't build a
    // useful `require` with an error, it couldn't do anything.
    return std::nullopt;
  }

  auto constraint_facet_type =
      context.types().TryGetAsIfValid<SemIR::FacetType>(constraint_type_id);
  if (!constraint_facet_type) {
    CARBON_DIAGNOSTIC(
        RequireImplsMissingFacetType, Error,
        "`require` declaration constrained by a non-facet type; "
        "expected an `interface` or `constraint` name after `impls`");
    context.emitter().Emit(constraint_loc_id, RequireImplsMissingFacetType);
    // Can't continue without a constraint to use.
    return std::nullopt;
  }

  if (auto named_constraint =
          context.insts().TryGetAs<SemIR::NamedConstraintWithSelfDecl>(
              scope_inst_id)) {
    const auto& constraint_facet_type_info =
        context.facet_types().Get(constraint_facet_type->facet_type_id);
    // TODO: Handle other impls named constraints for the
    // RequireImplsReferenceCycle diagnostic.
    if (constraint_facet_type_info.other_requirements) {
      context.TODO(constraint_loc_id,
                   "facet type has constraints that we don't handle yet");
      return std::nullopt;
    }
    auto named_constraints_from_type_impls = llvm::map_range(
        constraint_facet_type_info.type_impls_named_constraints,
        [](auto impls) { return impls.specific_named_constraint; });
    auto named_constraints = llvm::concat<const SemIR::SpecificNamedConstraint>(
        constraint_facet_type_info.extend_named_constraints,
        constraint_facet_type_info.self_impls_named_constraints,
        named_constraints_from_type_impls);
    for (auto c : named_constraints) {
      if (c.named_constraint_id == named_constraint->named_constraint_id) {
        const auto& named_constraint =
            context.named_constraints().Get(c.named_constraint_id);
        CARBON_DIAGNOSTIC(RequireImplsReferenceCycle, Error,
                          "facet type in `require` declaration refers to the "
                          "named constraint `{0}` from within its definition",
                          SemIR::NameId);
        context.emitter().Emit(constraint_loc_id, RequireImplsReferenceCycle,
                               named_constraint.name_id);
        return std::nullopt;
      }
    }
  }

  auto identified_facet_type_id = RequireIdentifiedFacetType(
      context, constraint_loc_id, self_type_id.AsConstantId(),
      *constraint_facet_type, [&](auto& builder) {
        CARBON_DIAGNOSTIC(
            RequireImplsUnidentifiedFacetType, Context,
            "facet type {0} cannot be identified in `require` declaration",
            SemIR::TypeId);
        builder.Context(constraint_loc_id, RequireImplsUnidentifiedFacetType,
                        constraint_type_id);
      });
  if (!identified_facet_type_id.has_value()) {
    // The constraint can't be used. A diagnostic was emitted by
    // RequireIdentifiedFacetType().
    return std::nullopt;
  }
  const auto& identified =
      context.identified_facet_types().Get(identified_facet_type_id);

  if (!TypeStructureReferencesSelf(context, full_require_loc_id,
                                   self_type_id.AsConstantId(), identified)) {
    return std::nullopt;
  }

  return ValidateRequireResult{.identified_facet_type = &identified};
}

auto HandleParseNode(Context& context, Parse::RequireDeclId node_id) -> bool {
  auto [constraint_node_id, constraint_inst_id] =
      context.node_stack().PopExprWithNodeId();
  auto [self_node_id, self_inst_id] =
      context.node_stack().PopWithNodeId<Parse::NodeCategory::RequireImpls>();

  // Process modifiers.
  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Require>();
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::Extend);
  bool extend = introducer.modifier_set.HasAnyOf(KeywordModifierSet::Extend);

  auto scope_inst_id =
      context.node_stack().Pop<Parse::NodeKind::RequireIntroducer>();

  auto validated =
      ValidateRequire(context, node_id, constraint_node_id, self_inst_id,
                      constraint_inst_id, scope_inst_id);
  if (!validated) {
    // In an `extend` decl, errors get propagated into the parent scope just as
    // names do.
    if (extend) {
      auto scope_id = context.scope_stack().PeekNameScopeId();
      context.name_scopes().Get(scope_id).set_has_error();
    }
    context.inst_block_stack().Pop();
    DiscardGenericDecl(context);
    return true;
  }

  auto [identified_facet_type] = *validated;
  if (identified_facet_type->required_impls().empty()) {
    // A `require T impls type` adds no actual constraints, so nothing to do.
    // This is not an error though.
    context.inst_block_stack().Pop();
    DiscardGenericDecl(context);
    return true;
  }

  // The identified facet type also replaced `.Self` references, but we want to
  // store the full facet type not just the identified one. So we have to
  // replace `.Self` references explicitly here in the canonical constraint. We
  // do this after `ValidateRequire()` which has ensured the constraint is in
  // fact a FacetType.
  auto constraint_type_inst_id = SubstPeriodSelfInFacetType(
      context, constraint_node_id, self_inst_id,
      context.types().GetAsTypeInstId(constraint_inst_id));
  // The replacement of `.Self` can create a new FacetType instruction which we
  // want to be part of the require decl's inst block, so we defer the Pop until
  // after the subst.
  auto decl_block_id = context.inst_block_stack().Pop();

  auto require_impls_decl =
      SemIR::RequireImplsDecl{// To be filled in after.
                              .require_impls_id = SemIR::RequireImplsId::None,
                              .decl_block_id = decl_block_id};
  auto decl_id = AddPlaceholderInst(context, node_id, require_impls_decl);
  // TODO: We don't need to store the `self_inst_id` anymore, since we've
  // encoded it into the constraints of the facet type which was converted to
  // the form `<Self> where .Self impls <Constraint>`.
  auto require_impls_id = context.require_impls().Add(
      {.self_id = self_inst_id,
       .facet_type_inst_id = constraint_type_inst_id,
       .extend_self = extend,
       .decl_id = decl_id,
       .parent_scope_id = context.scope_stack().PeekNameScopeId(),
       .generic_id = BuildGenericDecl(context, decl_id)});

  require_impls_decl.require_impls_id = require_impls_id;
  ReplaceInstBeforeConstantUse(context, decl_id, require_impls_decl);

  // We look for a complete type after BuildGenericDecl, so that the resulting
  // RequireCompleteType instruction is part of the enclosing interface or named
  // constraint generic definition. Then requiring enclosing entity to be
  // complete will resolve that definition (via ResolveSpecificDefinition()) and
  // also construct a specific for the `constraint_inst_id`, finding any
  // monomorphization errors that result.
  if (extend) {
    if (!RequireCompleteType(
            context,
            context.types().GetTypeIdForTypeInstId(constraint_type_inst_id),
            constraint_node_id, [&](auto& builder) {
              CARBON_DIAGNOSTIC(RequireImplsIncompleteFacetType, Context,
                                "`extend require` of incomplete facet type {0}",
                                InstIdAsType);
              builder.Context(constraint_node_id,
                              RequireImplsIncompleteFacetType,
                              constraint_type_inst_id);
            })) {
      return true;
    }

    // The extended scope instruction must be part of the enclosing scope (and
    // generic). A specific for the enclosing scope will be applied to it when
    // using the instruction later. To do so, we wrap the constraint facet type
    // it in a SpecificConstant, which preserves the require declaration's
    // specific along with the facet type.
    //
    // TODO: Remove the separate generic for each require decl, then we don't
    // need a SpecificConstant anymore, as the constraint_inst_id will already
    // be in the generic of the interface-with-self.
    auto self_specific_id = context.generics().GetSelfSpecific(
        context.require_impls().Get(require_impls_id).generic_id);
    auto constraint_id_in_self_specific = AddTypeInst<SemIR::SpecificConstant>(
        context, node_id,
        {.type_id = SemIR::TypeType::TypeId,
         .inst_id = constraint_inst_id,
         .specific_id = self_specific_id});
    auto enclosing_scope_id = context.scope_stack().PeekNameScopeId();
    auto& enclosing_scope = context.name_scopes().Get(enclosing_scope_id);
    enclosing_scope.AddExtendedScope(constraint_id_in_self_specific);
  }

  context.require_impls_stack().AppendToTop(require_impls_id);
  return true;
}

}  // namespace Carbon::Check
