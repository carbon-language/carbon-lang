// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/subst.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/named_constraint.h"
#include "toolchain/sem_ir/type_iterator.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::RequireIntroducerId node_id)
    -> bool {
  // Create an instruction block to hold the instructions created for the type
  // and constraint.
  context.inst_block_stack().Push();

  // Optional modifiers follow.
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Require>();

  auto scope_id = context.scope_stack().PeekNameScopeId();
  auto scope_inst_id = context.name_scopes().Get(scope_id).inst_id();
  auto scope_inst = context.insts().Get(scope_inst_id);
  if (!scope_inst.Is<SemIR::InterfaceDecl>() &&
      !scope_inst.Is<SemIR::NamedConstraintDecl>()) {
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

  auto scope_id = context.scope_stack().PeekNameScopeId();
  auto lookup_result =
      LookupNameInExactScope(context, node_id, SemIR::NameId::SelfType,
                             scope_id, context.name_scopes().Get(scope_id),
                             /*is_being_declared=*/false);
  CARBON_CHECK(lookup_result.is_found());

  auto self_inst_id = lookup_result.target_inst_id();
  auto self_type_id = context.insts().Get(self_inst_id).type_id();
  CARBON_CHECK(context.types().Is<SemIR::FacetType>(self_type_id));

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
  context.node_stack().Push(node_id, self_type.inst_id);
  return true;
}

static auto ConstraintHasInterface(Context& context,
                                   SemIR::FacetType facet_type) -> bool {
  const auto& facet_type_info =
      context.facet_types().Get(facet_type.facet_type_id);

  return !facet_type_info.extend_constraints.empty() ||
         !facet_type_info.self_impls_constraints.empty();
}

static auto TypeStructureReferencesSelf(Context& context,
                                        SemIR::TypeInstId inst_id,
                                        SemIR::FacetType facet_type) -> bool {
  if (inst_id == SemIR::ErrorInst::TypeInstId) {
    // Don't generate more diagnostics.
    return true;
  }

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
        case CARBON_KIND(SemIR::TypeIterator::Step::SymbolicBinding bind): {
          if (context.entity_names().Get(bind.entity_name_id).name_id ==
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
    type_iter.Add(context.constant_values().GetConstantTypeInstId(inst_id));
    if (find_self(type_iter)) {
      return true;
    }
  }

  const auto& facet_type_info =
      context.facet_types().Get(facet_type.facet_type_id);
  for (auto extend : facet_type_info.extend_constraints) {
    SemIR::TypeIterator type_iter(&context.sem_ir());
    type_iter.Add(extend);
    if (!find_self(type_iter)) {
      return false;
    }
  }
  for (auto self_impls : facet_type_info.self_impls_constraints) {
    SemIR::TypeIterator type_iter(&context.sem_ir());
    type_iter.Add(self_impls);
    if (!find_self(type_iter)) {
      return false;
    }
  }

  return true;
}

static auto RequirementReferencesSelf(
    Context& context, const SemIR::FacetTypeInfo& facet_type_info) -> bool {
  class FindSelfCallbacks : public SubstInstCallbacks {
   public:
    explicit FindSelfCallbacks(Context* context, bool* found)
        : SubstInstCallbacks(context), found_(found) {}
    auto Subst(SemIR::InstId& inst_id) -> SubstResult override {
      if (*found_ || context().constant_values().Get(inst_id).is_concrete()) {
        return FullySubstituted;
      }
      if (auto bind =
              context().insts().TryGetAs<SemIR::SymbolicBindingType>(inst_id)) {
        const auto& entity_name =
            context().entity_names().Get(bind->entity_name_id);
        if (entity_name.name_id == SemIR::NameId::SelfType) {
          // It would be nice to return a location, but we're working with
          // canonical instructions so there's no location available here.
          *found_ = true;
          return FullySubstituted;
        }
      }
      return SubstOperands;
    }
    auto Rebuild(SemIR::InstId /*orig_inst_id*/, SemIR::Inst /*new_inst*/)
        -> SemIR::InstId override {
      CARBON_FATAL();
    }

    bool* found_;
  };

  bool found = false;
  FindSelfCallbacks callbacks(&context, &found);
  for (const auto& rewrite : facet_type_info.rewrite_constraints) {
    SubstInst(context, rewrite.lhs_id, callbacks);
    if (found) {
      return true;
    }
    SubstInst(context, rewrite.rhs_id, callbacks);
    if (found) {
      return true;
    }
  }

  return false;
}

auto HandleParseNode(Context& context, Parse::RequireDeclId node_id) -> bool {
  auto [constraint_node_id, constraint_inst_id] =
      context.node_stack().PopExprWithNodeId();
  auto [self_node_id, self_inst_id] =
      context.node_stack().PopWithNodeId<Parse::NodeCategory::RequireImpls>();

  auto constraint_constant_value_inst_id =
      context.constant_values().GetConstantInstId(constraint_inst_id);
  auto constraint_facet_type = context.insts().TryGetAs<SemIR::FacetType>(
      constraint_constant_value_inst_id);
  if (constraint_constant_value_inst_id == SemIR::ErrorInst::InstId) {
    constraint_inst_id = self_inst_id = SemIR::ErrorInst::TypeInstId;
  } else if (!constraint_facet_type) {
    CARBON_DIAGNOSTIC(
        RequireImplsMissingFacetType, Error,
        "`require` declaration constrained by a non-facet type; "
        "expected an `interface` or `constraint` name after `impls`");
    context.emitter().Emit(constraint_node_id, RequireImplsMissingFacetType);
    constraint_inst_id = self_inst_id = SemIR::ErrorInst::TypeInstId;
  } else if (!ConstraintHasInterface(context, *constraint_facet_type)) {
    CARBON_DIAGNOSTIC(
        RequireImplsHasEmptyFacetType, Error,
        "`require` declaration constrained by an empty constraint; "
        "expected an `interface` or a non-empty `constraint`");
    context.emitter().Emit(constraint_node_id, RequireImplsHasEmptyFacetType);
    constraint_inst_id = self_inst_id = SemIR::ErrorInst::TypeInstId;
  } else if (!TypeStructureReferencesSelf(context, self_inst_id,
                                          *constraint_facet_type)) {
    CARBON_DIAGNOSTIC(RequireImplsMissingSelf, Error,
                      "no `Self` reference found in `require` declaration; "
                      "`Self` must appear in the self-type or as a generic "
                      "parameter for each `interface` or `constraint`");
    context.emitter().Emit(node_id, RequireImplsMissingSelf);
    constraint_inst_id = self_inst_id = SemIR::ErrorInst::TypeInstId;
  } else if (RequirementReferencesSelf(
                 context, context.facet_types().Get(
                              constraint_facet_type->facet_type_id))) {
    // TODO: Should this be allowed? For now, no, but leads question:
    // https://github.com/carbon-language/carbon-lang/issues/6285
    CARBON_DIAGNOSTIC(RequireImplsSelfInWhereExpr, Error,
                      "`require` declaration with `Self` in the `where` "
                      "expression of the constraint");
    context.emitter().Emit(constraint_node_id, RequireImplsSelfInWhereExpr);
    constraint_inst_id = self_inst_id = SemIR::ErrorInst::TypeInstId;
  }

  [[maybe_unused]] auto decl_block_id = context.inst_block_stack().Pop();

  // Process modifiers.
  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Require>();
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::Extend);

  auto scope_inst_id =
      context.node_stack().Pop<Parse::NodeKind::RequireIntroducer>();
  if (scope_inst_id == SemIR::ErrorInst::InstId) {
    // `require` is in the wrong scope.
    return true;
  }

  // TODO: Add the `require` constraint to the InterfaceDecl or ConstraintDecl
  // from `scope_inst_id`.

  return true;
}

}  // namespace Carbon::Check
