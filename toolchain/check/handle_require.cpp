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
#include "toolchain/check/type_completion.h"
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

static auto TypeStructureReferencesSelf(
    Context& context, SemIR::TypeInstId inst_id,
    const SemIR::IdentifiedFacetType& identified_facet_type) -> bool {
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

  if (identified_facet_type.required_interfaces().empty()) {
    return false;
  }

  for (auto specific_interface : identified_facet_type.required_interfaces()) {
    SemIR::TypeIterator type_iter(&context.sem_ir());
    type_iter.Add(specific_interface);
    if (!find_self(type_iter)) {
      return false;
    }
  }

  return true;
}

auto HandleParseNode(Context& context, Parse::RequireDeclId node_id) -> bool {
  auto [constraint_node_id, constraint_inst_id] =
      context.node_stack().PopExprWithNodeId();
  auto [self_node_id, self_inst_id] =
      context.node_stack().PopWithNodeId<Parse::NodeCategory::RequireImpls>();

  [[maybe_unused]] auto decl_block_id = context.inst_block_stack().Pop();

  // Process modifiers.
  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Require>();
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::Extend);

  auto scope_inst_id =
      context.node_stack().Pop<Parse::NodeKind::RequireIntroducer>();

  auto constraint_constant_value_inst_id =
      context.constant_values().GetConstantInstId(constraint_inst_id);
  auto constraint_facet_type = context.insts().TryGetAs<SemIR::FacetType>(
      constraint_constant_value_inst_id);
  if (!constraint_facet_type) {
    if (constraint_constant_value_inst_id != SemIR::ErrorInst::InstId) {
      CARBON_DIAGNOSTIC(
          RequireImplsMissingFacetType, Error,
          "`require` declaration constrained by a non-facet type; "
          "expected an `interface` or `constraint` name after `impls`");
      context.emitter().Emit(constraint_node_id, RequireImplsMissingFacetType);
    }
    // Can't continue without a constraint to use.
    return true;
  }

  auto identified_facet_type_id =
      RequireIdentifiedFacetType(context, *constraint_facet_type, [&] {
        CARBON_DIAGNOSTIC(
            RequireImplsUnidentifiedFacetType, Error,
            "facet type {0} cannot be identified in `require` declaration",
            InstIdAsType);
        return context.emitter().Build(constraint_node_id,
                                       RequireImplsUnidentifiedFacetType,
                                       constraint_inst_id);
      });
  if (!identified_facet_type_id.has_value()) {
    // The constraint can't be used.
    return true;
  }
  const auto& identified =
      context.identified_facet_types().Get(identified_facet_type_id);

  if (!TypeStructureReferencesSelf(context, self_inst_id, identified)) {
    CARBON_DIAGNOSTIC(RequireImplsMissingSelf, Error,
                      "no `Self` reference found in `require` declaration; "
                      "`Self` must appear in the self-type or as a generic "
                      "parameter for each `interface` or `constraint`");
    context.emitter().Emit(node_id, RequireImplsMissingSelf);
    return true;
  }

  if (scope_inst_id == SemIR::ErrorInst::InstId) {
    // `require` is in the wrong scope.
    return true;
  }

  if (identified.required_interfaces().empty()) {
    // A `require T impls type` adds no actual constraints.
    return true;
  }

  // TODO: Add the `require` constraint to the InterfaceDecl or ConstraintDecl
  // from `scope_inst_id`.

  return true;
}

}  // namespace Carbon::Check
