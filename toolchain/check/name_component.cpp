// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/name_component.h"

#include "toolchain/check/context.h"
#include "toolchain/check/pattern_match.h"

namespace Carbon::Check {

auto PopNameComponent(Context& context, SemIR::InstId return_slot_pattern_id)
    -> NameComponent {
  NameComponent name_component = {
      .name_loc_id = Parse::NodeId::Invalid,
      .name_id = SemIR::NameId::Invalid,
      .first_param_node_id = Parse::NodeId::Invalid,
      .last_param_node_id = Parse::NodeId::Invalid,
      .implicit_params_loc_id = Parse::NodeId::Invalid,
      .implicit_param_patterns_id = SemIR::InstBlockId::Invalid,
      .params_loc_id = Parse::NodeId::Invalid,
      .param_patterns_id = SemIR::InstBlockId::Invalid,
      .call_params_id = SemIR::InstBlockId::Invalid,
      .return_slot_pattern_id = return_slot_pattern_id,
      .pattern_block_id = SemIR::InstBlockId::Invalid,
  };

  // Explicit params.
  if (auto [params_loc_id, param_patterns_id] =
          context.node_stack().PopWithNodeIdIf<Parse::NodeKind::TuplePattern>();
      param_patterns_id) {
    name_component.first_param_node_id =
        context.node_stack()
            .PopForSoloNodeId<Parse::NodeKind::TuplePatternStart>();
    name_component.last_param_node_id = params_loc_id;
    name_component.params_loc_id = params_loc_id;
    name_component.param_patterns_id = *param_patterns_id;
  }

  // Implicit params.
  if (auto [implicit_params_loc_id, implicit_param_patterns_id] =
          context.node_stack()
              .PopWithNodeIdIf<Parse::NodeKind::ImplicitParamList>();
      implicit_param_patterns_id) {
    // Implicit params always come before explicit params.
    name_component.first_param_node_id =
        context.node_stack()
            .PopForSoloNodeId<Parse::NodeKind::ImplicitParamListStart>();
    // Only use the end of implicit params if there weren't explicit params.
    if (!name_component.last_param_node_id.is_valid()) {
      name_component.last_param_node_id = implicit_params_loc_id;
    }
    name_component.implicit_params_loc_id = implicit_params_loc_id;
    name_component.implicit_param_patterns_id = *implicit_param_patterns_id;
  }

  if (name_component.param_patterns_id.is_valid() ||
      name_component.implicit_param_patterns_id.is_valid()) {
    std::tie(name_component.name_loc_id, name_component.name_id) =
        context.node_stack()
            .PopWithNodeId<Parse::NodeKind::IdentifierNameBeforeParams>();
    name_component.call_params_id = CalleePatternMatch(
        context, name_component.implicit_param_patterns_id,
        name_component.param_patterns_id, return_slot_pattern_id);
    name_component.pattern_block_id = context.pattern_block_stack().Pop();
  } else {
    std::tie(name_component.name_loc_id, name_component.name_id) =
        context.node_stack()
            .PopWithNodeId<Parse::NodeKind::IdentifierNameNotBeforeParams>();
  }

  return name_component;
}

// Pop the name of a declaration from the node stack, and diagnose if it has
// parameters.
auto PopNameComponentWithoutParams(Context& context, Lex::TokenKind introducer)
    -> NameComponent {
  NameComponent name = PopNameComponent(context);
  if (name.call_params_id.is_valid()) {
    CARBON_DIAGNOSTIC(UnexpectedDeclNameParams, Error,
                      "`{0}` declaration cannot have parameters",
                      Lex::TokenKind);
    // Point to the lexically first parameter list in the diagnostic.
    context.emitter().Emit(name.implicit_param_patterns_id.is_valid()
                               ? name.implicit_params_loc_id
                               : name.params_loc_id,
                           UnexpectedDeclNameParams, introducer);

    name.call_params_id = SemIR::InstBlockId::Invalid;
  }
  return name;
}

}  // namespace Carbon::Check
