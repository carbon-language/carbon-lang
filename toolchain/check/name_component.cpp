// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/name_component.h"

#include "toolchain/check/context.h"
#include "toolchain/check/pattern_match.h"

namespace Carbon::Check {

auto PopNameComponent(Context& context, SemIR::InstId return_slot_pattern_id)
    -> NameComponent {
  Parse::NodeId first_param_node_id = Parse::NoneNodeId();
  Parse::NodeId last_param_node_id = Parse::NoneNodeId();

  // Explicit params.
  auto [params_node_id, param_patterns_id] =
      context.node_stack()
          .PopWithNodeIdIf<Parse::NodeKind::ExplicitParamList>();
  if (param_patterns_id) {
    first_param_node_id =
        context.node_stack()
            .PopForSoloNodeId<Parse::NodeKind::ExplicitParamListStart>();
    last_param_node_id = params_node_id;
  } else {
    param_patterns_id = SemIR::InstBlockId::None;
  }

  // Implicit params.
  auto [implicit_params_node_id, implicit_param_patterns_id] =
      context.node_stack()
          .PopWithNodeIdIf<Parse::NodeKind::ImplicitParamList>();
  if (implicit_param_patterns_id) {
    // Implicit params always come before explicit params.
    first_param_node_id =
        context.node_stack()
            .PopForSoloNodeId<Parse::NodeKind::ImplicitParamListStart>();
    // Only use the end of implicit params if there weren't explicit params.
    if (!last_param_node_id.has_value()) {
      last_param_node_id = implicit_params_node_id;
    }
  } else {
    implicit_param_patterns_id = SemIR::InstBlockId::None;
  }

  auto call_params_id = SemIR::InstBlockId::None;
  auto pattern_block_id = SemIR::InstBlockId::None;
  if (param_patterns_id->has_value() ||
      implicit_param_patterns_id->has_value() ||
      return_slot_pattern_id.has_value()) {
    call_params_id =
        CalleePatternMatch(context, *implicit_param_patterns_id,
                           *param_patterns_id, return_slot_pattern_id);
    pattern_block_id = context.pattern_block_stack().Pop();
    context.full_pattern_stack().PopFullPattern();
  }

  auto [name_loc_id, name_id] =
      context.node_stack().PopWithNodeId<Parse::NodeCategory::NonExprName>();

  return {
      .name_loc_id = name_loc_id,
      .name_id = name_id,
      .first_param_node_id = first_param_node_id,
      .last_param_node_id = last_param_node_id,
      .implicit_params_loc_id = implicit_params_node_id,
      .implicit_param_patterns_id = *implicit_param_patterns_id,
      .params_loc_id = params_node_id,
      .param_patterns_id = *param_patterns_id,
      .call_params_id = call_params_id,
      .return_slot_pattern_id = return_slot_pattern_id,
      .pattern_block_id = pattern_block_id,
  };
}

// Pop the name of a declaration from the node stack, and diagnose if it has
// parameters.
auto PopNameComponentWithoutParams(Context& context, Lex::TokenKind introducer)
    -> NameComponent {
  NameComponent name = PopNameComponent(context);
  if (name.call_params_id.has_value()) {
    CARBON_DIAGNOSTIC(UnexpectedDeclNameParams, Error,
                      "`{0}` declaration cannot have parameters",
                      Lex::TokenKind);
    // Point to the lexically first parameter list in the diagnostic.
    context.emitter().Emit(name.implicit_param_patterns_id.has_value()
                               ? name.implicit_params_loc_id
                               : name.params_loc_id,
                           UnexpectedDeclNameParams, introducer);

    name.call_params_id = SemIR::InstBlockId::None;
  }
  return name;
}

}  // namespace Carbon::Check
