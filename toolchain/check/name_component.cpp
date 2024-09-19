// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/name_component.h"

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto PopNameComponent(Context& context) -> NameComponent {
  Parse::NodeId first_param_node_id = Parse::InvalidNodeId();
  Parse::NodeId last_param_node_id = Parse::InvalidNodeId();

  // Explicit params.
  auto [params_loc_id, params_id] =
      context.node_stack().PopWithNodeIdIf<Parse::NodeKind::TuplePattern>();
  if (params_id) {
    first_param_node_id =
        context.node_stack()
            .PopForSoloNodeId<Parse::NodeKind::TuplePatternStart>();
    last_param_node_id = params_loc_id;
  }

  // Implicit params.
  auto [implicit_params_loc_id, implicit_params_id] =
      context.node_stack()
          .PopWithNodeIdIf<Parse::NodeKind::ImplicitParamList>();
  if (implicit_params_id) {
    // Implicit params always come before explicit params.
    first_param_node_id =
        context.node_stack()
            .PopForSoloNodeId<Parse::NodeKind::ImplicitParamListStart>();
    // Only use the end of implicit params if there weren't explicit params.
    if (last_param_node_id.is_valid()) {
      last_param_node_id = params_loc_id;
    }
  }

  auto [name_loc_id, name_id] = context.node_stack().PopNameWithNodeId();
  return {
      .name_loc_id = name_loc_id,
      .name_id = name_id,
      .first_param_node_id = first_param_node_id,
      .last_param_node_id = last_param_node_id,
      .implicit_params_loc_id = implicit_params_loc_id,
      .implicit_params_id =
          implicit_params_id.value_or(SemIR::InstBlockId::Invalid),
      .params_loc_id = params_loc_id,
      .params_id = params_id.value_or(SemIR::InstBlockId::Invalid),
  };
}

// Pop the name of a declaration from the node stack, and diagnose if it has
// parameters.
auto PopNameComponentWithoutParams(Context& context, Lex::TokenKind introducer)
    -> NameComponent {
  NameComponent name = PopNameComponent(context);
  if (name.implicit_params_id.is_valid() || name.params_id.is_valid()) {
    CARBON_DIAGNOSTIC(UnexpectedDeclNameParams, Error,
                      "`{0}` declaration cannot have parameters",
                      Lex::TokenKind);
    // Point to the lexically first parameter list in the diagnostic.
    context.emitter().Emit(name.implicit_params_id.is_valid()
                               ? name.implicit_params_loc_id
                               : name.params_loc_id,
                           UnexpectedDeclNameParams, introducer);

    name.implicit_params_id = SemIR::InstBlockId::Invalid;
    name.params_id = SemIR::InstBlockId::Invalid;
  }
  return name;
}

}  // namespace Carbon::Check
