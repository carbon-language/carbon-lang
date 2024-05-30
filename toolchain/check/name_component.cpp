// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/name_component.h"

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto PopNameComponent(Context& context) -> NameComponent {
  auto [params_loc_id, params_id] =
      context.node_stack().PopWithNodeIdIf<Parse::NodeKind::TuplePattern>();
  auto [implicit_params_loc_id, implicit_params_id] =
      context.node_stack()
          .PopWithNodeIdIf<Parse::NodeKind::ImplicitParamList>();
  auto [name_loc_id, name_id] = context.node_stack().PopNameWithNodeId();
  return {
      .name_loc_id = name_loc_id,
      .name_id = name_id,
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
                      "`{0}` declaration cannot have parameters.",
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
