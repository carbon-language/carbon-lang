// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/decl_introducer_state.h"
#include "toolchain/check/function.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::LambdaIntroducerId node_id)
    -> bool {
  // The function expression is potentially generic.
  StartGenericDecl(context);
  StartFunctionSignature(context);
  context.node_stack().Push(node_id);
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Fn>();
  return true;
}

auto HandleParseNode(Context& context, Parse::LambdaDefinitionStartId node_id)
    -> bool {
  auto return_decl = PopReturnDecl(context);

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
    first_param_node_id =
        context.node_stack()
            .PopForSoloNodeId<Parse::NodeKind::ImplicitParamListStart>();
    if (!last_param_node_id.has_value()) {
      last_param_node_id = implicit_params_node_id;
    }
  } else {
    implicit_param_patterns_id = SemIR::InstBlockId::None;
  }

  auto match_results =
      CalleePatternMatch(context, *implicit_param_patterns_id,
                         *param_patterns_id, return_decl.pattern_id);
  context.full_pattern_stack().PopFullPattern();
  auto pattern_block_id = context.pattern_block_stack().Pop();
  if (!param_patterns_id->has_value() &&
      !implicit_param_patterns_id->has_value() &&
      !return_decl.has_return_decl()) {
    pattern_block_id = SemIR::InstBlockId::None;
  }
  auto decl_block_id = context.inst_block_stack().Pop();

  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::LambdaIntroducer>();
  context.decl_introducer_state_stack().Pop<Lex::TokenKind::Fn>();

  // Function expressions may not take `self` as a parameter. (`self` in the
  // implicit parameter list is already diagnosed by `SelfInImplicitParamList`.)
  if (auto self_param_id = FindSelfPattern(context, SemIR::InstBlockId::None,
                                           *param_patterns_id);
      self_param_id.has_value()) {
    CARBON_DIAGNOSTIC(SelfParameterInLambda, Error,
                      "`self` parameter in function expression");
    context.emitter().Emit(SemIR::LocId(self_param_id), SelfParameterInLambda);
  }

  auto [decl_id, function_id] = MakeFunctionDecl(
      context, node_id, decl_block_id, /*build_generic=*/true,
      /*is_definition=*/true,
      SemIR::Function{
          {
              .name_id = SemIR::NameId::None,
              .parent_scope_id = SemIR::NameScopeId::None,
              .generic_id = SemIR::GenericId::None,
              .first_param_node_id = first_param_node_id,
              .last_param_node_id = last_param_node_id,
              .pattern_block_id = pattern_block_id,
              .implicit_param_patterns_id = *implicit_param_patterns_id,
              .param_patterns_id = *param_patterns_id,
              .is_extern = false,
              .extern_library_id = SemIR::LibraryNameId::None,
              .non_owning_decl_id = SemIR::InstId::None,
              .first_owning_decl_id = SemIR::InstId::None,
          },
          {
              .call_param_patterns_id = match_results.call_param_patterns_id,
              .call_params_id = match_results.call_params_id,
              .call_param_ranges = match_results.param_ranges,
              .return_type_inst_id = return_decl.form.type_component_inst_id,
              .return_form_inst_id = return_decl.form.form_inst_id,
              .return_pattern_id = return_decl.pattern_id,
              .has_deduced_return_type = return_decl.is_deduced,
          }});
  context.inst_block_stack().AddInstId(decl_id);

  CheckFunctionParams(context, context.functions().Get(function_id));

  StartFunctionDefinition(context, decl_id, function_id);
  context.node_stack().Push(node_id, function_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::LambdaId node_id) -> bool {
  if (context.node_stack().PeekIs(Parse::NodeCategory::Expr)) {
    return context.TODO(node_id, "terse lambda body");
  }
  auto function_id =
      context.node_stack().Pop<Parse::NodeKind::LambdaDefinitionStart>();
  CheckFunctionReturnOnFinish(context, node_id, function_id);
  FinishFunctionDefinition(context, function_id);
  context.scope_stack().Pop(/*check_unused=*/true);
  context.node_stack().Push(
      node_id, context.functions().Get(function_id).first_owning_decl_id);
  return true;
}

}  // namespace Carbon::Check
