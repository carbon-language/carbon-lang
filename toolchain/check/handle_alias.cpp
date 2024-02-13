// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/typed_insts.h"
#include "toolchain/sem_ir/value_stores.h"

namespace Carbon::Check {

auto HandleAliasIntroducer(Context& context,
                           Parse::AliasIntroducerId /*parse_node*/) -> bool {
  context.decl_state_stack().Push(DeclState::Alias);
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

auto HandleAliasInitializer(Context& /*context*/,
                            Parse::AliasInitializerId /*parse_node*/) -> bool {
  return true;
}

auto HandleAlias(Context& context, Parse::AliasId /*parse_node*/) -> bool {
  auto [expr_node, expr_id] = context.node_stack().PopExprWithParseNode();

  auto name_context = context.decl_name_stack().FinishName();

  LimitModifiersOnDecl(context, KeywordModifierSet::None,
                       Lex::TokenKind::Namespace);
  context.decl_state_stack().Pop(DeclState::Alias);

  auto name_id = name_context.name_id_for_new_inst();
  auto bind_name_id = context.bind_names().Add(
      {.name_id = name_id,
       .enclosing_scope_id = name_context.enclosing_scope_id_for_new_inst()});

  SemIR::TypeId type_id = context.insts().Get(expr_id).type_id();
  if (type_id != SemIR::TypeId::Error &&
      !context.constant_values().Get(expr_id).is_template()) {
    CARBON_DIAGNOSTIC(AliasNotConstant, Error,
                      "Alias initializer must be a template constant.");
    context.emitter().Emit(expr_node, AliasNotConstant);
    type_id = SemIR::TypeId::Error;
  }

  auto alias_id = context.AddInst(
      {name_context.parse_node,
       SemIR::BindTemplateName{type_id, bind_name_id, expr_id}});

  // Add the name of the binding to the current scope.
  context.decl_name_stack().PopScope();
  context.decl_name_stack().AddNameToLookup(name_context, alias_id);
  return true;
}

}  // namespace Carbon::Check
