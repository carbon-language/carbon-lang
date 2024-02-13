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

  auto bind_name_id = context.bind_names().Add(
      {.name_id = name_context.name_id_for_new_inst(),
       .enclosing_scope_id = name_context.enclosing_scope_id_for_new_inst()});

  auto alias_id = SemIR::InstId::Invalid;
  if (expr_id.is_builtin()) {
    // Type (`bool`) and value (`false`) literals provided by the builtin
    // structure should be turned into name references.
    alias_id = context.AddInst(
        {name_context.parse_node,
         SemIR::BindAlias{context.insts().Get(expr_id).type_id(), bind_name_id,
                          expr_id}});
  } else if (auto inst = context.insts().TryGetAs<SemIR::NameRef>(expr_id)) {
    // Pass through name references, albeit changing the name in use.
    alias_id = context.AddInst(
        {name_context.parse_node,
         SemIR::BindAlias{inst->type_id, bind_name_id, inst->value_id}});
  } else {
    CARBON_DIAGNOSTIC(AliasRequiresNameRef, Error,
                      "Alias initializer must be a name reference.");
    context.emitter().Emit(expr_node, AliasRequiresNameRef);
    alias_id = context.AddInst(
        {name_context.parse_node,
         SemIR::BindAlias{SemIR::TypeId::Error, bind_name_id, expr_id}});
  }

  // Add the name of the binding to the current scope.
  context.decl_name_stack().PopScope();
  context.decl_name_stack().AddNameToLookup(name_context, alias_id);
  return true;
}

}  // namespace Carbon::Check
