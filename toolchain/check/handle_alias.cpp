// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleAliasIntroducer(Context& context,
                           Parse::AliasIntroducerId /*node_id*/) -> bool {
  context.decl_state_stack().Push(DeclState::Alias);
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

auto HandleAliasInitializer(Context& /*context*/,
                            Parse::AliasInitializerId /*node_id*/) -> bool {
  return true;
}

auto HandleAlias(Context& context, Parse::AliasId /*node_id*/) -> bool {
  auto [expr_node, expr_id] = context.node_stack().PopExprWithNodeId();

  auto name_context = context.decl_name_stack().FinishName();

  LimitModifiersOnDecl(context, KeywordModifierSet::Access,
                       Lex::TokenKind::Alias);
  auto modifiers = context.decl_state_stack().innermost().modifier_set;
  if (!!(modifiers & KeywordModifierSet::Access)) {
    context.TODO(context.decl_state_stack().innermost().modifier_node_id(
                     ModifierOrder::Access),
                 "access modifier");
  }
  context.decl_state_stack().Pop(DeclState::Alias);

  auto bind_name_id = context.bind_names().Add(
      {.name_id = name_context.name_id_for_new_inst(),
       .enclosing_scope_id = name_context.enclosing_scope_id_for_new_inst()});

  auto alias_id = SemIR::InstId::Invalid;
  if (expr_id.is_builtin()) {
    // Type (`bool`) and value (`false`) literals provided by the builtin
    // structure should be turned into name references.
    // TODO: Look into handling `false`, this doesn't do it right now because it
    // sees a value instruction instead of a builtin.
    alias_id = context.AddInst(
        {name_context.loc_id,
         SemIR::BindAlias{context.insts().Get(expr_id).type_id(), bind_name_id,
                          expr_id}});
  } else if (auto inst = context.insts().TryGetAs<SemIR::NameRef>(expr_id)) {
    // Pass through name references, albeit changing the name in use.
    alias_id = context.AddInst(
        {name_context.loc_id,
         SemIR::BindAlias{inst->type_id, bind_name_id, inst->value_id}});
  } else {
    CARBON_DIAGNOSTIC(AliasRequiresNameRef, Error,
                      "Alias initializer must be a name reference.");
    context.emitter().Emit(expr_node, AliasRequiresNameRef);
    alias_id =
        context.AddInst({name_context.loc_id,
                         SemIR::BindAlias{SemIR::TypeId::Error, bind_name_id,
                                          SemIR::InstId::BuiltinError}});
  }

  // Add the name of the binding to the current scope.
  context.decl_name_stack().PopScope();
  context.decl_name_stack().AddNameToLookup(name_context, alias_id);
  return true;
}

}  // namespace Carbon::Check
