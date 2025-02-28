// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/name_component.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::AliasIntroducerId /*node_id*/)
    -> bool {
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Alias>();
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

auto HandleParseNode(Context& /*context*/,
                     Parse::AliasInitializerId /*node_id*/) -> bool {
  return true;
}

auto HandleParseNode(Context& context, Parse::AliasId /*node_id*/) -> bool {
  auto [expr_node, expr_id] = context.node_stack().PopExprWithNodeId();

  auto name_context = context.decl_name_stack().FinishName(
      PopNameComponentWithoutParams(context, Lex::TokenKind::Alias));

  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Alias>();
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::Access);

  auto entity_name_id = context.entity_names().Add(
      {.name_id = name_context.name_id_for_new_inst(),
       .parent_scope_id = name_context.parent_scope_id});

  auto alias_type_id = SemIR::TypeId::None;
  auto alias_value_id = SemIR::InstId::None;
  if (SemIR::IsSingletonInstId(expr_id)) {
    // Type (`bool`) and value (`false`) literals provided by the builtin
    // structure should be turned into name references.
    // TODO: Look into handling `false`, this doesn't do it right now because it
    // sees a value instruction instead of a builtin.
    alias_type_id = context.insts().Get(expr_id).type_id();
    alias_value_id = expr_id;
  } else if (auto inst = context.insts().TryGetAs<SemIR::NameRef>(expr_id)) {
    // Pass through name references, albeit changing the name in use.
    alias_type_id = inst->type_id;
    alias_value_id = inst->value_id;
  } else {
    CARBON_DIAGNOSTIC(AliasRequiresNameRef, Error,
                      "alias initializer must be a name reference");
    context.emitter().Emit(expr_node, AliasRequiresNameRef);
    alias_type_id = SemIR::ErrorInst::SingletonTypeId;
    alias_value_id = SemIR::ErrorInst::SingletonInstId;
  }
  auto alias_id = AddInst<SemIR::BindAlias>(context, name_context.loc_id,
                                            {.type_id = alias_type_id,
                                             .entity_name_id = entity_name_id,
                                             .value_id = alias_value_id});

  // Add the name of the binding to the current scope.
  context.decl_name_stack().PopScope();
  context.decl_name_stack().AddNameOrDiagnose(
      name_context, alias_id, introducer.modifier_set.GetAccessKind());
  return true;
}

}  // namespace Carbon::Check
