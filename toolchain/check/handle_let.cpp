// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleLetDecl(Context& context, Parse::NodeId parse_node) -> bool {
  auto value_id = context.node_stack().PopExpr();
  if (context.node_stack().PeekIs<Parse::NodeKind::TuplePattern>()) {
    return context.TODO(parse_node, "tuple pattern in let");
  }
  SemIR::InstId pattern_id =
      context.node_stack().Pop<Parse::NodeKind::BindingPattern>();
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::LetIntroducer>();
  // Process declaration modifiers.
  CheckAccessModifiersOnDecl(context, Lex::TokenKind::Let);
  RequireDefaultFinalOnlyInInterfaces(context, Lex::TokenKind::Let);
  LimitModifiersOnDecl(
      context, KeywordModifierSet::Access | KeywordModifierSet::Interface,
      Lex::TokenKind::Let);

  auto modifiers = context.decl_state_stack().innermost().modifier_set;
  if (!!(modifiers & KeywordModifierSet::Access)) {
    context.TODO(context.decl_state_stack().innermost().saw_access_modifier,
                 "access modifier");
  }
  if (!!(modifiers & KeywordModifierSet::Interface)) {
    context.TODO(context.decl_state_stack().innermost().saw_decl_modifier,
                 "interface modifier");
  }
  context.decl_state_stack().Pop(DeclState::Let);

  // Convert the value to match the type of the pattern.
  auto pattern = context.insts().Get(pattern_id);
  value_id =
      ConvertToValueOfType(context, parse_node, value_id, pattern.type_id());

  // Update the binding with its value and add it to the current block, after
  // the computation of the value.
  // TODO: Support other kinds of pattern here.
  auto bind_name = pattern.As<SemIR::BindName>();
  CARBON_CHECK(!bind_name.value_id.is_valid())
      << "Binding should not already have a value!";
  bind_name.value_id = value_id;
  context.insts().Set(pattern_id, bind_name);
  context.inst_block_stack().AddInstId(pattern_id);

  // Add the name of the binding to the current scope.
  context.AddNameToLookup(pattern.parse_node(), bind_name.name_id, pattern_id);
  return true;
}

auto HandleLetIntroducer(Context& context, Parse::NodeId parse_node) -> bool {
  context.decl_state_stack().Push(DeclState::Let, parse_node);
  // Push a bracketing node to establish the pattern context.
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleLetInitializer(Context& /*context*/, Parse::NodeId /*parse_node*/)
    -> bool {
  return true;
}

}  // namespace Carbon::Check
