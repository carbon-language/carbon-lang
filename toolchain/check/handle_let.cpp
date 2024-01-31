// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleLetIntroducer(Context& context, Parse::LetIntroducerId parse_node)
    -> bool {
  context.decl_state_stack().Push(DeclState::Let);
  // Push a bracketing node to establish the pattern context.
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleLetInitializer(Context& /*context*/,
                          Parse::LetInitializerId /*parse_node*/) -> bool {
  return true;
}

auto HandleLetDecl(Context& context, Parse::LetDeclId parse_node) -> bool {
  auto value_id = context.node_stack().PopExpr();
  if (context.node_stack().PeekIs<Parse::NodeKind::TuplePattern>()) {
    return context.TODO(parse_node, "tuple pattern in let");
  }
  SemIR::InstId pattern_id = context.node_stack().PopPattern();
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::LetIntroducer>();
  // Process declaration modifiers.
  // TODO: For a qualified `let` declaration, this should use the target scope
  // of the name introduced in the declaration. See #2590.
  CheckAccessModifiersOnDecl(context, Lex::TokenKind::Let,
                             context.current_scope_id());
  RequireDefaultFinalOnlyInInterfaces(context, Lex::TokenKind::Let,
                                      context.current_scope_id());
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
  auto pattern = context.insts().GetWithParseNode(pattern_id);
  value_id = ConvertToValueOfType(context, parse_node, value_id,
                                  pattern.inst.type_id());

  // Update the binding with its value and add it to the current block, after
  // the computation of the value.
  // TODO: Support other kinds of pattern here.
  auto bind_name = pattern.inst.As<SemIR::AnyBindName>();
  CARBON_CHECK(!bind_name.value_id.is_valid())
      << "Binding should not already have a value!";
  bind_name.value_id = value_id;
  pattern.inst = bind_name;
  context.ReplaceInstBeforeConstantUse(pattern_id, pattern);
  context.inst_block_stack().AddInstId(pattern_id);

  // Add the name of the binding to the current scope.
  auto name_id = context.bind_names().Get(bind_name.bind_name_id).name_id;
  context.AddNameToLookup(name_id, pattern_id);
  return true;
}

}  // namespace Carbon::Check
