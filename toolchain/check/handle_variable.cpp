// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/modifiers.h"

namespace Carbon::Check {

auto HandleVariableIntroducer(Context& context,
                              Parse::VariableIntroducerId parse_node) -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  context.decl_state_stack().Push(DeclState::Var);
  return true;
}

auto HandleReturnedModifier(Context& context,
                            Parse::ReturnedModifierId parse_node) -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleVariableInitializer(Context& context,
                               Parse::VariableInitializerId parse_node)
    -> bool {
  SemIR::InstId init_id = context.node_stack().PopExpr();
  context.node_stack().Push(parse_node, init_id);
  return true;
}

auto HandleVariableDecl(Context& context, Parse::VariableDeclId parse_node)
    -> bool {
  // Handle the optional initializer.
  std::optional<SemIR::InstId> init_id =
      context.node_stack().PopIf<Parse::NodeKind::VariableInitializer>();

  if (context.node_stack().PeekIs<Parse::NodeKind::TuplePattern>()) {
    return context.TODO(parse_node, "tuple pattern in var");
  }

  // Extract the name binding.
  auto value_id = context.node_stack().PopPattern();
  if (auto bind_name = context.insts().TryGetAs<SemIR::AnyBindName>(value_id)) {
    // Form a corresponding name in the current context, and bind the name to
    // the variable.
    auto name_context = context.decl_name_stack().MakeUnqualifiedName(
        context.insts().GetParseNode(value_id),
        context.bind_names().Get(bind_name->bind_name_id).name_id);
    context.decl_name_stack().AddNameToLookup(name_context, value_id);
    value_id = bind_name->value_id;
  } else if (auto field_decl =
                 context.insts().TryGetAs<SemIR::FieldDecl>(value_id)) {
    // Introduce the field name into the class.
    auto name_context = context.decl_name_stack().MakeUnqualifiedName(
        context.insts().GetParseNode(value_id), field_decl->name_id);
    context.decl_name_stack().AddNameToLookup(name_context, value_id);
  }
  // TODO: Handle other kinds of pattern.

  // Pop the `returned` specifier if present.
  context.node_stack()
      .PopAndDiscardSoloParseNodeIf<Parse::NodeKind::ReturnedModifier>();

  // If there was an initializer, assign it to the storage.
  if (init_id.has_value()) {
    if (context.GetCurrentScopeAs<SemIR::ClassDecl>()) {
      // TODO: In a class scope, we should instead save the initializer
      // somewhere so that we can use it as a default.
      context.TODO(parse_node, "Field initializer");
    } else {
      init_id = Initialize(context, parse_node, value_id, *init_id);
      // TODO: Consider using different instruction kinds for assignment versus
      // initialization.
      context.AddInst({parse_node, SemIR::Assign{value_id, *init_id}});
    }
  }

  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::VariableIntroducer>();

  // Process declaration modifiers.
  // TODO: For a qualified `var` declaration, this should use the target scope
  // of the name introduced in the declaration. See #2590.
  CheckAccessModifiersOnDecl(context, Lex::TokenKind::Var,
                             context.current_scope_id());
  LimitModifiersOnDecl(context, KeywordModifierSet::Access,
                       Lex::TokenKind::Var);
  auto modifiers = context.decl_state_stack().innermost().modifier_set;
  if (!!(modifiers & KeywordModifierSet::Access)) {
    context.TODO(context.decl_state_stack().innermost().saw_access_modifier,
                 "access modifier");
  }

  context.decl_state_stack().Pop(DeclState::Var);

  return true;
}

}  // namespace Carbon::Check
