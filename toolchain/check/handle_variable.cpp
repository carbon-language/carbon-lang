// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleVariableIntroducer(Context& context, Parse::NodeId parse_node)
    -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  context.decl_state_stack().Push(DeclState::Var, parse_node);
  return true;
}

auto HandleReturnedModifier(Context& context, Parse::NodeId parse_node)
    -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleVariableInitializer(Context& context, Parse::NodeId parse_node)
    -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleVariableDecl(Context& context, Parse::NodeId parse_node) -> bool {
  // Handle the optional initializer.
  auto init_id = SemIR::InstId::Invalid;
  Parse::NodeKind next_kind =
      context.parse_tree().node_kind(context.node_stack().PeekParseNode());
  if (next_kind == Parse::NodeKind::TuplePattern) {
    return context.TODO(parse_node, "tuple pattern in var");
  }
  // TODO: find a more robust way to determine if there was an initializer.
  bool has_init = next_kind != Parse::NodeKind::BindingPattern;
  if (has_init) {
    init_id = context.node_stack().PopExpr();
    context.node_stack()
        .PopAndDiscardSoloParseNode<Parse::NodeKind::VariableInitializer>();
  }

  if (context.node_stack().PeekIs<Parse::NodeKind::TuplePattern>()) {
    return context.TODO(parse_node, "tuple pattern in var");
  }

  // Extract the name binding.
  auto value_id = context.node_stack().Pop<Parse::NodeKind::BindingPattern>();
  if (auto bind_name = context.insts().Get(value_id).TryAs<SemIR::BindName>()) {
    // Form a corresponding name in the current context, and bind the name to
    // the variable.
    context.decl_name_stack().AddNameToLookup(
        context.decl_name_stack().MakeUnqualifiedName(bind_name->parse_node,
                                                      bind_name->name_id),
        value_id);
    value_id = bind_name->value_id;
  }

  // Pop the `returned` specifier if present.
  context.node_stack()
      .PopAndDiscardSoloParseNodeIf<Parse::NodeKind::ReturnedModifier>();

  // If there was an initializer, assign it to the storage.
  if (has_init) {
    if (context.GetCurrentScopeAs<SemIR::ClassDecl>()) {
      // TODO: In a class scope, we should instead save the initializer
      // somewhere so that we can use it as a default.
      context.TODO(parse_node, "Field initializer");
    } else {
      init_id = Initialize(context, parse_node, value_id, init_id);
      // TODO: Consider using different instruction kinds for assignment versus
      // initialization.
      context.AddInst(SemIR::Assign{parse_node, value_id, init_id});
    }
  }

  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::VariableIntroducer>();

  // Process declaration modifiers.
  CheckAccessModifiersOnDecl(context, Lex::TokenKind::Var);
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
