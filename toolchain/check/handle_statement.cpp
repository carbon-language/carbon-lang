// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

// TODO: Find a better home for this. We'll likely need it for more than just
// expression statements.
static auto HandleDiscardedExpression(Context& context, SemIR::InstId expr_id)
    -> void {
  // If we discard an initializing expression, convert it to a value or
  // reference so that it has something to initialize.
  auto expr = context.insts().Get(expr_id);
  Convert(context, expr.parse_node(), expr_id,
          {.kind = ConversionTarget::Discarded, .type_id = expr.type_id()});

  // TODO: This will eventually need to do some "do not discard" analysis.
}

auto HandleExpressionStatement(Context& context, Parse::Lamp /*parse_node*/)
    -> bool {
  HandleDiscardedExpression(context, context.lamp_stack().PopExpression());
  return true;
}

auto HandleReturnStatement(Context& context, Parse::Lamp parse_node) -> bool {
  CARBON_CHECK(!context.return_scope_stack().empty());
  auto fn_node = context.insts().GetAs<SemIR::FunctionDeclaration>(
      context.return_scope_stack().back());
  const auto& callable = context.functions().Get(fn_node.function_id);

  if (context.parse_tree().node_kind(context.lamp_stack().PeekParseNode()) ==
      Parse::LampKind::ReturnStatementStart) {
    context.lamp_stack()
        .PopAndDiscardSoloParseNode<Parse::LampKind::ReturnStatementStart>();

    if (callable.return_type_id.is_valid()) {
      // TODO: Add a note pointing at the return type's parse node.
      CARBON_DIAGNOSTIC(ReturnStatementMissingExpression, Error,
                        "Must return a {0}.", std::string);
      context.emitter()
          .Build(parse_node, ReturnStatementMissingExpression,
                 context.sem_ir().StringifyType(callable.return_type_id))
          .Emit();
    }

    context.AddNode(SemIR::Return{parse_node});
  } else {
    auto arg = context.lamp_stack().PopExpression();
    context.lamp_stack()
        .PopAndDiscardSoloParseNode<Parse::LampKind::ReturnStatementStart>();

    if (!callable.return_type_id.is_valid()) {
      CARBON_DIAGNOSTIC(
          ReturnStatementDisallowExpression, Error,
          "No return expression should be provided in this context.");
      CARBON_DIAGNOSTIC(ReturnStatementImplicitNote, Note,
                        "There was no return type provided.");
      context.emitter()
          .Build(parse_node, ReturnStatementDisallowExpression)
          .Note(fn_node.parse_node, ReturnStatementImplicitNote)
          .Emit();
    } else if (callable.return_slot_id.is_valid()) {
      arg = Initialize(context, parse_node, callable.return_slot_id, arg);
    } else {
      arg = ConvertToValueOfType(context, parse_node, arg,
                                 callable.return_type_id);
    }

    context.AddNode(SemIR::ReturnExpression{parse_node, arg});
  }

  // Switch to a new, unreachable, empty node block. This typically won't
  // contain any semantics IR, but it can do if there are statements following
  // the `return` statement.
  context.inst_block_stack().Pop();
  context.inst_block_stack().PushUnreachable();
  return true;
}

auto HandleReturnStatementStart(Context& context, Parse::Lamp parse_node)
    -> bool {
  // No action, just a bracketing node.
  context.lamp_stack().Push(parse_node);
  return true;
}

}  // namespace Carbon::Check
