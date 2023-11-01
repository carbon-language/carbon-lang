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
  Convert(context, expr.parse_lamp(), expr_id,
          {.kind = ConversionTarget::Discarded, .type_id = expr.type_id()});

  // TODO: This will eventually need to do some "do not discard" analysis.
}

auto HandleExpressionStatement(Context& context, Parse::Lamp /*parse_lamp*/)
    -> bool {
  HandleDiscardedExpression(context, context.lamp_stack().PopExpression());
  return true;
}

auto HandleReturnStatement(Context& context, Parse::Lamp parse_lamp) -> bool {
  CARBON_CHECK(!context.return_scope_stack().empty());
  auto fn_inst = context.insts().GetAs<SemIR::FunctionDeclaration>(
      context.return_scope_stack().back());
  const auto& callable = context.functions().Get(fn_inst.function_id);

  if (context.parse_tree().node_kind(context.lamp_stack().PeekParseLamp()) ==
      Parse::LampKind::ReturnStatementStart) {
    context.lamp_stack()
        .PopAndDiscardSoloParseLamp<Parse::LampKind::ReturnStatementStart>();

    if (callable.return_type_id.is_valid()) {
      // TODO: Add a note pointing at the return type's parse node.
      CARBON_DIAGNOSTIC(ReturnStatementMissingExpression, Error,
                        "Must return a {0}.", std::string);
      context.emitter()
          .Build(parse_lamp, ReturnStatementMissingExpression,
                 context.sem_ir().StringifyType(callable.return_type_id))
          .Emit();
    }

    context.AddInst(SemIR::Return{parse_lamp});
  } else {
    auto arg = context.lamp_stack().PopExpression();
    context.lamp_stack()
        .PopAndDiscardSoloParseLamp<Parse::LampKind::ReturnStatementStart>();

    if (!callable.return_type_id.is_valid()) {
      CARBON_DIAGNOSTIC(
          ReturnStatementDisallowExpression, Error,
          "No return expression should be provided in this context.");
      CARBON_DIAGNOSTIC(ReturnStatementImplicitNote, Note,
                        "There was no return type provided.");
      context.emitter()
          .Build(parse_lamp, ReturnStatementDisallowExpression)
          .Note(fn_inst.parse_lamp, ReturnStatementImplicitNote)
          .Emit();
    } else if (callable.return_slot_id.is_valid()) {
      arg = Initialize(context, parse_lamp, callable.return_slot_id, arg);
    } else {
      arg = ConvertToValueOfType(context, parse_lamp, arg,
                                 callable.return_type_id);
    }

    context.AddInst(SemIR::ReturnExpression{parse_lamp, arg});
  }

  // Switch to a new, unreachable, empty inst block. This typically won't
  // contain any semantics IR, but it can do if there are statements following
  // the `return` statement.
  context.inst_block_stack().Pop();
  context.inst_block_stack().PushUnreachable();
  return true;
}

auto HandleReturnStatementStart(Context& context, Parse::Lamp parse_lamp)
    -> bool {
  // No action, just a bracketing node.
  context.lamp_stack().Push(parse_lamp);
  return true;
}

}  // namespace Carbon::Check
