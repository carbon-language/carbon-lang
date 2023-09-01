// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/node.h"

namespace Carbon::Check {

auto HandleExpressionStatement(Context& context, Parse::Node /*parse_node*/)
    -> bool {
  context.HandleDiscardedExpression(context.node_stack().PopExpression());
  return true;
}

auto HandleReturnStatement(Context& context, Parse::Node parse_node) -> bool {
  CARBON_CHECK(!context.return_scope_stack().empty());
  const auto& fn_node =
      context.semantics_ir().GetNode(context.return_scope_stack().back());
  const auto& callable =
      context.semantics_ir().GetFunction(fn_node.GetAsFunctionDeclaration());

  if (context.parse_tree().node_kind(context.node_stack().PeekParseNode()) ==
      Parse::NodeKind::ReturnStatementStart) {
    context.node_stack()
        .PopAndDiscardSoloParseNode<Parse::NodeKind::ReturnStatementStart>();

    if (callable.return_type_id.is_valid()) {
      // TODO: Add a note pointing at the return type's parse node.
      CARBON_DIAGNOSTIC(ReturnStatementMissingExpression, Error,
                        "Must return a {0}.", std::string);
      context.emitter()
          .Build(parse_node, ReturnStatementMissingExpression,
                 context.semantics_ir().StringifyType(callable.return_type_id))
          .Emit();
    }

    context.AddNode(SemIR::Node::Return::Make(parse_node));
  } else {
    auto arg = context.node_stack().PopExpression();
    context.node_stack()
        .PopAndDiscardSoloParseNode<Parse::NodeKind::ReturnStatementStart>();

    if (!callable.return_type_id.is_valid()) {
      CARBON_DIAGNOSTIC(
          ReturnStatementDisallowExpression, Error,
          "No return expression should be provided in this context.");
      CARBON_DIAGNOSTIC(ReturnStatementImplicitNote, Note,
                        "There was no return type provided.");
      context.emitter()
          .Build(parse_node, ReturnStatementDisallowExpression)
          .Note(fn_node.parse_node(), ReturnStatementImplicitNote)
          .Emit();
    } else if (callable.return_slot_id.is_valid()) {
      arg = context.Initialize(parse_node, callable.return_slot_id, arg);
    } else {
      arg = context.ConvertToValueOfType(parse_node, arg,
                                         callable.return_type_id);
    }

    context.AddNode(SemIR::Node::ReturnExpression::Make(parse_node, arg));
  }

  // Switch to a new, unreachable, empty node block. This typically won't
  // contain any semantics IR, but it can do if there are statements following
  // the `return` statement.
  context.node_block_stack().Pop();
  context.node_block_stack().PushUnreachable();
  return true;
}

auto HandleReturnStatementStart(Context& context, Parse::Node parse_node)
    -> bool {
  // No action, just a bracketing node.
  context.node_stack().Push(parse_node);
  return true;
}

}  // namespace Carbon::Check
