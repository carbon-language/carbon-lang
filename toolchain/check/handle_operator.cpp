// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleInfixOperator(Context& context, Parse::Node parse_node) -> bool {
  auto rhs_id = context.node_stack().PopExpression();
  auto [lhs_node, lhs_id] = context.node_stack().PopExpressionWithParseNode();

  // Figure out the operator for the token.
  auto token = context.parse_tree().node_token(parse_node);
  switch (auto token_kind = context.tokens().GetKind(token)) {
    case Lex::TokenKind::Plus:
      // TODO: This should search for a compatible interface. For now, it's a
      // very trivial check of validity on the operation.
      lhs_id = context.ConvertToValueOfType(
          parse_node, lhs_id, context.semantics_ir().GetNode(rhs_id).type_id());
      rhs_id = context.ConvertToValueExpression(rhs_id);

      context.AddNodeAndPush(
          parse_node,
          SemIR::Node::BinaryOperatorAdd::Make(
              parse_node, context.semantics_ir().GetNode(lhs_id).type_id(),
              lhs_id, rhs_id));
      return true;

    case Lex::TokenKind::And:
    case Lex::TokenKind::Or: {
      // The first operand is wrapped in a ShortCircuitOperand, which we
      // already handled by creating a RHS block and a resumption block, which
      // are the current block and its enclosing block.
      rhs_id = context.ConvertToBoolValue(parse_node, rhs_id);

      // When the second operand is evaluated, the result of `and` and `or` is
      // its value.
      auto resume_block_id = context.node_block_stack().PeekOrAdd(/*depth=*/1);
      context.AddNode(SemIR::Node::BranchWithArg::Make(
          parse_node, resume_block_id, rhs_id));
      context.node_block_stack().Pop();
      context.AddCurrentCodeBlockToFunction();

      // Collect the result from either the first or second operand.
      context.AddNodeAndPush(
          parse_node,
          SemIR::Node::BlockArg::Make(
              parse_node, context.semantics_ir().GetNode(rhs_id).type_id(),
              resume_block_id));
      return true;
    }
    case Lex::TokenKind::Equal: {
      // TODO: handle complex assignment expression such as `a += 1`.
      if (SemIR::GetExpressionCategory(context.semantics_ir(), lhs_id) !=
          SemIR::ExpressionCategory::DurableReference) {
        CARBON_DIAGNOSTIC(AssignmentToNonAssignable, Error,
                          "Expression is not assignable.");
        context.emitter().Emit(lhs_node, AssignmentToNonAssignable);
      }
      // TODO: Destroy the old value before reinitializing. This will require
      // building the destruction code before we build the RHS subexpression.
      rhs_id = context.Initialize(parse_node, lhs_id, rhs_id);
      context.AddNode(SemIR::Node::Assign::Make(parse_node, lhs_id, rhs_id));
      // We model assignment as an expression, so we need to push a value for
      // it, even though it doesn't produce a value.
      // TODO: Consider changing our parse tree to model assignment as a
      // different kind of statement than an expression statement.
      context.node_stack().Push(parse_node, lhs_id);
      return true;
    }
    default:
      return context.TODO(parse_node, llvm::formatv("Handle {0}", token_kind));
  }
}

auto HandlePostfixOperator(Context& context, Parse::Node parse_node) -> bool {
  auto value_id = context.node_stack().PopExpression();

  // Figure out the operator for the token.
  auto token = context.parse_tree().node_token(parse_node);
  switch (auto token_kind = context.tokens().GetKind(token)) {
    case Lex::TokenKind::Star: {
      auto inner_type_id = context.ExpressionAsType(parse_node, value_id);
      context.AddNodeAndPush(
          parse_node, SemIR::Node::PointerType::Make(
                          parse_node, SemIR::TypeId::TypeType, inner_type_id));
      return true;
    }

    default:
      CARBON_FATAL() << "Unexpected postfix operator " << token_kind;
  }
}

auto HandlePrefixOperator(Context& context, Parse::Node parse_node) -> bool {
  auto value_id = context.node_stack().PopExpression();

  // Figure out the operator for the token.
  auto token = context.parse_tree().node_token(parse_node);
  switch (auto token_kind = context.tokens().GetKind(token)) {
    case Lex::TokenKind::Amp: {
      // Only durable reference expressions can have their address taken.
      switch (SemIR::GetExpressionCategory(context.semantics_ir(), value_id)) {
        case SemIR::ExpressionCategory::DurableReference:
          break;
        case SemIR::ExpressionCategory::EphemeralReference:
          CARBON_DIAGNOSTIC(AddressOfEphemeralReference, Error,
                            "Cannot take the address of a temporary object.");
          context.emitter().Emit(parse_node, AddressOfEphemeralReference);
          break;
        default:
          CARBON_DIAGNOSTIC(
              AddressOfNonReference, Error,
              "Cannot take the address of non-reference expression.");
          context.emitter().Emit(parse_node, AddressOfNonReference);
          break;
      }
      context.AddNodeAndPush(
          parse_node,
          SemIR::Node::AddressOf::Make(
              parse_node,
              context.GetPointerType(
                  parse_node,
                  context.semantics_ir().GetNode(value_id).type_id()),
              value_id));
      return true;
    }

    case Lex::TokenKind::Const: {
      // `const (const T)` is probably not what the developer intended.
      // TODO: Detect `const (const T)*` and suggest moving the `*` inside the
      // parentheses.
      if (context.semantics_ir().GetNode(value_id).kind() ==
          SemIR::NodeKind::ConstType) {
        CARBON_DIAGNOSTIC(RepeatedConst, Warning,
                          "`const` applied repeatedly to the same type has no "
                          "additional effect.");
        context.emitter().Emit(parse_node, RepeatedConst);
      }
      auto inner_type_id = context.ExpressionAsType(parse_node, value_id);
      context.AddNodeAndPush(
          parse_node, SemIR::Node::ConstType::Make(
                          parse_node, SemIR::TypeId::TypeType, inner_type_id));
      return true;
    }

    case Lex::TokenKind::Not:
      value_id = context.ConvertToBoolValue(parse_node, value_id);
      context.AddNodeAndPush(
          parse_node,
          SemIR::Node::UnaryOperatorNot::Make(
              parse_node, context.semantics_ir().GetNode(value_id).type_id(),
              value_id));
      return true;

    case Lex::TokenKind::Star: {
      auto type_id = context.GetUnqualifiedType(
          context.semantics_ir().GetNode(value_id).type_id());
      auto type_node = context.semantics_ir().GetNode(
          context.semantics_ir().GetTypeAllowBuiltinTypes(type_id));
      auto result_type_id = SemIR::TypeId::Error;
      if (type_node.kind() == SemIR::NodeKind::PointerType) {
        result_type_id = type_node.GetAsPointerType();
      } else {
        CARBON_DIAGNOSTIC(
            DereferenceOfNonPointer, Error,
            "Cannot dereference operand of non-pointer type `{0}`.",
            std::string);
        auto builder = context.emitter().Build(
            parse_node, DereferenceOfNonPointer,
            context.semantics_ir().StringifyType(type_id));
        // TODO: Check for any facet here, rather than only a type.
        if (type_id == SemIR::TypeId::TypeType) {
          CARBON_DIAGNOSTIC(
              DereferenceOfType, Note,
              "To form a pointer type, write the `*` after the pointee type.");
          builder.Note(parse_node, DereferenceOfType);
        }
        builder.Emit();
      }
      value_id = context.ConvertToValueExpression(value_id);
      context.AddNodeAndPush(
          parse_node,
          SemIR::Node::Dereference::Make(parse_node, result_type_id, value_id));
      return true;
    }

    default:
      return context.TODO(parse_node, llvm::formatv("Handle {0}", token_kind));
  }
}

auto HandleShortCircuitOperand(Context& context, Parse::Node parse_node)
    -> bool {
  // Convert the condition to `bool`.
  auto cond_value_id = context.node_stack().PopExpression();
  cond_value_id = context.ConvertToBoolValue(parse_node, cond_value_id);
  auto bool_type_id = context.semantics_ir().GetNode(cond_value_id).type_id();

  // Compute the branch value: the condition for `and`, inverted for `or`.
  auto token = context.parse_tree().node_token(parse_node);
  SemIR::NodeId branch_value_id = SemIR::NodeId::Invalid;
  auto short_circuit_result_id = SemIR::NodeId::Invalid;
  switch (auto token_kind = context.tokens().GetKind(token)) {
    case Lex::TokenKind::And:
      branch_value_id = cond_value_id;
      short_circuit_result_id = context.AddNode(SemIR::Node::BoolLiteral::Make(
          parse_node, bool_type_id, SemIR::BoolValue::False));
      break;

    case Lex::TokenKind::Or:
      branch_value_id = context.AddNode(SemIR::Node::UnaryOperatorNot::Make(
          parse_node, bool_type_id, cond_value_id));
      short_circuit_result_id = context.AddNode(SemIR::Node::BoolLiteral::Make(
          parse_node, bool_type_id, SemIR::BoolValue::True));
      break;

    default:
      CARBON_FATAL() << "Unexpected short-circuiting operator " << token_kind;
  }

  // Create a block for the right-hand side and for the continuation.
  auto rhs_block_id =
      context.AddDominatedBlockAndBranchIf(parse_node, branch_value_id);
  auto end_block_id = context.AddDominatedBlockAndBranchWithArg(
      parse_node, short_circuit_result_id);

  // Push the resumption and the right-hand side blocks, and start emitting the
  // right-hand operand.
  context.node_block_stack().Pop();
  context.node_block_stack().Push(end_block_id);
  context.node_block_stack().Push(rhs_block_id);
  context.AddCurrentCodeBlockToFunction();

  // Put the condition back on the stack for HandleInfixOperator.
  context.node_stack().Push(parse_node, cond_value_id);
  return true;
}

}  // namespace Carbon::Check
