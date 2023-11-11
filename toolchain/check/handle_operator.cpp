// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"

namespace Carbon::Check {

auto HandleInfixOperator(Context& context, Parse::Node parse_node) -> bool {
  auto [rhs_node, rhs_id] = context.node_stack().PopExprWithParseNode();
  auto [lhs_node, lhs_id] = context.node_stack().PopExprWithParseNode();

  // Figure out the operator for the token.
  auto token = context.parse_tree().node_token(parse_node);
  switch (auto token_kind = context.tokens().GetKind(token)) {
    case Lex::TokenKind::Plus:
      // TODO: This should search for a compatible interface. For now, it's a
      // very trivial check of validity on the operation.
      lhs_id = ConvertToValueOfType(context, parse_node, lhs_id,
                                    context.insts().Get(rhs_id).type_id());
      rhs_id = ConvertToValueExpr(context, rhs_id);

      context.AddInstAndPush(
          parse_node, SemIR::BinaryOperatorAdd{
                          parse_node, context.insts().Get(lhs_id).type_id(),
                          lhs_id, rhs_id});
      return true;

    case Lex::TokenKind::And:
    case Lex::TokenKind::Or: {
      // The first operand is wrapped in a ShortCircuitOperand, which we
      // already handled by creating a RHS block and a resumption block, which
      // are the current block and its enclosing block.
      rhs_id = ConvertToBoolValue(context, parse_node, rhs_id);

      // When the second operand is evaluated, the result of `and` and `or` is
      // its value.
      auto resume_block_id = context.inst_block_stack().PeekOrAdd(/*depth=*/1);
      context.AddInst(
          SemIR::BranchWithArg{parse_node, resume_block_id, rhs_id});
      context.inst_block_stack().Pop();
      context.AddCurrentCodeBlockToFunction();

      // Collect the result from either the first or second operand.
      context.AddInstAndPush(
          parse_node,
          SemIR::BlockArg{parse_node, context.insts().Get(rhs_id).type_id(),
                          resume_block_id});
      return true;
    }
    case Lex::TokenKind::As: {
      auto rhs_type_id = ExprAsType(context, rhs_node, rhs_id);
      context.node_stack().Push(
          parse_node,
          ConvertForExplicitAs(context, parse_node, lhs_id, rhs_type_id));
      return true;
    }
    case Lex::TokenKind::Equal: {
      // TODO: handle complex assignment expression such as `a += 1`.
      if (auto lhs_cat = SemIR::GetExprCategory(context.sem_ir(), lhs_id);
          lhs_cat != SemIR::ExprCategory::DurableReference &&
          lhs_cat != SemIR::ExprCategory::Error) {
        CARBON_DIAGNOSTIC(AssignmentToNonAssignable, Error,
                          "Expression is not assignable.");
        context.emitter().Emit(lhs_node, AssignmentToNonAssignable);
      }
      // TODO: Destroy the old value before reinitializing. This will require
      // building the destruction code before we build the RHS subexpression.
      rhs_id = Initialize(context, parse_node, lhs_id, rhs_id);
      context.AddInst(SemIR::Assign{parse_node, lhs_id, rhs_id});
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
  auto value_id = context.node_stack().PopExpr();

  // Figure out the operator for the token.
  auto token = context.parse_tree().node_token(parse_node);
  switch (auto token_kind = context.tokens().GetKind(token)) {
    case Lex::TokenKind::Star: {
      auto inner_type_id = ExprAsType(context, parse_node, value_id);
      context.AddInstAndPush(
          parse_node, SemIR::PointerType{parse_node, SemIR::TypeId::TypeType,
                                         inner_type_id});
      return true;
    }

    default:
      CARBON_FATAL() << "Unexpected postfix operator " << token_kind;
  }
}

auto HandlePrefixOperator(Context& context, Parse::Node parse_node) -> bool {
  auto value_id = context.node_stack().PopExpr();

  // Figure out the operator for the token.
  auto token = context.parse_tree().node_token(parse_node);
  switch (auto token_kind = context.tokens().GetKind(token)) {
    case Lex::TokenKind::Amp: {
      // Only durable reference expressions can have their address taken.
      switch (SemIR::GetExprCategory(context.sem_ir(), value_id)) {
        case SemIR::ExprCategory::DurableReference:
        case SemIR::ExprCategory::Error:
          break;
        case SemIR::ExprCategory::EphemeralReference:
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
      context.AddInstAndPush(
          parse_node,
          SemIR::AddressOf{
              parse_node,
              context.GetPointerType(parse_node,
                                     context.insts().Get(value_id).type_id()),
              value_id});
      return true;
    }

    case Lex::TokenKind::Const: {
      // `const (const T)` is probably not what the developer intended.
      // TODO: Detect `const (const T)*` and suggest moving the `*` inside the
      // parentheses.
      if (context.insts().Get(value_id).kind() == SemIR::ConstType::Kind) {
        CARBON_DIAGNOSTIC(RepeatedConst, Warning,
                          "`const` applied repeatedly to the same type has no "
                          "additional effect.");
        context.emitter().Emit(parse_node, RepeatedConst);
      }
      auto inner_type_id = ExprAsType(context, parse_node, value_id);
      context.AddInstAndPush(
          parse_node,
          SemIR::ConstType{parse_node, SemIR::TypeId::TypeType, inner_type_id});
      return true;
    }

    case Lex::TokenKind::Not:
      value_id = ConvertToBoolValue(context, parse_node, value_id);
      context.AddInstAndPush(
          parse_node,
          SemIR::UnaryOperatorNot{
              parse_node, context.insts().Get(value_id).type_id(), value_id});
      return true;

    case Lex::TokenKind::Star: {
      value_id = ConvertToValueExpr(context, value_id);
      auto type_id =
          context.GetUnqualifiedType(context.insts().Get(value_id).type_id());
      auto type_inst = context.insts().Get(
          context.sem_ir().GetTypeAllowBuiltinTypes(type_id));
      auto result_type_id = SemIR::TypeId::Error;
      if (auto pointer_type = type_inst.TryAs<SemIR::PointerType>()) {
        result_type_id = pointer_type->pointee_id;
      } else if (type_id != SemIR::TypeId::Error) {
        CARBON_DIAGNOSTIC(
            DereferenceOfNonPointer, Error,
            "Cannot dereference operand of non-pointer type `{0}`.",
            std::string);
        auto builder =
            context.emitter().Build(parse_node, DereferenceOfNonPointer,
                                    context.sem_ir().StringifyType(type_id));
        // TODO: Check for any facet here, rather than only a type.
        if (type_id == SemIR::TypeId::TypeType) {
          CARBON_DIAGNOSTIC(
              DereferenceOfType, Note,
              "To form a pointer type, write the `*` after the pointee type.");
          builder.Note(parse_node, DereferenceOfType);
        }
        builder.Emit();
      }
      context.AddInstAndPush(
          parse_node, SemIR::Dereference{parse_node, result_type_id, value_id});
      return true;
    }

    default:
      return context.TODO(parse_node, llvm::formatv("Handle {0}", token_kind));
  }
}

auto HandleShortCircuitOperand(Context& context, Parse::Node parse_node)
    -> bool {
  // Convert the condition to `bool`.
  auto cond_value_id = context.node_stack().PopExpr();
  cond_value_id = ConvertToBoolValue(context, parse_node, cond_value_id);
  auto bool_type_id = context.insts().Get(cond_value_id).type_id();

  // Compute the branch value: the condition for `and`, inverted for `or`.
  auto token = context.parse_tree().node_token(parse_node);
  SemIR::InstId branch_value_id = SemIR::InstId::Invalid;
  auto short_circuit_result_id = SemIR::InstId::Invalid;
  switch (auto token_kind = context.tokens().GetKind(token)) {
    case Lex::TokenKind::And:
      branch_value_id = cond_value_id;
      short_circuit_result_id = context.AddInst(SemIR::BoolLiteral{
          parse_node, bool_type_id, SemIR::BoolValue::False});
      break;

    case Lex::TokenKind::Or:
      branch_value_id = context.AddInst(
          SemIR::UnaryOperatorNot{parse_node, bool_type_id, cond_value_id});
      short_circuit_result_id = context.AddInst(
          SemIR::BoolLiteral{parse_node, bool_type_id, SemIR::BoolValue::True});
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
  context.inst_block_stack().Pop();
  context.inst_block_stack().Push(end_block_id);
  context.inst_block_stack().Push(rhs_block_id);
  context.AddCurrentCodeBlockToFunction();

  // Put the condition back on the stack for HandleInfixOperator.
  context.node_stack().Push(parse_node, cond_value_id);
  return true;
}

}  // namespace Carbon::Check
