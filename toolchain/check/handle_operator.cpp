// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/operator.h"
#include "toolchain/check/pointer_dereference.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon::Check {

// Common logic for unary operator handlers.
static auto HandleUnaryOperator(Context& context, Parse::AnyExprId expr_node_id,
                                Operator op) -> bool {
  auto operand_id = context.node_stack().PopExpr();
  auto result_id = BuildUnaryOperator(context, expr_node_id, op, operand_id);
  context.node_stack().Push(expr_node_id, result_id);
  return true;
}

// Common logic for binary operator handlers.
static auto HandleBinaryOperator(Context& context,
                                 Parse::AnyExprId expr_node_id, Operator op)
    -> bool {
  auto rhs_id = context.node_stack().PopExpr();
  auto lhs_id = context.node_stack().PopExpr();
  auto result_id =
      BuildBinaryOperator(context, expr_node_id, op, lhs_id, rhs_id);
  context.node_stack().Push(expr_node_id, result_id);
  return true;
}

auto HandleInfixOperatorAmp(Context& context, Parse::InfixOperatorAmpId node_id)
    -> bool {
  // TODO: Facet type intersection may need to be handled directly.
  return HandleBinaryOperator(context, node_id, {"BitAnd"});
}

auto HandleInfixOperatorAmpEqual(Context& context,
                                 Parse::InfixOperatorAmpEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"BitAndAssign"});
}

auto HandleInfixOperatorAs(Context& context, Parse::InfixOperatorAsId node_id)
    -> bool {
  auto [rhs_node, rhs_id] = context.node_stack().PopExprWithNodeId();
  auto [lhs_node, lhs_id] = context.node_stack().PopExprWithNodeId();

  auto rhs_type_id = ExprAsType(context, rhs_node, rhs_id);
  context.node_stack().Push(
      node_id, ConvertForExplicitAs(context, node_id, lhs_id, rhs_type_id));
  return true;
}

auto HandleInfixOperatorCaret(Context& context,
                              Parse::InfixOperatorCaretId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"BitXor"});
}

auto HandleInfixOperatorCaretEqual(Context& context,
                                   Parse::InfixOperatorCaretEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"BitXorAssign"});
}

auto HandleInfixOperatorEqual(Context& context,
                              Parse::InfixOperatorEqualId node_id) -> bool {
  // TODO: Switch to using assignment interface for most assignment. Some cases
  // may need to be handled directly.
  //
  //   return HandleBinaryOperator(context, node_id, {"Assign"});

  auto [rhs_node, rhs_id] = context.node_stack().PopExprWithNodeId();
  auto [lhs_node, lhs_id] = context.node_stack().PopExprWithNodeId();
  if (auto lhs_cat = SemIR::GetExprCategory(context.sem_ir(), lhs_id);
      lhs_cat != SemIR::ExprCategory::DurableRef &&
      lhs_cat != SemIR::ExprCategory::Error) {
    CARBON_DIAGNOSTIC(AssignmentToNonAssignable, Error,
                      "Expression is not assignable.");
    context.emitter().Emit(lhs_node, AssignmentToNonAssignable);
  }
  // TODO: Destroy the old value before reinitializing. This will require
  // building the destruction code before we build the RHS subexpression.
  rhs_id = Initialize(context, node_id, lhs_id, rhs_id);
  context.AddInst({node_id, SemIR::Assign{lhs_id, rhs_id}});
  // We model assignment as an expression, so we need to push a value for
  // it, even though it doesn't produce a value.
  // TODO: Consider changing our parse tree to model assignment as a
  // different kind of statement than an expression statement.
  context.node_stack().Push(node_id, lhs_id);
  return true;
}

auto HandleInfixOperatorEqualEqual(Context& context,
                                   Parse::InfixOperatorEqualEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"Eq", "Equal"});
}

auto HandleInfixOperatorExclaimEqual(Context& context,
                                     Parse::InfixOperatorExclaimEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"Eq", "NotEqual"});
}

auto HandleInfixOperatorGreater(Context& context,
                                Parse::InfixOperatorGreaterId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"Ordered", "Greater"});
}

auto HandleInfixOperatorGreaterEqual(Context& context,
                                     Parse::InfixOperatorGreaterEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id,
                              {"Ordered", "GreaterOrEquivalent"});
}

auto HandleInfixOperatorGreaterGreater(
    Context& context, Parse::InfixOperatorGreaterGreaterId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"RightShift"});
}

auto HandleInfixOperatorGreaterGreaterEqual(
    Context& context, Parse::InfixOperatorGreaterGreaterEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"RightShiftAssign"});
}

auto HandleInfixOperatorLess(Context& context,
                             Parse::InfixOperatorLessId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"Ordered", "Less"});
}

auto HandleInfixOperatorLessEqual(Context& context,
                                  Parse::InfixOperatorLessEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id,
                              {"Ordered", "LessOrEquivalent"});
}

auto HandleInfixOperatorLessEqualGreater(
    Context& context, Parse::InfixOperatorLessEqualGreaterId node_id) -> bool {
  return context.TODO(node_id, "remove <=> operator that is not in the design");
}

auto HandleInfixOperatorLessLess(Context& context,
                                 Parse::InfixOperatorLessLessId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"LeftShift"});
}

auto HandleInfixOperatorLessLessEqual(
    Context& context, Parse::InfixOperatorLessLessEqualId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"LeftShiftAssign"});
}

auto HandleInfixOperatorMinus(Context& context,
                              Parse::InfixOperatorMinusId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"Sub"});
}

auto HandleInfixOperatorMinusEqual(Context& context,
                                   Parse::InfixOperatorMinusEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"SubAssign"});
}

auto HandleInfixOperatorPercent(Context& context,
                                Parse::InfixOperatorPercentId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"Mod"});
}

auto HandleInfixOperatorPercentEqual(Context& context,
                                     Parse::InfixOperatorPercentEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"ModAssign"});
}

auto HandleInfixOperatorPipe(Context& context,
                             Parse::InfixOperatorPipeId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"BitOr"});
}

auto HandleInfixOperatorPipeEqual(Context& context,
                                  Parse::InfixOperatorPipeEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"BitOrAssign"});
}

auto HandleInfixOperatorPlus(Context& context,
                             Parse::InfixOperatorPlusId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"Add"});
}

auto HandleInfixOperatorPlusEqual(Context& context,
                                  Parse::InfixOperatorPlusEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"AddAssign"});
}

auto HandleInfixOperatorSlash(Context& context,
                              Parse::InfixOperatorSlashId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"Div"});
}

auto HandleInfixOperatorSlashEqual(Context& context,
                                   Parse::InfixOperatorSlashEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"DivAssign"});
}

auto HandleInfixOperatorStar(Context& context,
                             Parse::InfixOperatorStarId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"Mul"});
}

auto HandleInfixOperatorStarEqual(Context& context,
                                  Parse::InfixOperatorStarEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"MulAssign"});
}

auto HandlePostfixOperatorStar(Context& context,
                               Parse::PostfixOperatorStarId node_id) -> bool {
  auto value_id = context.node_stack().PopExpr();
  auto inner_type_id = ExprAsType(context, node_id, value_id);
  context.AddInstAndPush(
      {node_id, SemIR::PointerType{SemIR::TypeId::TypeType, inner_type_id}});
  return true;
}

auto HandlePrefixOperatorAmp(Context& context,
                             Parse::PrefixOperatorAmpId node_id) -> bool {
  auto value_id = context.node_stack().PopExpr();
  auto type_id = context.insts().Get(value_id).type_id();
  // Only durable reference expressions can have their address taken.
  switch (SemIR::GetExprCategory(context.sem_ir(), value_id)) {
    case SemIR::ExprCategory::DurableRef:
    case SemIR::ExprCategory::Error:
      break;
    case SemIR::ExprCategory::EphemeralRef:
      CARBON_DIAGNOSTIC(AddrOfEphemeralRef, Error,
                        "Cannot take the address of a temporary object.");
      context.emitter().Emit(TokenOnly(node_id), AddrOfEphemeralRef);
      value_id = SemIR::InstId::BuiltinError;
      break;
    default:
      CARBON_DIAGNOSTIC(AddrOfNonRef, Error,
                        "Cannot take the address of non-reference expression.");
      context.emitter().Emit(TokenOnly(node_id), AddrOfNonRef);
      value_id = SemIR::InstId::BuiltinError;
      break;
  }
  context.AddInstAndPush(
      {node_id, SemIR::AddrOf{context.GetPointerType(type_id), value_id}});
  return true;
}

auto HandlePrefixOperatorCaret(Context& context,
                               Parse::PrefixOperatorCaretId node_id) -> bool {
  return HandleUnaryOperator(context, node_id, {"BitComplement"});
}

auto HandlePrefixOperatorConst(Context& context,
                               Parse::PrefixOperatorConstId node_id) -> bool {
  auto value_id = context.node_stack().PopExpr();

  // `const (const T)` is probably not what the developer intended.
  // TODO: Detect `const (const T)*` and suggest moving the `*` inside the
  // parentheses.
  if (context.insts().Get(value_id).kind() == SemIR::ConstType::Kind) {
    CARBON_DIAGNOSTIC(RepeatedConst, Warning,
                      "`const` applied repeatedly to the same type has no "
                      "additional effect.");
    context.emitter().Emit(node_id, RepeatedConst);
  }
  auto inner_type_id = ExprAsType(context, node_id, value_id);
  context.AddInstAndPush(
      {node_id, SemIR::ConstType{SemIR::TypeId::TypeType, inner_type_id}});
  return true;
}

auto HandlePrefixOperatorMinus(Context& context,
                               Parse::PrefixOperatorMinusId node_id) -> bool {
  return HandleUnaryOperator(context, node_id, {"Negate"});
}

auto HandlePrefixOperatorMinusMinus(Context& context,
                                    Parse::PrefixOperatorMinusMinusId node_id)
    -> bool {
  return HandleUnaryOperator(context, node_id, {"Dec"});
}

auto HandlePrefixOperatorNot(Context& context,
                             Parse::PrefixOperatorNotId node_id) -> bool {
  auto value_id = context.node_stack().PopExpr();
  value_id = ConvertToBoolValue(context, node_id, value_id);
  context.AddInstAndPush(
      {node_id, SemIR::UnaryOperatorNot{context.insts().Get(value_id).type_id(),
                                        value_id}});
  return true;
}

auto HandlePrefixOperatorPlusPlus(Context& context,
                                  Parse::PrefixOperatorPlusPlusId node_id)
    -> bool {
  return HandleUnaryOperator(context, node_id, {"Inc"});
}

auto HandlePrefixOperatorStar(Context& context,
                              Parse::PrefixOperatorStarId node_id) -> bool {
  auto base_id = context.node_stack().PopExpr();

  auto deref_base_id = PerformPointerDereference(
      context, node_id, base_id,
      [&context, &node_id](SemIR::TypeId not_pointer_type_id) {
        CARBON_DIAGNOSTIC(
            DerefOfNonPointer, Error,
            "Cannot dereference operand of non-pointer type `{0}`.",
            SemIR::TypeId);

        auto builder = context.emitter().Build(
            TokenOnly(node_id), DerefOfNonPointer, not_pointer_type_id);

        // TODO: Check for any facet here, rather than only a type.
        if (not_pointer_type_id == SemIR::TypeId::TypeType) {
          CARBON_DIAGNOSTIC(
              DerefOfType, Note,
              "To form a pointer type, write the `*` after the pointee type.");
          builder.Note(TokenOnly(node_id), DerefOfType);
        }

        builder.Emit();
      });

  context.node_stack().Push(node_id, deref_base_id);
  return true;
}

// Adds the branch for a short circuit operand.
static auto HandleShortCircuitOperand(Context& context, Parse::NodeId node_id,
                                      bool is_or) -> bool {
  // Convert the condition to `bool`.
  auto [cond_node, cond_value_id] = context.node_stack().PopExprWithNodeId();
  cond_value_id = ConvertToBoolValue(context, node_id, cond_value_id);
  auto bool_type_id = context.insts().Get(cond_value_id).type_id();

  // Compute the branch value: the condition for `and`, inverted for `or`.
  SemIR::InstId branch_value_id =
      is_or ? context.AddInst({node_id, SemIR::UnaryOperatorNot{bool_type_id,
                                                                cond_value_id}})
            : cond_value_id;
  auto short_circuit_result_id = context.AddInst(
      {node_id,
       SemIR::BoolLiteral{bool_type_id, SemIR::BoolValue::From(is_or)}});

  // Create a block for the right-hand side and for the continuation.
  auto rhs_block_id =
      context.AddDominatedBlockAndBranchIf(node_id, branch_value_id);
  auto end_block_id = context.AddDominatedBlockAndBranchWithArg(
      node_id, short_circuit_result_id);

  // Push the branch condition and result for use when handling the complete
  // expression.
  context.node_stack().Push(cond_node, branch_value_id);
  context.node_stack().Push(cond_node, short_circuit_result_id);

  // Push the resumption and the right-hand side blocks, and start emitting the
  // right-hand operand.
  context.inst_block_stack().Pop();
  context.inst_block_stack().Push(end_block_id);
  context.inst_block_stack().Push(rhs_block_id);
  context.AddCurrentCodeBlockToFunction(node_id);

  // HandleShortCircuitOperator will follow, and doesn't need the operand on the
  // node stack.
  return true;
}

auto HandleShortCircuitOperandAnd(Context& context,
                                  Parse::ShortCircuitOperandAndId node_id)
    -> bool {
  return HandleShortCircuitOperand(context, node_id, /*is_or=*/false);
}

auto HandleShortCircuitOperandOr(Context& context,
                                 Parse::ShortCircuitOperandOrId node_id)
    -> bool {
  return HandleShortCircuitOperand(context, node_id, /*is_or=*/true);
}

// Short circuit operator handling is uniform because the branching logic
// occurs during operand handling.
static auto HandleShortCircuitOperator(Context& context, Parse::NodeId node_id)
    -> bool {
  auto [rhs_node, rhs_id] = context.node_stack().PopExprWithNodeId();
  auto short_circuit_result_id = context.node_stack().PopExpr();
  auto branch_value_id = context.node_stack().PopExpr();

  // The first operand is wrapped in a ShortCircuitOperand, which we
  // already handled by creating a RHS block and a resumption block, which
  // are the current block and its enclosing block.
  rhs_id = ConvertToBoolValue(context, node_id, rhs_id);

  // When the second operand is evaluated, the result of `and` and `or` is
  // its value.
  auto resume_block_id = context.inst_block_stack().PeekOrAdd(/*depth=*/1);
  context.AddInst({node_id, SemIR::BranchWithArg{resume_block_id, rhs_id}});
  context.inst_block_stack().Pop();
  context.AddCurrentCodeBlockToFunction(node_id);

  // Collect the result from either the first or second operand.
  auto result_id = context.AddInst(
      {node_id, SemIR::BlockArg{context.insts().Get(rhs_id).type_id(),
                                resume_block_id}});
  context.SetBlockArgResultBeforeConstantUse(result_id, branch_value_id, rhs_id,
                                             short_circuit_result_id);
  context.node_stack().Push(node_id, result_id);
  return true;
}

auto HandleShortCircuitOperatorAnd(Context& context,
                                   Parse::ShortCircuitOperatorAndId node_id)
    -> bool {
  return HandleShortCircuitOperator(context, node_id);
}

auto HandleShortCircuitOperatorOr(Context& context,
                                  Parse::ShortCircuitOperatorOrId node_id)
    -> bool {
  return HandleShortCircuitOperator(context, node_id);
}

}  // namespace Carbon::Check
