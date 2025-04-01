// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/operator.h"
#include "toolchain/check/pointer_dereference.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/sem_ir/expr_info.h"

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

auto HandleParseNode(Context& context, Parse::InfixOperatorAmpId node_id)
    -> bool {
  // TODO: Facet type intersection may need to be handled directly.
  return HandleBinaryOperator(context, node_id, {"BitAnd"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorAmpEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"BitAndAssign"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorAsId node_id)
    -> bool {
  auto [rhs_node, rhs_id] = context.node_stack().PopExprWithNodeId();
  auto [lhs_node, lhs_id] = context.node_stack().PopExprWithNodeId();

  auto rhs_type_id = ExprAsType(context, rhs_node, rhs_id).type_id;
  context.node_stack().Push(
      node_id, ConvertForExplicitAs(context, node_id, lhs_id, rhs_type_id));
  return true;
}

auto HandleParseNode(Context& context, Parse::InfixOperatorCaretId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"BitXor"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorCaretEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"BitXorAssign"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorEqualId node_id)
    -> bool {
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
                      "expression is not assignable");
    context.emitter().Emit(lhs_node, AssignmentToNonAssignable);
  }
  // TODO: Destroy the old value before reinitializing. This will require
  // building the destruction code before we build the RHS subexpression.
  rhs_id = Initialize(context, node_id, lhs_id, rhs_id);
  AddInst<SemIR::Assign>(context, node_id,
                         {.lhs_id = lhs_id, .rhs_id = rhs_id});
  // We model assignment as an expression, so we need to push a value for
  // it, even though it doesn't produce a value.
  // TODO: Consider changing our parse tree to model assignment as a
  // different kind of statement than an expression statement.
  context.node_stack().Push(node_id, lhs_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::InfixOperatorEqualEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"Eq", {}, "Equal"});
}

auto HandleParseNode(Context& context,
                     Parse::InfixOperatorExclaimEqualId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"Eq", {}, "NotEqual"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorGreaterId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"Ordered", {}, "Greater"});
}

auto HandleParseNode(Context& context,
                     Parse::InfixOperatorGreaterEqualId node_id) -> bool {
  return HandleBinaryOperator(context, node_id,
                              {"Ordered", {}, "GreaterOrEquivalent"});
}

auto HandleParseNode(Context& context,
                     Parse::InfixOperatorGreaterGreaterId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"RightShift"});
}

auto HandleParseNode(Context& context,
                     Parse::InfixOperatorGreaterGreaterEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"RightShiftAssign"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorLessId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"Ordered", {}, "Less"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorLessEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id,
                              {"Ordered", {}, "LessOrEquivalent"});
}

auto HandleParseNode(Context& context,
                     Parse::InfixOperatorLessEqualGreaterId node_id) -> bool {
  return context.TODO(node_id, "remove <=> operator that is not in the design");
}

auto HandleParseNode(Context& context, Parse::InfixOperatorLessLessId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"LeftShift"});
}

auto HandleParseNode(Context& context,
                     Parse::InfixOperatorLessLessEqualId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"LeftShiftAssign"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorMinusId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"Sub"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorMinusEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"SubAssign"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorPercentId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"Mod"});
}

auto HandleParseNode(Context& context,
                     Parse::InfixOperatorPercentEqualId node_id) -> bool {
  return HandleBinaryOperator(context, node_id, {"ModAssign"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorPipeId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"BitOr"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorPipeEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"BitOrAssign"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorPlusId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"Add"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorPlusEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"AddAssign"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorSlashId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"Div"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorSlashEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"DivAssign"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorStarId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"Mul"});
}

auto HandleParseNode(Context& context, Parse::InfixOperatorStarEqualId node_id)
    -> bool {
  return HandleBinaryOperator(context, node_id, {"MulAssign"});
}

auto HandleParseNode(Context& context, Parse::PostfixOperatorStarId node_id)
    -> bool {
  auto value_id = context.node_stack().PopExpr();
  auto inner_type_id = ExprAsType(context, node_id, value_id).type_id;
  AddInstAndPush<SemIR::PointerType>(
      context, node_id,
      {.type_id = SemIR::TypeType::SingletonTypeId,
       .pointee_id = inner_type_id});
  return true;
}

auto HandleParseNode(Context& context, Parse::PrefixOperatorAmpId node_id)
    -> bool {
  auto value_id = context.node_stack().PopExpr();
  auto type_id = context.insts().Get(value_id).type_id();
  // Only durable reference expressions can have their address taken.
  switch (SemIR::GetExprCategory(context.sem_ir(), value_id)) {
    case SemIR::ExprCategory::DurableRef:
    case SemIR::ExprCategory::Error:
      break;
    case SemIR::ExprCategory::EphemeralRef:
      CARBON_DIAGNOSTIC(AddrOfEphemeralRef, Error,
                        "cannot take the address of a temporary object");
      context.emitter().Emit(TokenOnly(node_id), AddrOfEphemeralRef);
      value_id = SemIR::ErrorInst::SingletonInstId;
      break;
    default:
      CARBON_DIAGNOSTIC(AddrOfNonRef, Error,
                        "cannot take the address of non-reference expression");
      context.emitter().Emit(TokenOnly(node_id), AddrOfNonRef);
      value_id = SemIR::ErrorInst::SingletonInstId;
      break;
  }
  AddInstAndPush<SemIR::AddrOf>(
      context, node_id,
      SemIR::AddrOf{.type_id = GetPointerType(context, type_id),
                    .lvalue_id = value_id});
  return true;
}

auto HandleParseNode(Context& context, Parse::PrefixOperatorCaretId node_id)
    -> bool {
  return HandleUnaryOperator(context, node_id, {"BitComplement"});
}

auto HandleParseNode(Context& context, Parse::PrefixOperatorConstId node_id)
    -> bool {
  auto value_id = context.node_stack().PopExpr();

  // `const (const T)` is probably not what the developer intended.
  // TODO: Detect `const (const T)*` and suggest moving the `*` inside the
  // parentheses.
  if (context.insts().Get(value_id).kind() == SemIR::ConstType::Kind) {
    CARBON_DIAGNOSTIC(RepeatedConst, Warning,
                      "`const` applied repeatedly to the same type has no "
                      "additional effect");
    context.emitter().Emit(node_id, RepeatedConst);
  }
  auto inner_type_id = ExprAsType(context, node_id, value_id).type_id;
  AddInstAndPush<SemIR::ConstType>(
      context, node_id,
      {.type_id = SemIR::TypeType::SingletonTypeId, .inner_id = inner_type_id});
  return true;
}

auto HandleParseNode(Context& context, Parse::PrefixOperatorMinusId node_id)
    -> bool {
  return HandleUnaryOperator(context, node_id, {"Negate"});
}

auto HandleParseNode(Context& context,
                     Parse::PrefixOperatorMinusMinusId node_id) -> bool {
  return HandleUnaryOperator(context, node_id, {"Dec"});
}

auto HandleParseNode(Context& context, Parse::PrefixOperatorNotId node_id)
    -> bool {
  auto value_id = context.node_stack().PopExpr();
  value_id = ConvertToBoolValue(context, node_id, value_id);
  AddInstAndPush<SemIR::UnaryOperatorNot>(
      context, node_id,
      {.type_id = context.insts().Get(value_id).type_id(),
       .operand_id = value_id});
  return true;
}

auto HandleParseNode(Context& context, Parse::PrefixOperatorPartialId node_id)
    -> bool {
  return context.TODO(node_id, "partial operator");
}

auto HandleParseNode(Context& context, Parse::PrefixOperatorPlusPlusId node_id)
    -> bool {
  return HandleUnaryOperator(context, node_id, {"Inc"});
}

auto HandleParseNode(Context& context, Parse::PrefixOperatorStarId node_id)
    -> bool {
  auto base_id = context.node_stack().PopExpr();

  auto deref_base_id = PerformPointerDereference(
      context, node_id, base_id,
      [&context, &node_id](SemIR::TypeId not_pointer_type_id) {
        // TODO: Pass in the expression we're trying to dereference to produce a
        // better diagnostic.
        CARBON_DIAGNOSTIC(DerefOfNonPointer, Error,
                          "cannot dereference operand of non-pointer type {0}",
                          SemIR::TypeId);

        auto builder = context.emitter().Build(
            TokenOnly(node_id), DerefOfNonPointer, not_pointer_type_id);

        // TODO: Check for any facet here, rather than only a type.
        if (not_pointer_type_id == SemIR::TypeType::SingletonTypeId) {
          CARBON_DIAGNOSTIC(
              DerefOfType, Note,
              "to form a pointer type, write the `*` after the pointee type");
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
      is_or ? AddInst<SemIR::UnaryOperatorNot>(
                  context, node_id,
                  {.type_id = bool_type_id, .operand_id = cond_value_id})
            : cond_value_id;
  auto short_circuit_result_id = AddInst<SemIR::BoolLiteral>(
      context, node_id,
      {.type_id = bool_type_id, .value = SemIR::BoolValue::From(is_or)});

  // Create a block for the right-hand side and for the continuation.
  auto rhs_block_id =
      AddDominatedBlockAndBranchIf(context, node_id, branch_value_id);
  auto end_block_id = AddDominatedBlockAndBranchWithArg(
      context, node_id, short_circuit_result_id);

  // Push the branch condition and result for use when handling the complete
  // expression.
  context.node_stack().Push(cond_node, branch_value_id);
  context.node_stack().Push(cond_node, short_circuit_result_id);

  // Push the resumption and the right-hand side blocks, and start emitting the
  // right-hand operand.
  context.inst_block_stack().Pop();
  context.inst_block_stack().Push(end_block_id);
  context.inst_block_stack().Push(rhs_block_id);
  context.region_stack().AddToRegion(rhs_block_id, node_id);

  // HandleShortCircuitOperator will follow, and doesn't need the operand on the
  // node stack.
  return true;
}

auto HandleParseNode(Context& context, Parse::ShortCircuitOperandAndId node_id)
    -> bool {
  return HandleShortCircuitOperand(context, node_id, /*is_or=*/false);
}

auto HandleParseNode(Context& context, Parse::ShortCircuitOperandOrId node_id)
    -> bool {
  return HandleShortCircuitOperand(context, node_id, /*is_or=*/true);
}

// Short circuit operator handling is uniform because the branching logic
// occurs during operand handling.
static auto HandleShortCircuitOperator(Context& context, Parse::NodeId node_id)
    -> bool {
  if (context.return_scope_stack().empty()) {
    context.TODO(node_id,
                 "Control flow expressions are currently only supported inside "
                 "functions.");
  }
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
  AddInst<SemIR::BranchWithArg>(
      context, node_id, {.target_id = resume_block_id, .arg_id = rhs_id});
  context.inst_block_stack().Pop();
  context.region_stack().AddToRegion(resume_block_id, node_id);

  // Collect the result from either the first or second operand.
  auto result_id = AddInst<SemIR::BlockArg>(
      context, node_id,
      {.type_id = context.insts().Get(rhs_id).type_id(),
       .block_id = resume_block_id});
  SetBlockArgResultBeforeConstantUse(context, result_id, branch_value_id,
                                     rhs_id, short_circuit_result_id);
  context.node_stack().Push(node_id, result_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::ShortCircuitOperatorAndId node_id)
    -> bool {
  return HandleShortCircuitOperator(context, node_id);
}

auto HandleParseNode(Context& context, Parse::ShortCircuitOperatorOrId node_id)
    -> bool {
  return HandleShortCircuitOperator(context, node_id);
}

}  // namespace Carbon::Check
