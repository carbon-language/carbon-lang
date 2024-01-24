// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"

namespace Carbon::Check {

auto HandleInfixOperatorAmp(Context& context,
                            Parse::InfixOperatorAmpId parse_node) -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorAmp");
}

auto HandleInfixOperatorAmpEqual(Context& context,
                                 Parse::InfixOperatorAmpEqualId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorAmpEqual");
}

auto HandleInfixOperatorAs(Context& context,
                           Parse::InfixOperatorAsId parse_node) -> bool {
  auto [rhs_node, rhs_id] = context.node_stack().PopExprWithParseNode();
  auto [lhs_node, lhs_id] = context.node_stack().PopExprWithParseNode();

  auto rhs_type_id = ExprAsType(context, rhs_node, rhs_id);
  context.node_stack().Push(
      parse_node,
      ConvertForExplicitAs(context, parse_node, lhs_id, rhs_type_id));
  return true;
}

auto HandleInfixOperatorCaret(Context& context,
                              Parse::InfixOperatorCaretId parse_node) -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorCaret");
}

auto HandleInfixOperatorCaretEqual(Context& context,
                                   Parse::InfixOperatorCaretEqualId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorCaretEqual");
}

auto HandleInfixOperatorEqual(Context& context,
                              Parse::InfixOperatorEqualId parse_node) -> bool {
  auto [rhs_node, rhs_id] = context.node_stack().PopExprWithParseNode();
  auto [lhs_node, lhs_id] = context.node_stack().PopExprWithParseNode();

  // TODO: handle complex assignment expression such as `a += 1`.
  if (auto lhs_cat = SemIR::GetExprCategory(context.sem_ir(), lhs_id);
      lhs_cat != SemIR::ExprCategory::DurableRef &&
      lhs_cat != SemIR::ExprCategory::Error) {
    CARBON_DIAGNOSTIC(AssignmentToNonAssignable, Error,
                      "Expression is not assignable.");
    context.emitter().Emit(lhs_node, AssignmentToNonAssignable);
  }
  // TODO: Destroy the old value before reinitializing. This will require
  // building the destruction code before we build the RHS subexpression.
  rhs_id = Initialize(context, parse_node, lhs_id, rhs_id);
  context.AddInst({parse_node, SemIR::Assign{lhs_id, rhs_id}});
  // We model assignment as an expression, so we need to push a value for
  // it, even though it doesn't produce a value.
  // TODO: Consider changing our parse tree to model assignment as a
  // different kind of statement than an expression statement.
  context.node_stack().Push(parse_node, lhs_id);
  return true;
}

auto HandleInfixOperatorEqualEqual(Context& context,
                                   Parse::InfixOperatorEqualEqualId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorEqualEqual");
}

auto HandleInfixOperatorExclaimEqual(
    Context& context, Parse::InfixOperatorExclaimEqualId parse_node) -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorExclaimEqual");
}

auto HandleInfixOperatorGreater(Context& context,
                                Parse::InfixOperatorGreaterId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorGreater");
}

auto HandleInfixOperatorGreaterEqual(
    Context& context, Parse::InfixOperatorGreaterEqualId parse_node) -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorGreaterEqual");
}

auto HandleInfixOperatorGreaterGreater(
    Context& context, Parse::InfixOperatorGreaterGreaterId parse_node) -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorGreaterGreater");
}

auto HandleInfixOperatorGreaterGreaterEqual(
    Context& context, Parse::InfixOperatorGreaterGreaterEqualId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorGreaterGreaterEqual");
}

auto HandleInfixOperatorLess(Context& context,
                             Parse::InfixOperatorLessId parse_node) -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorLess");
}

auto HandleInfixOperatorLessEqual(Context& context,
                                  Parse::InfixOperatorLessEqualId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorLessEqual");
}

auto HandleInfixOperatorLessEqualGreater(
    Context& context, Parse::InfixOperatorLessEqualGreaterId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorLessEqualGreater");
}

auto HandleInfixOperatorLessLess(Context& context,
                                 Parse::InfixOperatorLessLessId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorLessLess");
}

auto HandleInfixOperatorLessLessEqual(
    Context& context, Parse::InfixOperatorLessLessEqualId parse_node) -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorLessLessEqual");
}

auto HandleInfixOperatorMinus(Context& context,
                              Parse::InfixOperatorMinusId parse_node) -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorMinus");
}

auto HandleInfixOperatorMinusEqual(Context& context,
                                   Parse::InfixOperatorMinusEqualId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorMinusEqual");
}

auto HandleInfixOperatorPercent(Context& context,
                                Parse::InfixOperatorPercentId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorPercent");
}

auto HandleInfixOperatorPercentEqual(
    Context& context, Parse::InfixOperatorPercentEqualId parse_node) -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorPercentEqual");
}

auto HandleInfixOperatorPipe(Context& context,
                             Parse::InfixOperatorPipeId parse_node) -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorPipe");
}

auto HandleInfixOperatorPipeEqual(Context& context,
                                  Parse::InfixOperatorPipeEqualId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorPipeEqual");
}

auto HandleInfixOperatorPlus(Context& context,
                             Parse::InfixOperatorPlusId parse_node) -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorPlus");
}

auto HandleInfixOperatorPlusEqual(Context& context,
                                  Parse::InfixOperatorPlusEqualId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorPlusEqual");
}

auto HandleInfixOperatorSlash(Context& context,
                              Parse::InfixOperatorSlashId parse_node) -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorSlash");
}

auto HandleInfixOperatorSlashEqual(Context& context,
                                   Parse::InfixOperatorSlashEqualId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorSlashEqual");
}

auto HandleInfixOperatorStar(Context& context,
                             Parse::InfixOperatorStarId parse_node) -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorStar");
}

auto HandleInfixOperatorStarEqual(Context& context,
                                  Parse::InfixOperatorStarEqualId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandleInfixOperatorStarEqual");
}

auto HandlePostfixOperatorStar(Context& context,
                               Parse::PostfixOperatorStarId parse_node)
    -> bool {
  auto value_id = context.node_stack().PopExpr();
  auto inner_type_id = ExprAsType(context, parse_node, value_id);
  context.AddInstAndPush(
      {parse_node, SemIR::PointerType{SemIR::TypeId::TypeType, inner_type_id}});
  return true;
}

auto HandlePrefixOperatorAmp(Context& context,
                             Parse::PrefixOperatorAmpId parse_node) -> bool {
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
      context.emitter().Emit(TokenOnly(parse_node), AddrOfEphemeralRef);
      value_id = SemIR::InstId::BuiltinError;
      break;
    default:
      CARBON_DIAGNOSTIC(AddrOfNonRef, Error,
                        "Cannot take the address of non-reference expression.");
      context.emitter().Emit(TokenOnly(parse_node), AddrOfNonRef);
      value_id = SemIR::InstId::BuiltinError;
      break;
  }
  context.AddInstAndPush(
      {parse_node, SemIR::AddrOf{context.GetPointerType(type_id), value_id}});
  return true;
}

auto HandlePrefixOperatorCaret(Context& context,
                               Parse::PrefixOperatorCaretId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandlePrefixOperatorCaret");
}

auto HandlePrefixOperatorConst(Context& context,
                               Parse::PrefixOperatorConstId parse_node)
    -> bool {
  auto value_id = context.node_stack().PopExpr();

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
      {parse_node, SemIR::ConstType{SemIR::TypeId::TypeType, inner_type_id}});
  return true;
}

auto HandlePrefixOperatorMinus(Context& context,
                               Parse::PrefixOperatorMinusId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandlePrefixOperatorMinus");
}

auto HandlePrefixOperatorMinusMinus(
    Context& context, Parse::PrefixOperatorMinusMinusId parse_node) -> bool {
  return context.TODO(parse_node, "HandlePrefixOperatorMinusMinus");
}

auto HandlePrefixOperatorNot(Context& context,
                             Parse::PrefixOperatorNotId parse_node) -> bool {
  auto value_id = context.node_stack().PopExpr();
  value_id = ConvertToBoolValue(context, parse_node, value_id);
  context.AddInstAndPush(
      {parse_node, SemIR::UnaryOperatorNot{
                       context.insts().Get(value_id).type_id(), value_id}});
  return true;
}

auto HandlePrefixOperatorPlusPlus(Context& context,
                                  Parse::PrefixOperatorPlusPlusId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandlePrefixOperatorPlusPlus");
}

auto HandlePrefixOperatorStar(Context& context,
                              Parse::PrefixOperatorStarId parse_node) -> bool {
  auto value_id = context.node_stack().PopExpr();
  value_id = ConvertToValueExpr(context, value_id);
  auto type_id =
      context.GetUnqualifiedType(context.insts().Get(value_id).type_id());
  auto result_type_id = SemIR::TypeId::Error;
  if (auto pointer_type =
          context.types().TryGetAs<SemIR::PointerType>(type_id)) {
    result_type_id = pointer_type->pointee_id;
  } else if (type_id != SemIR::TypeId::Error) {
    CARBON_DIAGNOSTIC(DerefOfNonPointer, Error,
                      "Cannot dereference operand of non-pointer type `{0}`.",
                      std::string);
    auto builder =
        context.emitter().Build(TokenOnly(parse_node), DerefOfNonPointer,
                                context.sem_ir().StringifyType(type_id));
    // TODO: Check for any facet here, rather than only a type.
    if (type_id == SemIR::TypeId::TypeType) {
      CARBON_DIAGNOSTIC(
          DerefOfType, Note,
          "To form a pointer type, write the `*` after the pointee type.");
      builder.Note(TokenOnly(parse_node), DerefOfType);
    }
    builder.Emit();
  }
  context.AddInstAndPush({parse_node, SemIR::Deref{result_type_id, value_id}});
  return true;
}

// Adds the branch for a short circuit operand.
static auto HandleShortCircuitOperand(Context& context,
                                      Parse::NodeId parse_node, bool is_or)
    -> bool {
  // Convert the condition to `bool`.
  auto cond_value_id = context.node_stack().PopExpr();
  cond_value_id = ConvertToBoolValue(context, parse_node, cond_value_id);
  auto bool_type_id = context.insts().Get(cond_value_id).type_id();

  // Compute the branch value: the condition for `and`, inverted for `or`.
  SemIR::InstId branch_value_id =
      is_or ? context.AddInst(
                  {parse_node,
                   SemIR::UnaryOperatorNot{bool_type_id, cond_value_id}})
            : cond_value_id;
  auto short_circuit_result_id = context.AddInst(
      {parse_node,
       SemIR::BoolLiteral{bool_type_id, is_or ? SemIR::BoolValue::True
                                              : SemIR::BoolValue::False}});

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
  context.AddCurrentCodeBlockToFunction(parse_node);

  // HandleShortCircuitOperator will follow, and doesn't need the operand on the
  // node stack.
  return true;
}

auto HandleShortCircuitOperandAnd(Context& context,
                                  Parse::ShortCircuitOperandAndId parse_node)
    -> bool {
  return HandleShortCircuitOperand(context, parse_node, /*is_or=*/false);
}

auto HandleShortCircuitOperandOr(Context& context,
                                 Parse::ShortCircuitOperandOrId parse_node)
    -> bool {
  return HandleShortCircuitOperand(context, parse_node, /*is_or=*/true);
}

// Short circuit operator handling is uniform because the branching logic
// occurs during operand handling.
static auto HandleShortCircuitOperator(Context& context,
                                       Parse::NodeId parse_node) -> bool {
  auto [rhs_node, rhs_id] = context.node_stack().PopExprWithParseNode();

  // The first operand is wrapped in a ShortCircuitOperand, which we
  // already handled by creating a RHS block and a resumption block, which
  // are the current block and its enclosing block.
  rhs_id = ConvertToBoolValue(context, parse_node, rhs_id);

  // When the second operand is evaluated, the result of `and` and `or` is
  // its value.
  auto resume_block_id = context.inst_block_stack().PeekOrAdd(/*depth=*/1);
  context.AddInst({parse_node, SemIR::BranchWithArg{resume_block_id, rhs_id}});
  context.inst_block_stack().Pop();
  context.AddCurrentCodeBlockToFunction(parse_node);

  // Collect the result from either the first or second operand.
  context.AddInstAndPush(
      {parse_node, SemIR::BlockArg{context.insts().Get(rhs_id).type_id(),
                                   resume_block_id}});
  return true;
}

auto HandleShortCircuitOperatorAnd(Context& context,
                                   Parse::ShortCircuitOperatorAndId parse_node)
    -> bool {
  return HandleShortCircuitOperator(context, parse_node);
}

auto HandleShortCircuitOperatorOr(Context& context,
                                  Parse::ShortCircuitOperatorOrId parse_node)
    -> bool {
  return HandleShortCircuitOperator(context, parse_node);
}

}  // namespace Carbon::Check
