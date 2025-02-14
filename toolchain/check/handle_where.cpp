// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::WhereOperandId node_id) -> bool {
  // The expression at the top of the stack represents a constraint type that
  // is being modified by the `where` operator. It would be `MyInterface` in
  // `MyInterface where .Member = i32`.
  auto [self_node, self_id] = context.node_stack().PopExprWithNodeId();
  auto self_type_id = ExprAsType(context, self_node, self_id).type_id;
  // Only facet types may have `where` restrictions.
  if (self_type_id != SemIR::ErrorInst::SingletonTypeId &&
      !context.types().IsFacetType(self_type_id)) {
    CARBON_DIAGNOSTIC(WhereOnNonFacetType, Error,
                      "left argument of `where` operator must be a facet type");
    context.emitter().Emit(self_node, WhereOnNonFacetType);
    self_type_id = SemIR::ErrorInst::SingletonTypeId;
  }

  // Introduce a name scope so that we can remove the `.Self` entry we are
  // adding to name lookup at the end of the `where` expression.
  context.scope_stack().Push();
  // Introduce `.Self` as a symbolic binding. Its type is the value of the
  // expression to the left of `where`, so `MyInterface` in the example above.
  auto entity_name_id = context.entity_names().Add(
      {.name_id = SemIR::NameId::PeriodSelf,
       .parent_scope_id = context.scope_stack().PeekNameScopeId()});
  auto inst_id = AddInst(
      context, SemIR::LocIdAndInst::NoLoc<SemIR::BindSymbolicName>(
                   {.type_id = self_type_id,
                    .entity_name_id = entity_name_id,
                    // `None` because there is no equivalent non-symbolic value.
                    .value_id = SemIR::InstId::None}));
  auto existing =
      context.scope_stack().LookupOrAddName(SemIR::NameId::PeriodSelf, inst_id);
  // Shouldn't have any names in newly created scope.
  CARBON_CHECK(!existing.has_value());

  // Save the `.Self` symbolic binding on the node stack. It will become the
  // first argument to the `WhereExpr` instruction.
  context.node_stack().Push(node_id, inst_id);

  // Going to put each requirement on `args_type_info_stack`, so we can have an
  // inst block with the varying number of requirements but keeping other
  // instructions on the current inst block from the `inst_block_stack()`.
  context.args_type_info_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::RequirementEqualId node_id)
    -> bool {
  auto [rhs_node, rhs_id] = context.node_stack().PopExprWithNodeId();
  auto lhs = context.node_stack().PopExpr();

  // Convert rhs to type of lhs.
  auto lhs_type_id = context.insts().Get(lhs).type_id();
  if (context.types().GetConstantId(lhs_type_id).is_symbolic()) {
    // If the type of the associated constant is symbolic, we defer conversion
    // until the constraint is resolved, in case it depends on `Self` (which
    // will now be a reference to `.Self`).
    // For now we convert to a value expression eagerly because otherwise we'll
    // often be unable to constant-evaluate the enclosing `where` expression.
    // TODO: Perform the conversion symbolically and add an implicit constraint
    // that this conversion is valid and produces a constant.
    rhs_id = ConvertToValueExpr(context, rhs_id);
  } else {
    rhs_id = ConvertToValueOfType(context, rhs_node, rhs_id,
                                  context.insts().Get(lhs).type_id());
  }

  // Build up the list of arguments for the `WhereExpr` inst.
  context.args_type_info_stack().AddInstId(
      AddInstInNoBlock<SemIR::RequirementRewrite>(
          context, node_id, {.lhs_id = lhs, .rhs_id = rhs_id}));
  return true;
}

auto HandleParseNode(Context& context, Parse::RequirementEqualEqualId node_id)
    -> bool {
  auto rhs = context.node_stack().PopExpr();
  auto lhs = context.node_stack().PopExpr();
  // TODO: Type check lhs and rhs are comparable.
  // TODO: Require that at least one side uses a designator.

  // Build up the list of arguments for the `WhereExpr` inst.
  context.args_type_info_stack().AddInstId(
      AddInstInNoBlock<SemIR::RequirementEquivalent>(
          context, node_id, {.lhs_id = lhs, .rhs_id = rhs}));
  return true;
}

auto HandleParseNode(Context& context, Parse::RequirementImplsId node_id)
    -> bool {
  auto [rhs_node, rhs_id] = context.node_stack().PopExprWithNodeId();
  auto [lhs_node, lhs_id] = context.node_stack().PopExprWithNodeId();

  // Check lhs is a facet and rhs is a facet type.
  auto lhs_as_type = ExprAsType(context, lhs_node, lhs_id);
  auto rhs_as_type = ExprAsType(context, rhs_node, rhs_id);
  if (rhs_as_type.type_id != SemIR::ErrorInst::SingletonTypeId &&
      !context.types().IsFacetType(rhs_as_type.type_id)) {
    CARBON_DIAGNOSTIC(
        ImplsOnNonFacetType, Error,
        "right argument of `impls` requirement must be a facet type");
    context.emitter().Emit(rhs_node, ImplsOnNonFacetType);
    rhs_as_type.inst_id = SemIR::ErrorInst::SingletonInstId;
  }
  // TODO: Require that at least one side uses a designator.
  // TODO: For things like `HashSet(.T) as type`, add an implied constraint
  // that `.T impls Hash`.

  // Build up the list of arguments for the `WhereExpr` inst.
  context.args_type_info_stack().AddInstId(
      AddInstInNoBlock<SemIR::RequirementImpls>(
          context, node_id,
          {.lhs_id = lhs_as_type.inst_id, .rhs_id = rhs_as_type.inst_id}));
  return true;
}

auto HandleParseNode(Context& /*context*/, Parse::RequirementAndId /*node_id*/)
    -> bool {
  // Nothing to do.
  return true;
}

auto HandleParseNode(Context& context, Parse::WhereExprId node_id) -> bool {
  // Remove `PeriodSelf` from name lookup, undoing the `Push` done for the
  // `WhereOperand`.
  context.scope_stack().Pop();
  SemIR::InstId period_self_id =
      context.node_stack().Pop<Parse::NodeKind::WhereOperand>();
  SemIR::InstBlockId requirements_id = context.args_type_info_stack().Pop();
  AddInstAndPush<SemIR::WhereExpr>(context, node_id,
                                   {.type_id = SemIR::TypeType::SingletonTypeId,
                                    .period_self_id = period_self_id,
                                    .requirements_id = requirements_id});
  return true;
}

}  // namespace Carbon::Check
