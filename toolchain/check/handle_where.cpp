// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::WhereOperandId node_id) -> bool {
  // The expression at the top of the stack represents a constraint type that
  // is being modified by the `where` operator. It would be `MyInterface` in
  // `MyInterface where .Member = i32`.
  auto [self_node, self_id] = context.node_stack().PopExprWithNodeId();
  auto self_type_id = ExprAsType(context, self_node, self_id);
  context.node_stack().Push(node_id, self_type_id);

  // Introduce a name scope so that we can remove the `.Self` entry we are
  // adding to name lookup at the end of the `where` expression.
  context.scope_stack().Push();
  // Introduce `.Self` as a symbolic binding. Its type is the value of the
  // expression to the left of `where`, so `MyInterface` in the example above.
  // Because there is no equivalent non-symbolic value, we use `Invalid` as
  // the `value_id` on the `BindSymbolicName`.
  auto entity_name_id = context.entity_names().Add(
      {.name_id = SemIR::NameId::PeriodSelf,
       .parent_scope_id = context.decl_name_stack().PeekParentScopeId(),
       .bind_index = context.scope_stack().AddCompileTimeBinding()});
  auto inst_id =
      context.AddInst(SemIR::LocIdAndInst::NoLoc<SemIR::BindSymbolicName>(
          {.type_id = self_type_id,
           .entity_name_id = entity_name_id,
           .value_id = SemIR::InstId::Invalid}));
  context.scope_stack().PushCompileTimeBinding(inst_id);
  auto existing =
      context.scope_stack().LookupOrAddName(SemIR::NameId::PeriodSelf, inst_id);
  // Shouldn't have any names in newly created scope.
  CARBON_CHECK(!existing.is_valid());

  // Going to put each requirement on `args_type_info_stack`, so we can have an
  // inst block with the varying number of requirements but keeping other
  // instructions on the current inst block from the `inst_block_stack()`.
  context.args_type_info_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::RequirementEqualId node_id)
    -> bool {
  auto rhs = context.node_stack().PopExpr();
  auto lhs = context.node_stack().PopExpr();
  // TODO: convert rhs to type of lhs
  // Build up the list of arguments for the `WhereExpr` inst.
  context.args_type_info_stack().AddInstId(
      context.AddInstInNoBlock<SemIR::RequirementRewrite>(
          node_id, {.lhs_id = lhs, .rhs_id = rhs}));
  return true;
}

auto HandleParseNode(Context& context, Parse::RequirementEquaEqualId node_id)
    -> bool {
  auto rhs = context.node_stack().PopExpr();
  auto lhs = context.node_stack().PopExpr();
  // TODO: type check lhs and rhs are compatible
  // Build up the list of arguments for the `WhereExpr` inst.
  context.args_type_info_stack().AddInstId(
      context.AddInstInNoBlock<SemIR::RequirementEquivalent>(
          node_id, {.lhs_id = lhs, .rhs_id = rhs}));
  return true;
}

auto HandleParseNode(Context& context, Parse::RequirementImplsId node_id)
    -> bool {
  auto rhs = context.node_stack().PopExpr();
  auto lhs = context.node_stack().PopExpr();
  // TODO: check lhs is a facet and rhs is a facet type
  // Build up the list of arguments for the `WhereExpr` inst.
  context.args_type_info_stack().AddInstId(
      context.AddInstInNoBlock<SemIR::RequirementImpls>(
          node_id, {.lhs_id = lhs, .rhs_id = rhs}));
  return true;
}

auto HandleParseNode(Context& /*context*/, Parse::RequirementAndId /*node_id*/)
    -> bool {
  // Nothing to do.
  return true;
}

auto HandleParseNode(Context& context, Parse::WhereExprId /*node_id*/) -> bool {
  // Remove `PeriodSelf` from name lookup, undoing the `Push` done for the
  // `WhereOperand`.
  context.scope_stack().Pop();
  // FIXME: Pop WhereIntroducer node with type_id from context.node_stack()
  // auto requirements_id = context.args_type_info_stack().Pop();
  // FIXME: AddAndPush instruction for newly formed restricted constraint type.
  return true;
}

}  // namespace Carbon::Check
