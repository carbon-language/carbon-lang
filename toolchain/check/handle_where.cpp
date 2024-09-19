// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::WhereOperandId /*node_id*/)
    -> bool {
  // The expression at the top of the stack represents a constraint type that
  // is being modified by the `where` operator. It would be `MyInterface` in
  // `MyInterface where .Member = i32`.
  auto [self_node, self_id] = context.node_stack().PopExprWithNodeId();
  auto self_type_id = ExprAsType(context, self_node, self_id);
  // TODO: Do this once `WhereExpr` is ready to do something with this:
  // context.node_stack().Push(node_id, self_type_id);
  context.node_stack().Push(self_node, self_id);

  // Introduce a name scope so that we can remove the `.Self` entry we are
  // adding to name lookup at the end of the `where` expression.
  // FIXME: is there a declaration that should be used as the InstId here?
  auto scope_id =
      context.name_scopes().Add(SemIR::InstId::Invalid, SemIR::NameId::Invalid,
                                context.decl_name_stack().PeekParentScopeId());
  // FIXME: specify any of the arguments instead of using defaults?
  // Where would I get scope_inst_id or specific_id?
  context.scope_stack().Push(/*scope_inst_id=*/SemIR::InstId::Invalid,
                             scope_id);
  // Introduce `.Self` as a symbolic binding. Its type is the value of the
  // expression to the left of `where`, so `MyInterface` in the example above.
  // Because there is no equivalent non-symbolic value, we use `Invalid` as
  // the `value_id` on the `BindSymbolicName`.
  auto entity_name_id = context.entity_names().Add(
      {.name_id = SemIR::NameId::PeriodSelf,
       .parent_scope_id = scope_id,
       .bind_index = context.scope_stack().AddCompileTimeBinding()});
  // FIXME: should this have a location associated with `node_id`?
  auto inst_id =
      context.AddInst(SemIR::LocIdAndInst::NoLoc<SemIR::BindSymbolicName>(
          {.type_id = self_type_id,
           .entity_name_id = entity_name_id,
           .value_id = SemIR::InstId::Invalid}));
  context.scope_stack().PushCompileTimeBinding(inst_id);
  context.name_scopes().AddRequiredName(scope_id, SemIR::NameId::PeriodSelf,
                                        inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::RequirementEqualId /*node_id*/)
    -> bool {
  // TODO: Implement
  context.node_stack().PopExpr();
  context.node_stack().PopExpr();
  return true;
}

auto HandleParseNode(Context& context,
                     Parse::RequirementEqualEqualId /*node_id*/) -> bool {
  // TODO: Implement
  context.node_stack().PopExpr();
  context.node_stack().PopExpr();
  return true;
}

auto HandleParseNode(Context& context, Parse::RequirementImplsId /*node_id*/)
    -> bool {
  // TODO: Implement
  context.node_stack().PopExpr();
  context.node_stack().PopExpr();
  return true;
}

auto HandleParseNode(Context& /*context*/, Parse::RequirementAndId /*node_id*/)
    -> bool {
  // Nothing to do
  return true;
}

auto HandleParseNode(Context& context, Parse::WhereExprId /*node_id*/) -> bool {
  // Remove `PeriodSelf` from name lookup, undoing the `Push` done for the
  // `WhereOperand`.
  context.scope_stack().Pop();
  // TODO: Output instruction for newly formed restricted constraint type.
  return true;
}

}  // namespace Carbon::Check
