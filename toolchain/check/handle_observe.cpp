// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/hashtable_key_context.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::ObserveIntroducerId node_id)
    -> bool {
  auto scope_id = context.scope_stack().PeekNameScopeId();
  if (!context.scope_stack().IsInFunctionScope() &&
      (scope_id == SemIR::NameScopeId::None ||
       !context.insts().Is<SemIR::InterfaceDecl>(
           context.name_scopes().Get(scope_id).inst_id()))) {
    CARBON_DIAGNOSTIC(
        ObserveInWrongScope, Error,
        "`observe` can only be used in an `interface` or `function`");
    context.emitter().Emit(node_id, ObserveInWrongScope);
  }

  context.inst_block_stack().Push();

  context.node_stack().Push(node_id, SemIR::ErrorInst::InstId);
  return true;
}

auto HandleParseNode(Context& context, Parse::ObserveEqualEqualId node_id)
    -> bool {
  auto [rhs_node_id, rhs_inst_id] = context.node_stack().PopExprWithNodeId();
  auto [lhs_node_id, lhs_inst_id] = context.node_stack().PopExprWithNodeId();

  auto rhs_as_type = ExprAsType(context, rhs_node_id, rhs_inst_id);
  auto lhs_as_type = ExprAsType(context, lhs_node_id, lhs_inst_id);

  // TODO: Type check lhs and rhs are same type.

  // Push rhs again for chain == expressions.
  context.node_stack().Push(rhs_node_id, rhs_inst_id);
  context.inst_block_stack().AddInstId(
      AddInstInNoBlock<SemIR::ObserveEquivalent>(
          context, node_id,
          {.lhs_id = lhs_as_type.inst_id, .rhs_id = rhs_as_type.inst_id}));
  return true;
}

auto HandleParseNode(Context& context, Parse::ObserveImplsId node_id) -> bool {
  auto [rhs_node_id, rhs_inst_id] = context.node_stack().PopExprWithNodeId();
  auto [lhs_node_id, lhs_inst_id] = context.node_stack().PopExprWithNodeId();

  auto rhs_as_type = ExprAsType(context, rhs_node_id, rhs_inst_id);
  auto lhs_as_type = ExprAsType(context, lhs_node_id, lhs_inst_id);

  if (!context.types().IsFacetTypeOrError(rhs_as_type.type_id)) {
    DiagnoseImplsOnNonFacetType(context, rhs_node_id);
    rhs_as_type.inst_id = SemIR::ErrorInst::TypeInstId;
  }

  // Dummy node for ObserveDeclId.
  context.node_stack().Push(rhs_node_id, rhs_inst_id);
  context.inst_block_stack().AddInstId(AddInstInNoBlock<SemIR::ObserveImpls>(
      context, node_id,
      {.lhs_id = lhs_as_type.inst_id, .rhs_id = rhs_as_type.inst_id}));
  return true;
}

auto HandleParseNode(Context& context, Parse::ObserveDeclId node_id) -> bool {
  context.node_stack().PopAndIgnore();
  context.node_stack().Pop<Parse::NodeKind::ObserveIntroducer>();
  auto operations_id = context.inst_block_stack().Pop();
  AddInst<SemIR::ObserveDecl>(context, node_id,
                              {.operations_id = operations_id});
  return true;
}

}  // namespace Carbon::Check
