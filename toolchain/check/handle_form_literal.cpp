// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/parse/node_category.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::RefPrimitiveFormId node_id)
    -> bool {
  auto [type_node_id, type_inst_id] = context.node_stack().PopExprWithNodeId();
  auto type_expr = ExprAsType(context, type_node_id, type_inst_id);
  auto inst_id =
      AddInst<SemIR::RefForm>(context, node_id,
                              {.type_id = SemIR::FormType::TypeId,
                               .type_component_inst_id = type_expr.inst_id});
  context.node_stack().Push(node_id, inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::ValPrimitiveFormId node_id)
    -> bool {
  auto [type_node_id, type_inst_id] = context.node_stack().PopExprWithNodeId();
  auto type_expr = ExprAsType(context, type_node_id, type_inst_id);
  auto inst_id =
      AddInst<SemIR::ValueForm>(context, node_id,
                                {.type_id = SemIR::FormType::TypeId,
                                 .type_component_inst_id = type_expr.inst_id});
  context.node_stack().Push(node_id, inst_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::VarPrimitiveFormId node_id)
    -> bool {
  auto [type_node_id, type_inst_id] = context.node_stack().PopExprWithNodeId();
  auto type_expr = ExprAsType(context, type_node_id, type_inst_id);
  auto inst_id =
      AddInst<SemIR::InitForm>(context, node_id,
                               {.type_id = SemIR::FormType::TypeId,
                                .type_component_inst_id = type_expr.inst_id,
                                .index = SemIR::CallParamIndex::None});
  context.node_stack().Push(node_id, inst_id);
  return true;
}

auto HandleParseNode(Context& /*context*/,
                     Parse::FormLiteralKeywordId /*node_id*/) -> bool {
  return true;
}

auto HandleParseNode(Context& /*context*/,
                     Parse::FormLiteralOpenParenId /*node_id*/) -> bool {
  return true;
}

auto HandleParseNode(Context& /*context*/, Parse::FormLiteralId /*node_id*/)
    -> bool {
  return true;
}

}  // namespace Carbon::Check
