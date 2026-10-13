// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/literal.h"
#include "toolchain/check/type.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Check {

auto HandleParseNode(Context& /*context*/,
                     Parse::ArrayExprOpenParenId /*node_id*/) -> bool {
  return true;
}

auto HandleParseNode(Context& /*context*/,
                     Parse::ArrayExprKeywordId /*node_id*/) -> bool {
  return true;
}

auto HandleParseNode(Context& /*context*/, Parse::ArrayExprCommaId /*node_id*/)
    -> bool {
  return true;
}

auto HandleParseNode(Context& context, Parse::ArrayExprId node_id) -> bool {
  auto bound_inst_id = context.node_stack().PopExpr();
  auto [element_type_node_id, element_type_inst_id] =
      context.node_stack().PopExprWithNodeId();

  auto element_type =
      ExprAsType(context, element_type_node_id, element_type_inst_id);

  // The array bound must be a constant. Diagnose this prior to conversion
  // because conversion to `IntLiteral` will produce a generic "non-constant
  // call to compile-time-only function" error.
  //
  // TODO: Should we support runtime-phase bounds in cases such as:
  //   comptime fn F(n: i32) -> type { return array(i32; n); }
  if (!context.constant_values().Get(bound_inst_id).is_constant()) {
    CARBON_DIAGNOSTIC(InvalidArrayExpr, Error, "array bound is not a constant");
    context.emitter().Emit(bound_inst_id, InvalidArrayExpr);
    context.node_stack().Push(node_id, SemIR::ErrorInst::InstId);
    return true;
  }

  bound_inst_id = ConvertToValueOfType(
      context, SemIR::LocId(bound_inst_id), bound_inst_id,
      GetSingletonType(context, SemIR::IntLiteralType::TypeInstId));

  if (element_type.type_id == SemIR::ErrorInst::TypeId) {
    context.node_stack().Push(node_id, SemIR::ErrorInst::InstId);
    return true;
  }

  // Diagnose an invalid concrete bound here, rather than when completing
  // `Core.Array` below, so that the diagnostic points at the bound.
  //
  // As with `Core.Int`, an invalid symbolic bound is diagnosed only when the
  // type is completed.
  // TODO: Express the constraint on the bound in the prelude.
  if (!ValidateArrayType(
          context, SemIR::LocId(bound_inst_id),
          {.type_id = SemIR::TypeType::TypeId,
           .bound_id =
               context.constant_values().GetConstantInstId(bound_inst_id),
           .element_type_inst_id = element_type.inst_id})) {
    context.node_stack().Push(node_id, SemIR::ErrorInst::InstId);
    return true;
  }

  // `array(T, N)` is `Core.Array(T, N)`. The call is attributed to the
  // `array(T, N)` expression, so that the resulting type has a location.
  auto type_expr =
      MakeArrayType(context, node_id, element_type.inst_id, bound_inst_id);
  context.node_stack().Push(node_id, type_expr.inst_id);
  return true;
}

}  // namespace Carbon::Check
