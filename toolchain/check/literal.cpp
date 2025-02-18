// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/literal.h"

#include "toolchain/check/call.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type.h"

namespace Carbon::Check {

auto MakeIntLiteral(Context& context, Parse::NodeId node_id, IntId int_id)
    -> SemIR::InstId {
  return AddInst<SemIR::IntValue>(
      context, node_id,
      {.type_id =
           GetSingletonType(context, SemIR::IntLiteralType::SingletonInstId),
       .int_id = int_id});
}

auto MakeIntTypeLiteral(Context& context, Parse::NodeId node_id,
                        SemIR::IntKind int_kind, IntId size_id)
    -> SemIR::InstId {
  auto width_id = MakeIntLiteral(context, node_id, size_id);
  auto fn_inst_id = LookupNameInCore(
      context, node_id, int_kind == SemIR::IntKind::Signed ? "Int" : "UInt");
  return PerformCall(context, node_id, fn_inst_id, {width_id});
}

auto MakeIntType(Context& context, Parse::NodeId node_id,
                 SemIR::IntKind int_kind, IntId size_id) -> SemIR::TypeId {
  auto type_inst_id = MakeIntTypeLiteral(context, node_id, int_kind, size_id);
  return ExprAsType(context, node_id, type_inst_id).type_id;
}

}  // namespace Carbon::Check
