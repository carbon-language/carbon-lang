// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {
auto HandleParseNode(Context& context, Parse::PositionalParamExprId node_id)
    -> bool {
  auto positional_param_id = AddInstInNoBlock<SemIR::PositionalParam>(
      context, node_id,
      {.type_id = SemIR::TypeType::TypeId,
       .int_id = context.tokens().GetDollarIntLiteral(
           context.parse_tree().node_token(node_id))});
  context.args_type_info_stack().AddInstId(positional_param_id);
  context.inst_block_stack().AddInstId(positional_param_id);
  context.node_stack().Push(node_id, positional_param_id);
  return true;
}
}  // namespace Carbon::Check
