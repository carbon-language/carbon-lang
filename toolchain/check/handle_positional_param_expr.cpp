// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/function.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {
auto HandleParseNode(Context& context, Parse::PositionalParamExprId node_id)
    -> bool {
  if (!context.scope_stack().IsInFunctionScope()) {
    CARBON_DIAGNOSTIC(PositionalParamInWrongScope, Error,
                      "positional parameters can only be used in a function");
    context.emitter().Emit(node_id, PositionalParamInWrongScope);
  }

  auto int_id = context.tokens().GetDollarIntLiteral(
      context.parse_tree().node_token(node_id));
  auto positional_param_id = AddInstInNoBlock<SemIR::PositionalParam>(
      context, node_id, {.type_id = SemIR::AutoType::TypeId, .int_id = int_id});
  if (!context.args_type_info_stack().empty()) {
    size_t position =
        int_id.is_embedded_value()
            ? static_cast<size_t>(int_id.AsValue())
            : static_cast<size_t>(context.ints().Get(int_id).getZExtValue());
    auto exists = false;
    for (auto param_id :
         context.args_type_info_stack().PeekCurrentBlockContents()) {
      if (position == GetPositionalParamNumber(context, param_id)) {
        exists = true;
        break;
      }
    }
    if (!exists) {
      context.args_type_info_stack().AddInstId(positional_param_id);
    }
  }
  if (!context.inst_block_stack().empty()) {
    context.inst_block_stack().AddInstId(positional_param_id);
  }
  context.node_stack().Push(node_id, positional_param_id);
  return true;
}
}  // namespace Carbon::Check
