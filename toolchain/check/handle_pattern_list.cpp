// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/subpattern.h"
#include "toolchain/check/type.h"

namespace Carbon::Check {

// Handle the start of any kind of pattern list.
static auto HandlePatternListStart(Context& context, Parse::NodeId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  context.param_and_arg_refs_stack().Push();
  BeginSubpattern(context);
  return true;
}

auto HandleParseNode(Context& context, Parse::ImplicitParamListStartId node_id)
    -> bool {
  return HandlePatternListStart(context, node_id);
}

auto HandleParseNode(Context& context, Parse::TuplePatternStartId node_id)
    -> bool {
  return HandlePatternListStart(context, node_id);
}

auto HandleParseNode(Context& context, Parse::ExplicitParamListStartId node_id)
    -> bool {
  context.full_pattern_stack().EndImplicitParamList();
  return HandlePatternListStart(context, node_id);
}

// Handle the end of any kind of parameter list (tuple patterns have separate
// logic).
static auto HandleParamListEnd(Context& context, Parse::NodeId node_id,
                               Parse::NodeKind start_kind) -> bool {
  if (context.node_stack().PeekIs(start_kind)) {
    // End the subpattern started by a trailing comma, or the opening delimiter
    // of an empty list.
    EndSubpatternAsNonExpr(context);
  }
  // Note the Start node remains on the stack, where the param list handler can
  // make use of it.
  auto refs_id = context.param_and_arg_refs_stack().EndAndPop(start_kind);
  context.node_stack().Push(node_id, refs_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::ImplicitParamListId node_id)
    -> bool {
  return HandleParamListEnd(context, node_id,
                            Parse::NodeKind::ImplicitParamListStart);
}

auto HandleParseNode(Context& context, Parse::ExplicitParamListId node_id)
    -> bool {
  return HandleParamListEnd(context, node_id,
                            Parse::NodeKind::ExplicitParamListStart);
}

auto HandleParseNode(Context& context, Parse::TuplePatternId node_id) -> bool {
  if (context.node_stack().PeekIs(Parse::NodeKind::TuplePatternStart)) {
    // End the subpattern started by a trailing comma, or the opening delimiter
    // of an empty list.
    EndSubpatternAsNonExpr(context);
  }
  auto refs_id = context.param_and_arg_refs_stack().EndAndPop(
      Parse::NodeKind::TuplePatternStart);
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::TuplePatternStart>();

  const auto& inst_block = context.inst_blocks().Get(refs_id);
  llvm::SmallVector<SemIR::TypeId> type_ids;
  type_ids.reserve(inst_block.size());
  for (auto inst : inst_block) {
    type_ids.push_back(context.insts().Get(inst).type_id());
  }
  auto type_id = GetTupleType(context, type_ids);
  context.node_stack().Push(
      node_id,
      AddPatternInst<SemIR::TuplePattern>(
          context, node_id, {.type_id = type_id, .elements_id = refs_id}));
  EndSubpatternAsNonExpr(context);
  return true;
}

auto HandleParseNode(Context& context, Parse::PatternListCommaId /*node_id*/)
    -> bool {
  context.param_and_arg_refs_stack().ApplyComma();
  BeginSubpattern(context);
  return true;
}

}  // namespace Carbon::Check
