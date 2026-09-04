// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/class.h"
#include "toolchain/check/context.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/emitter.h"

namespace Carbon::Check {

// Handle the start of any kind of pattern list.
static auto HandlePatternListStart(Context& context, Parse::NodeId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  context.param_and_arg_refs_stack().Push();
  BeginExprRegionForPattern(context);
  return true;
}

auto HandleParseNode(Context& context, Parse::ImplicitParamListStartId node_id)
    -> bool {
  context.full_pattern_stack().StartImplicitParamList();
  return HandlePatternListStart(context, node_id);
}

auto HandleParseNode(Context& context, Parse::TuplePatternStartId node_id)
    -> bool {
  // End the pending `ExprRegion`, so that we can start a new one in
  // `HandlePatternListStart`.
  EndEmptyExprRegionForPattern(context);
  return HandlePatternListStart(context, node_id);
}

auto HandleParseNode(Context& context, Parse::StructPatternStartId node_id)
    -> bool {
  return context.TODO(node_id, "struct pattern start");
}

auto HandleParseNode(Context& context, Parse::ExplicitParamListStartId node_id)
    -> bool {
  context.full_pattern_stack().StartExplicitParamList();
  return HandlePatternListStart(context, node_id);
}

// Handle the end of any kind of parameter list (tuple patterns have separate
// logic).
static auto HandleParamListEnd(Context& context, Parse::NodeId node_id,
                               Parse::NodeKind start_kind) -> bool {
  if (context.node_stack().PeekIs(start_kind)) {
    // End the pending region started by a trailing comma, or the opening
    // delimiter of an empty list.
    EndEmptyExprRegionForPattern(context);
  } else {
    // End the pending region for the last pattern in the list.
    EndExprRegionForPattern(context, context.node_stack());
  }
  // Note the Start node remains on the stack, where the param list handler can
  // make use of it.
  auto refs_id = context.param_and_arg_refs_stack().EndAndPop(start_kind);
  context.node_stack().Push(node_id, refs_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::ImplicitParamListId node_id)
    -> bool {
  context.full_pattern_stack().EndImplicitParamList();
  return HandleParamListEnd(context, node_id,
                            Parse::NodeKind::ImplicitParamListStart);
}

auto HandleParseNode(Context& context, Parse::ExplicitParamListId node_id)
    -> bool {
  context.full_pattern_stack().EndExplicitParamList();
  return HandleParamListEnd(context, node_id,
                            Parse::NodeKind::ExplicitParamListStart);
}

auto HandleParseNode(Context& context, Parse::ParenPatternId node_id) -> bool {
  EndExprRegionForPattern(context, context.node_stack());
  auto pattern_id = context.node_stack().PopPattern();
  context.param_and_arg_refs_stack().PopAndDiscard();
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::TuplePatternStart>();
  context.node_stack().Push(node_id, pattern_id);
  // Start a new pending `ExprRegion`, to maintain the invariant that one is
  // pending at the end of handling for a pattern.
  BeginExprRegionForPattern(context);
  return true;
}

auto HandleParseNode(Context& context, Parse::TuplePatternId node_id) -> bool {
  if (context.node_stack().PeekIs(Parse::NodeKind::TuplePatternStart)) {
    // End the pending region started by a trailing comma, or the opening
    // delimiter of an empty list.
    EndEmptyExprRegionForPattern(context);
  } else {
    // End the pending region for the last pattern in the list.
    EndExprRegionForPattern(context, context.node_stack());
  }
  auto refs_id = context.param_and_arg_refs_stack().EndAndPop(
      Parse::NodeKind::TuplePatternStart);
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::TuplePatternStart>();

  const auto& inst_block = context.inst_blocks().Get(refs_id);
  llvm::SmallVector<SemIR::InstId> type_inst_ids;
  type_inst_ids.reserve(inst_block.size());
  for (auto inst : inst_block) {
    if (InNonStaticFieldDecl(context)) {
      CARBON_DIAGNOSTIC(FieldWithTuplePattern, Error,
                        "found tuple pattern in class `var` decl");
      context.emitter().Emit(LocIdForDiagnostics::TokenOnly(node_id),
                             FieldWithTuplePattern);

      return false;
    }

    auto type_id = ExtractScrutineeType(context.sem_ir(),
                                        context.insts().Get(inst).type_id());
    type_inst_ids.push_back(context.types().GetTypeInstId(type_id));
  }
  auto type_id = GetPatternType(context, GetTupleType(context, type_inst_ids));
  context.node_stack().Push(
      node_id,
      AddInst<SemIR::TuplePattern>(
          context, node_id, {.type_id = type_id, .elements_id = refs_id}));
  // Start a new pending `ExprRegion`, to maintain the invariant that one is
  // pending at the end of handling for a pattern.
  BeginExprRegionForPattern(context);
  return true;
}

auto HandleParseNode(Context& context, Parse::StructPatternId node_id) -> bool {
  return context.TODO(node_id, "struct pattern");
}

auto HandleParseNode(Context& context,
                     Parse::StructPatternDesignatedFieldId node_id) -> bool {
  return context.TODO(node_id, "struct pattern field");
}

auto HandleParseNode(Context& context, Parse::PatternListCommaId /*node_id*/)
    -> bool {
  EndExprRegionForPattern(context, context.node_stack());
  context.param_and_arg_refs_stack().ApplyComma();
  BeginExprRegionForPattern(context);
  return true;
}

auto HandleParseNode(Context& context, Parse::DefaultValueUnspecifiedId node_id)
    -> bool {
  return context.TODO(node_id, "pattern default values");
}

auto HandleParseNode(Context& context,
                     Parse::DefaultValueExprStartId /*node_id*/) -> bool {
  // We want to check the default value expression as a normal expression,
  // and not convert it into a pattern.
  EndEmptyExprRegionForPattern(context);
  return true;
}

auto HandleParseNode(Context& context, Parse::DefaultValuePatternId node_id)
    -> bool {
  // On entry, the top of the node stack should have an expression for the
  // default value. We evaluate it to get a constant.
  auto [expr_node_id, expr_inst_id] = context.node_stack().PopExprWithNodeId();

  // Ensure we are in an explicit parameter list, otherwise issue a diagnostic.
  auto full_pattern_kind = context.full_pattern_stack().CurrentKind();
  if (full_pattern_kind != FullPatternStack::Kind::ExplicitParamList) {
    CARBON_DIAGNOSTIC(PatternDefaultValueNotInParameterList, Error,
                      "default values are only supported in parameter lists");
    context.emitter().Emit(LocIdForDiagnostics(expr_node_id),
                           PatternDefaultValueNotInParameterList);
    return false;
  }

  auto expr_const_id = TryEvalInst(context, expr_inst_id);
  if (expr_const_id == SemIR::ConstantId::NotConstant) {
    CARBON_DIAGNOSTIC(PatternDefaultValueNotConstant, Error,
                      "default value for pattern must be constant");
    context.emitter().Emit(LocIdForDiagnostics(expr_node_id),
                           PatternDefaultValueNotConstant);
    return false;
  }

  // Look up the instruction associated with the evaluated constant.
  auto constant_inst_id = context.constant_values().GetInstId(expr_const_id);
  CARBON_CHECK(constant_inst_id != SemIR::InstId::None);

  // Add the value to the default values array in the full pattern stack, for
  // recovery later in the NameComponent.
  auto default_value_id =
      context.full_pattern_stack().AddDefaultValue(constant_inst_id);

  // Next on the node stack should be the pattern for which this default was
  // specified. We pop that so we can issue the DefaultValuePattern in its
  // place.
  auto pattern_inst_id = context.node_stack().PopPattern();

  // The default value pattern should have the same type as the subpattern.
  auto pattern_type_id = context.insts().Get(pattern_inst_id).type_id();
  auto default_value_inst_id = AddInst<SemIR::DefaultValuePattern>(
      context, node_id,
      {.type_id = pattern_type_id,
       .subpattern_id = pattern_inst_id,
       .default_value_id = default_value_id});
  context.node_stack().Push(node_id, default_value_inst_id);

  // We turned off expr region for pattern checking while parsing the default
  // value expression, so turn it back on again for further pattern checking.
  BeginExprRegionForPattern(context);
  return true;
}

}  // namespace Carbon::Check
