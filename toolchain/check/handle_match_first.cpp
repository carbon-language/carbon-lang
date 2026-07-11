// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/diagnostics/diagnostic.h"

namespace Carbon::Check {

auto HandleParseNode(Context& /*context*/,
                     Parse::MatchFirstIntroducerId /*node_id*/) -> bool {
  return true;
}

auto HandleParseNode(Context& context,
                     Parse::MatchFirstDefinitionStartId node_id) -> bool {
  auto enclosing_scope_inst_id = context.scope_stack().PeekInstId();
  if (!context.insts()
           .IsOneOf<SemIR::ClassDecl, SemIR::FunctionDecl, SemIR::Namespace>(
               enclosing_scope_inst_id)) {
    CARBON_DIAGNOSTIC(
        MatchFirstInWrongScope, Error,
        "found `match_first` in invalid scope; expected namespace or class");
    auto builder = context.emitter().Build(node_id, MatchFirstInWrongScope);
    CARBON_DIAGNOSTIC(MatchFirstInWrongScopeNote, Note,
                      "in enclosing scope here");
    builder.Note(enclosing_scope_inst_id, MatchFirstInWrongScopeNote);
    builder.Emit();
    context.node_stack().Push(node_id, SemIR::ErrorInst::InstId);
    return true;
  }

  auto decl_id = AddInst<SemIR::MatchFirstDecl>(
      context, node_id, {.enclosing_scope_inst_id = enclosing_scope_inst_id});
  context.scope_stack().PushForMatchFirstBlock(decl_id);
  context.node_stack().Push(node_id, decl_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::MatchFirstId /*node_id*/)
    -> bool {
  auto decl_id =
      context.node_stack().Pop<Parse::NodeKind::MatchFirstDefinitionStart>();
  if (decl_id != SemIR::ErrorInst::InstId) {
    context.scope_stack().Pop();
  }
  return true;
}

}  // namespace Carbon::Check
