// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/keyword_modifier_set.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/diagnostics/diagnostic.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context,
                     Parse::MatchFirstIntroducerId /*node_id*/) -> bool {
  // Optional modifiers follow.
  context.decl_introducer_state_stack().Push<Lex::TokenKind::MatchFirst>();
  return true;
}

auto HandleParseNode(Context& context,
                     Parse::MatchFirstDefinitionStartId node_id) -> bool {
  auto enclosing_scope_inst_id = context.scope_stack().PeekInstId();
  if (context.match_first_context() ||
      !context.insts()
           .IsOneOf<SemIR::ClassDecl, SemIR::FunctionDecl, SemIR::Namespace>(
               enclosing_scope_inst_id)) {
    CARBON_DIAGNOSTIC(
        MatchFirstInWrongScope, Error,
        "found `match_first` in invalid scope; expected namespace or class");
    auto builder = context.emitter().Build(node_id, MatchFirstInWrongScope);
    CARBON_DIAGNOSTIC_LABEL(MatchFirstInWrongScopeNote, Info,
                            "in enclosing scope here");
    builder.Attach(enclosing_scope_inst_id, MatchFirstInWrongScopeNote);
    builder.Emit();
    context.node_stack().Push(node_id, SemIR::ErrorInst::InstId);
    return true;
  }

  const auto& introducer = context.decl_introducer_state_stack().innermost();
  bool is_final = introducer.modifier_set.HasAnyOf(KeywordModifierSet::Final);

  auto decl_id = AddInst<SemIR::MatchFirstDecl>(
      context, node_id, {.enclosing_scope_inst_id = enclosing_scope_inst_id});
  context.match_first_context() = {.decl_id = decl_id, .is_final = is_final};
  context.node_stack().Push(node_id, decl_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::MatchFirstId /*node_id*/)
    -> bool {
  auto decl_id =
      context.node_stack().Pop<Parse::NodeKind::MatchFirstDefinitionStart>();
  if (decl_id != SemIR::ErrorInst::InstId) {
    context.match_first_context() = std::nullopt;
  }

  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::MatchFirst>();
  // Diagnose modifiers that are not allowed.
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::MatchFirst);

  return true;
}

}  // namespace Carbon::Check
