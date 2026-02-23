// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleStructPatternList(Context& context) -> void {
  auto state = context.PopState();

  context.PushStateForPattern(StateKind::StructPatternFinish,
                              state.in_var_pattern, state.in_unused_pattern);
  context.AddLeafNode(NodeKind::StructPatternStart,
                      context.ConsumeChecked(Lex::TokenKind::OpenCurlyBrace));

  if (!context.PositionIs(Lex::TokenKind::CloseCurlyBrace)) {
    context.PushStateForPattern(StateKind::StructPatternField,
                                state.in_var_pattern, state.in_unused_pattern);
  }
}

auto HandleStructPatternField(Context& context) -> void {
  auto state = context.PopState();

  if (!context.PositionIs(Lex::TokenKind::Period)) {
    if (context.PositionIs(Lex::TokenKind::Identifier)) {
      CARBON_DIAGNOSTIC(StructPatternShorthandNotImplemented, Error,
                        "TODO: struct pattern shorthand");
      context.emitter().Emit(*context.position(),
                             StructPatternShorthandNotImplemented);
    } else {
      CARBON_DIAGNOSTIC(ExpectedStructPatternField, Error,
                        "expected `.field = pattern` in struct pattern");
      context.emitter().Emit(*context.position(), ExpectedStructPatternField);
    }
    state.has_error = true;
    context.PushState(state, StateKind::StructPatternFieldFinish);
    return;
  }

  context.PushState(state, StateKind::StructPatternFieldValue);
  context.PushState(StateKind::PeriodAsStruct);
}

auto HandleStructPatternFieldValue(Context& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    auto recovery_pos =
        context.FindNextOf({Lex::TokenKind::Equal, Lex::TokenKind::Comma,
                            Lex::TokenKind::CloseCurlyBrace});
    if (!recovery_pos ||
        context.tokens().GetKind(*recovery_pos) != Lex::TokenKind::Equal) {
      context.PushState(state, StateKind::StructPatternFieldFinish);
      return;
    }
    context.SkipTo(*recovery_pos);
  }

  if (!context.PositionIs(Lex::TokenKind::Equal)) {
    CARBON_DIAGNOSTIC(ExpectedStructPatternFieldEquals, Error,
                      "expected `=` after field designator in struct pattern");
    context.emitter().Emit(*context.position(),
                           ExpectedStructPatternFieldEquals);
    state.has_error = true;
    context.PushState(state, StateKind::StructPatternFieldFinish);
    return;
  }

  state.token = context.Consume();
  context.PushState(state, StateKind::StructPatternFieldFinish);
  context.PushStateForPattern(StateKind::Pattern, state.in_var_pattern,
                              state.in_unused_pattern);
}

auto HandleStructPatternFieldFinish(Context& context) -> void {
  auto state = context.PopState();

  if (state.has_error) {
    context.AddLeafNode(NodeKind::InvalidParse, state.token,
                        /*has_error=*/true);
    context.ReturnErrorOnState();
  } else {
    context.AddNode(NodeKind::StructPatternField, state.token,
                    /*has_error=*/false);
  }

  if (context.ConsumeListToken(
          NodeKind::PatternListComma, Lex::TokenKind::CloseCurlyBrace,
          state.has_error) == Context::ListTokenKind::Comma) {
    context.PushStateForPattern(StateKind::StructPatternField,
                                state.in_var_pattern, state.in_unused_pattern);
  }
}

auto HandleStructPatternFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::StructPattern,
                  context.ConsumeChecked(Lex::TokenKind::CloseCurlyBrace),
                  state.has_error);
}

}  // namespace Carbon::Parse
