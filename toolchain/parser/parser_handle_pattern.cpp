// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon {

// Handles PatternAs(DeducedParameter|FunctionParameter|Variable).
static auto ParserHandlePattern(ParserContext& context,
                                ParserContext::PatternKind pattern_kind)
    -> void {
  auto state = context.PopState();

  // Parameters may have keywords prefixing the pattern. They become the parent
  // for the full PatternBinding.
  if (pattern_kind != ParserContext::PatternKind::Variable) {
    context.ConsumeIfPatternKeyword(
        TokenKind::Template, ParserState::PatternTemplate, state.subtree_start);
    context.ConsumeIfPatternKeyword(
        TokenKind::Addr, ParserState::PatternAddress, state.subtree_start);
  }

  // ParserHandle an invalid pattern introducer for parameters and variables.
  auto on_error = [&]() {
    switch (pattern_kind) {
      case ParserContext::PatternKind::DeducedParameter:
      case ParserContext::PatternKind::Parameter: {
        CARBON_DIAGNOSTIC(ExpectedParameterName, Error,
                          "Expected parameter declaration.");
        context.emitter().Emit(*context.position(), ExpectedParameterName);
        break;
      }
      case ParserContext::PatternKind::Variable: {
        CARBON_DIAGNOSTIC(ExpectedVariableName, Error,
                          "Expected pattern in `var` declaration.");
        context.emitter().Emit(*context.position(), ExpectedVariableName);
        break;
      }
    }
    // Add a placeholder for the type.
    context.AddLeafNode(ParseNodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
    state.state = ParserState::PatternFinishAsRegular;
    state.has_error = true;
    context.PushState(state);
  };

  // The first item should be an identifier or, for deduced parameters, `self`.
  bool has_name = false;
  if (auto identifier = context.ConsumeIf(TokenKind::Identifier)) {
    context.AddLeafNode(ParseNodeKind::DeclaredName, *identifier);
    has_name = true;
  } else if (pattern_kind == ParserContext::PatternKind::DeducedParameter) {
    if (auto self = context.ConsumeIf(TokenKind::SelfValueIdentifier)) {
      context.AddLeafNode(ParseNodeKind::SelfValueIdentifier, *self);
      has_name = true;
    }
  }
  if (!has_name) {
    // Add a placeholder for the name.
    context.AddLeafNode(ParseNodeKind::DeclaredName, *context.position(),
                        /*has_error=*/true);
    on_error();
    return;
  }

  if (auto kind = context.PositionKind();
      kind == TokenKind::Colon || kind == TokenKind::ColonExclaim) {
    state.state = kind == TokenKind::Colon
                      ? ParserState::PatternFinishAsRegular
                      : ParserState::PatternFinishAsGeneric;
    // Use the `:` or `:!` for the root node.
    state.token = context.Consume();
    context.PushState(state);
    context.PushStateForExpression(PrecedenceGroup::ForType());
  } else {
    on_error();
    return;
  }
}

auto ParserHandlePatternAsDeducedParameter(ParserContext& context) -> void {
  ParserHandlePattern(context, ParserContext::PatternKind::DeducedParameter);
}

auto ParserHandlePatternAsParameter(ParserContext& context) -> void {
  ParserHandlePattern(context, ParserContext::PatternKind::Parameter);
}

auto ParserHandlePatternAsVariable(ParserContext& context) -> void {
  ParserHandlePattern(context, ParserContext::PatternKind::Variable);
}

// Handles PatternFinishAs(Generic|Regular).
static auto ParserHandlePatternFinish(ParserContext& context,
                                      ParseNodeKind node_kind) -> void {
  auto state = context.PopState();

  context.AddNode(node_kind, state.token, state.subtree_start, state.has_error);

  // Propagate errors to the parent state so that they can take different
  // actions on invalid patterns.
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

auto ParserHandlePatternFinishAsGeneric(ParserContext& context) -> void {
  ParserHandlePatternFinish(context, ParseNodeKind::GenericPatternBinding);
}

auto ParserHandlePatternFinishAsRegular(ParserContext& context) -> void {
  ParserHandlePatternFinish(context, ParseNodeKind::PatternBinding);
}

auto ParserHandlePatternAddress(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddNode(ParseNodeKind::Address, state.token, state.subtree_start,
                  state.has_error);

  // If an error was encountered, propagate it while adding a node.
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

auto ParserHandlePatternTemplate(ParserContext& context) -> void {
  auto state = context.PopState();

  context.AddNode(ParseNodeKind::Template, state.token, state.subtree_start,
                  state.has_error);

  // If an error was encountered, propagate it while adding a node.
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

}  // namespace Carbon
