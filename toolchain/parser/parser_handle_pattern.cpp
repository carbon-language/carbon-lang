// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parser/parser_context.h"

namespace Carbon::Parse {

// Handles PatternAs(DeducedParameter|FunctionParameter|Variable).
static auto HandlePattern(Context& context, Context::PatternKind pattern_kind)
    -> void {
  auto state = context.PopState();

  // Parameters may have keywords prefixing the pattern. They become the parent
  // for the full PatternBinding.
  if (pattern_kind != Context::PatternKind::Variable) {
    context.ConsumeIfPatternKeyword(TokenKind::Template, State::PatternTemplate,
                                    state.subtree_start);
    context.ConsumeIfPatternKeyword(TokenKind::Addr, State::PatternAddress,
                                    state.subtree_start);
  }

  // Handle an invalid pattern introducer for parameters and variables.
  auto on_error = [&]() {
    switch (pattern_kind) {
      case Context::PatternKind::DeducedParameter:
      case Context::PatternKind::Parameter: {
        CARBON_DIAGNOSTIC(ExpectedParameterName, Error,
                          "Expected parameter declaration.");
        context.emitter().Emit(*context.position(), ExpectedParameterName);
        break;
      }
      case Context::PatternKind::Variable: {
        CARBON_DIAGNOSTIC(ExpectedVariableName, Error,
                          "Expected pattern in `var` declaration.");
        context.emitter().Emit(*context.position(), ExpectedVariableName);
        break;
      }
    }
    // Add a placeholder for the type.
    context.AddLeafNode(NodeKind::InvalidParse, *context.position(),
                        /*has_error=*/true);
    state.state = State::PatternFinishAsRegular;
    state.has_error = true;
    context.PushState(state);
  };

  // The first item should be an identifier or, for deduced parameters, `self`.
  bool has_name = false;
  if (auto identifier = context.ConsumeIf(TokenKind::Identifier)) {
    context.AddLeafNode(NodeKind::Name, *identifier);
    has_name = true;
  } else if (pattern_kind == Context::PatternKind::DeducedParameter) {
    if (auto self = context.ConsumeIf(TokenKind::SelfValueIdentifier)) {
      context.AddLeafNode(NodeKind::SelfValueName, *self);
      has_name = true;
    }
  }
  if (!has_name) {
    // Add a placeholder for the name.
    context.AddLeafNode(NodeKind::Name, *context.position(),
                        /*has_error=*/true);
    on_error();
    return;
  }

  if (auto kind = context.PositionKind();
      kind == TokenKind::Colon || kind == TokenKind::ColonExclaim) {
    state.state = kind == TokenKind::Colon ? State::PatternFinishAsRegular
                                           : State::PatternFinishAsGeneric;
    // Use the `:` or `:!` for the root node.
    state.token = context.Consume();
    context.PushState(state);
    context.PushStateForExpression(PrecedenceGroup::ForType());
  } else {
    on_error();
    return;
  }
}

auto HandlePatternAsDeducedParameter(Context& context) -> void {
  HandlePattern(context, Context::PatternKind::DeducedParameter);
}

auto HandlePatternAsParameter(Context& context) -> void {
  HandlePattern(context, Context::PatternKind::Parameter);
}

auto HandlePatternAsVariable(Context& context) -> void {
  HandlePattern(context, Context::PatternKind::Variable);
}

// Handles PatternFinishAs(Generic|Regular).
static auto HandlePatternFinish(Context& context, NodeKind node_kind) -> void {
  auto state = context.PopState();

  context.AddNode(node_kind, state.token, state.subtree_start, state.has_error);

  // Propagate errors to the parent state so that they can take different
  // actions on invalid patterns.
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

auto HandlePatternFinishAsGeneric(Context& context) -> void {
  HandlePatternFinish(context, NodeKind::GenericPatternBinding);
}

auto HandlePatternFinishAsRegular(Context& context) -> void {
  HandlePatternFinish(context, NodeKind::PatternBinding);
}

auto HandlePatternAddress(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::Address, state.token, state.subtree_start,
                  state.has_error);

  // If an error was encountered, propagate it while adding a node.
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

auto HandlePatternTemplate(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::Template, state.token, state.subtree_start,
                  state.has_error);

  // If an error was encountered, propagate it while adding a node.
  if (state.has_error) {
    context.ReturnErrorOnState();
  }
}

}  // namespace Carbon::Parse
