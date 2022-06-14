// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_SYNTAX_LEX_HELPER_H_
#define CARBON_EXPLORER_SYNTAX_LEX_HELPER_H_

// Flex expands this macro immediately before each action.
//
// Advances the current token position by yyleng columns without changing
// the line number, and takes us out of the after-whitespace / after-operand
// state.
#define YY_USER_ACTION                                             \
  context.current_token_position.columns(yyleng);                  \
  if (YY_START == AFTER_WHITESPACE || YY_START == AFTER_OPERAND) { \
    BEGIN(INITIAL);                                                \
  }

#define CARBON_SIMPLE_TOKEN(name) \
  Carbon::Parser::make_##name(context.current_token_position);

#define CARBON_ARG_TOKEN(name, arg) \
  Carbon::Parser::make_##name(arg, context.current_token_position);

#endif  // CARBON_EXPLORER_SYNTAX_LEX_HELPER_H_
