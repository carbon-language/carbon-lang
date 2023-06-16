// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tree_sitter/parser.h"

enum TokenType {
  BINARY_STAR,
  PREFIX_STAR,
  POSTFIX_STAR,
  UNARY_STAR,
};

// our scanner is stateless
void* tree_sitter_carbon_external_scanner_create() { return NULL; }

unsigned tree_sitter_carbon_external_scanner_serialize(void* /* payload */,
                                                       char* /* buffer */) {
  return 0;  // zero bytes used to serialize
}

void tree_sitter_carbon_external_scanner_deserialize(void* /* payload */,
                                                     const char* /* buffer */,
                                                     unsigned /* length */) {}

void tree_sitter_carbon_external_scanner_destroy(void* /* payload */) {}

static bool is_operand_start(char c) {
  // TODO: do we need more?
  return c == '(' || c == '[' || c == '{' || c == '\"' || c == '_' ||
         (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9');
}

static bool is_whitespace(char c) { return c == ' ' || c == '\n'; }

bool tree_sitter_carbon_external_scanner_scan(void* /* payload */,
                                              TSLexer* lexer,
                                              const bool* valid_symbols) {
  // Case 1: called before an expression
  bool whitespace = false;
  if (valid_symbols[PREFIX_STAR]) {
    while (is_whitespace(lexer->lookahead)) {
      whitespace = true;
      lexer->advance(lexer, /* skip= */ true);
    }
    if (lexer->lookahead != '*') {
      return false;
    }

    lexer->advance(lexer, /* skip= */ false);
    if (is_whitespace(lexer->lookahead) && whitespace) {
      lexer->result_symbol = BINARY_STAR;  // does it ever happen?
    } else {
      lexer->result_symbol = PREFIX_STAR;
    }
    return true;
  }

  // Case 2: called after an expression
  else if (valid_symbols[POSTFIX_STAR] || valid_symbols[BINARY_STAR]) {
    while (is_whitespace(lexer->lookahead)) {
      whitespace = true;
      lexer->advance(lexer, /* skip= */ true);
    }
    if (lexer->lookahead != '*') {
      return false;
    }

    lexer->advance(lexer, /* skip= */ false);
    if (is_whitespace(lexer->lookahead) && whitespace) {
      lexer->result_symbol = BINARY_STAR;
    } else if (!whitespace && is_operand_start(lexer->lookahead)) {
      lexer->result_symbol = BINARY_STAR;
    } else {
      lexer->result_symbol = POSTFIX_STAR;
    }
    return true;
  } else {
    // should never happen
    return false;
  }
}
