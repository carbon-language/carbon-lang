// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tree_sitter/parser.h"

enum TokenType {
  BINARY_STAR,
  POSTFIX_STAR,
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
  return c == '(' || c == '[' || c == '{' || c == '\"' || c == '_' ||
         (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9');
}

static bool is_whitespace(char c) { return c == ' ' || c == '\n'; }

bool tree_sitter_carbon_external_scanner_scan(void* /* payload */,
                                              TSLexer* lexer,
                                              const bool* /* valid_symbols */) {
  // skip past whitespace if any
  bool whitespace = false;
  while (is_whitespace(lexer->lookahead)) {
    whitespace = true;
    lexer->advance(lexer, /* skip= */ true);
  }

  // if any other symbol than *, fallback to treesitter internal lexer
  if (lexer->lookahead != '*') {
    return false;
  }

  // move to past the *, add * to current token
  lexer->advance(lexer, /* skip= */ false);

  if (is_whitespace(lexer->lookahead) && whitespace) {
    // foo * bar
    lexer->result_symbol = BINARY_STAR;
  } else if (!whitespace && is_operand_start(lexer->lookahead)) {
    // foo*bar or foo*(bar)
    lexer->result_symbol = BINARY_STAR;
  } else {
    // foo*
    lexer->result_symbol = POSTFIX_STAR;
  }
  return true;
}
