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

// https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/lexical_conventions/symbolic_tokens.md#overview
// > the token after the operator must be an identifier, a literal, or any kind of opening bracket (for example, (, [, or {).
static bool token_allowed_after_binary_operator(char c) {
  return
      // identifier
      c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
      // string literal
      c == '\"' ||
      // TODO: character literal
      // number literal
      (c >= '0' && c <= '9') ||
      // opening bracket
      c == '(' || c == '[' || c == '{';
}

static bool is_whitespace(char c) { return c == ' ' || c == '\n'; }

// https://tree-sitter.github.io/tree-sitter/creating-parsers#external-scanners
//
// > If a token in the externals array is valid at a given position in the
// > parse, the external scanner will be called first before anything else is
// > done.
//
// > But if the external scanner may return false and in this case Tree-sitter
// > fallbacks to the internal lexing mechanism.
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

  // https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/lexical_conventions/symbolic_tokens.md
  if (is_whitespace(lexer->lookahead) && whitespace) {
    // foo * bar
    lexer->result_symbol = BINARY_STAR;
  } else if (!whitespace &&
             token_allowed_after_binary_operator(lexer->lookahead)) {
    // foo*bar or foo*(bar)
    lexer->result_symbol = BINARY_STAR;
  } else {
    // foo*
    lexer->result_symbol = POSTFIX_STAR;
  }
  return true;
}
