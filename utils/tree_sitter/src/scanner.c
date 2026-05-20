// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "utils/tree_sitter/src/tree_sitter/parser.h"

typedef enum {
  BINARY_STAR,
  POSTFIX_STAR,
  STRING,
} TokenType;

// This is part of a special rule that doesn't allow `copts` in Bazel, so we
// disable warnings using `#pragma`s here.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"

// our scanner is stateless
void* tree_sitter_carbon_external_scanner_create(void) {
  return NULL;
}

unsigned tree_sitter_carbon_external_scanner_serialize(
    void* payload,
    char* buffer) {
  (void)payload;
  (void)buffer;
  return 0;
}

void tree_sitter_carbon_external_scanner_deserialize(
    void* payload,
    const char* buffer,
    unsigned length) {
  (void)payload;
  (void)buffer;
  (void)length;
}

void tree_sitter_carbon_external_scanner_destroy(void* payload) {
  (void)payload;
}

// https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/lexical_conventions/symbolic_tokens.md#overview
static bool token_allowed_after_binary_operator(char c) {
  return
      (c >= 'a' && c <= 'z') ||
      (c >= 'A' && c <= 'Z') ||
      c == '_' ||
      c == '"' ||
      (c >= '0' && c <= '9') ||
      c == '(' || c == '[' || c == '{';
}

static bool is_whitespace(char c) {
  return c == ' ' || c == '\n';
}

static void advance(TSLexer* lexer) {
  lexer->advance(lexer, false);
}

static int eat_count(TSLexer* lexer, char ch, int count) {
  int matched = 0;
  while (matched < count && lexer->lookahead == ch) {
    advance(lexer);
    matched++;
  }
  return matched;
}

bool tree_sitter_carbon_external_scanner_scan(
    void* payload,
    TSLexer* lexer,
    const bool* valid_symbols) {

  (void)payload;

  bool saw_whitespace = false;

  while (is_whitespace(lexer->lookahead)) {
    saw_whitespace = true;
    lexer->advance(lexer, true);
  }

  // -------- STAR --------
  if (lexer->lookahead == '*' &&
      (valid_symbols[BINARY_STAR] || valid_symbols[POSTFIX_STAR])) {

    lexer->advance(lexer, false);

    if (is_whitespace(lexer->lookahead) && saw_whitespace) {
      lexer->result_symbol = BINARY_STAR;
      return true;
    }

    if (!saw_whitespace &&
        token_allowed_after_binary_operator(lexer->lookahead)) {
      lexer->result_symbol = BINARY_STAR;
      return true;
    }

    lexer->result_symbol = POSTFIX_STAR;
    return true;
  }

  // -------- STRING --------
  if ((lexer->lookahead == '#' ||
       lexer->lookahead == '\'' ||
       lexer->lookahead == '"') &&
      valid_symbols[STRING]) {

    lexer->result_symbol = STRING;

    int hash_count = 0;

    while (lexer->lookahead == '#') {
      advance(lexer);
      hash_count++;
    }

    // ''' multiline
    if (lexer->lookahead == '\'') {
      if (eat_count(lexer, '\'', 3) != 3) {
        return false;
      }

      while (!lexer->eof(lexer)) {
        if (lexer->lookahead == '\n') {
          advance(lexer);

          while (lexer->lookahead == ' ') {
            advance(lexer);
          }

          if (eat_count(lexer, '\'', 3) != 3) {
            continue;
          }

          if (eat_count(lexer, '#', hash_count) == hash_count) {
            return true;
          }

          continue;
        }

        advance(lexer);
      }
      return false;
    }

    // "string"
    if (lexer->lookahead == '"') {
      advance(lexer);

      while (!lexer->eof(lexer)) {
        if (lexer->lookahead == '\\') {
          advance(lexer);

          if (eat_count(lexer, '#', hash_count) == hash_count) {
            if (lexer->lookahead != '\n') {
              advance(lexer);
            }
          } else {
            continue;
          }

        } else if (lexer->lookahead == '"') {
          advance(lexer);

          if (eat_count(lexer, '#', hash_count) == hash_count) {
            return true;
          }

          continue;

        } else if (lexer->lookahead == '\n') {
          return false;

        } else {
          advance(lexer);
        }
      }

      return false;
    }

    return false;
  }

  return false;
}
