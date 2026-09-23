// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tree_sitter/parser.h"

enum TokenType {
  BINARY_STAR,
  POSTFIX_STAR,
  STRING,
};

// our scanner is stateless
void* tree_sitter_carbon_external_scanner_create(void) { return NULL; }

unsigned tree_sitter_carbon_external_scanner_serialize(
    __attribute__((unused)) void* payload,
    __attribute__((unused)) char* buffer) {
  return 0;  // zero bytes used to serialize
}

void tree_sitter_carbon_external_scanner_deserialize(
    __attribute__((unused)) void* payload,
    __attribute__((unused)) const char* buffer,
    __attribute__((unused)) unsigned length) {}

void tree_sitter_carbon_external_scanner_destroy(
    __attribute__((unused)) void* payload) {}

// https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/lexical_conventions/symbolic_tokens.md#overview
// > the token after the operator must be an identifier, a literal, or any kind
// of opening bracket (for example, (, [, or {).
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

static void advance(struct TSLexer* lexer) {
  lexer->advance(lexer, /* skip= */ false);
}

static int eat_count(struct TSLexer* lexer, char ch, int count) {
  int matched = 0;
  while (matched != count) {
    if (lexer->lookahead != ch) {
      break;
    }
    advance(lexer);
    matched++;
  }
  return matched;
}

// https://tree-sitter.github.io/tree-sitter/creating-parsers#external-scanners
//
// > If a token in the externals array is valid at a given position in the
// > parse, the external scanner will be called first before anything else is
// > done.
//
// > But the external scanner may return false and in this case Tree-sitter
// > fallbacks to the internal lexing mechanism.
bool tree_sitter_carbon_external_scanner_scan(
    __attribute__((unused)) void* payload,
    __attribute__((unused)) TSLexer* lexer, const bool* valid_symbols) {
  // skip past whitespace if any
  bool whitespace = false;
  while (is_whitespace(lexer->lookahead)) {
    whitespace = true;
    lexer->advance(lexer, /* skip= */ true);
  }

  if (lexer->lookahead == '*' &&
      (valid_symbols[BINARY_STAR] || valid_symbols[POSTFIX_STAR])) {
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
  } else if ((lexer->lookahead == '#' || lexer->lookahead == '\'' ||
              lexer->lookahead == '\"') &&
             valid_symbols[STRING]) {
    lexer->result_symbol = STRING;
    int hash_count = 0;
    while (lexer->lookahead == '#') {
      advance(lexer);
      hash_count++;
    }
    if (lexer->lookahead == '\'') {
      if (eat_count(lexer, '\'', 3) != 3) {
        // lexer state is ignored on return false
        return false;
      }
      while (!lexer->eof(lexer)) {
        // ''' must be on its own line.
        // so look for \n *'''#{hash_count}
        // we can ignore escapes
        if (lexer->lookahead == '\n') {
          advance(lexer);
          while (lexer->lookahead == ' ') {
            advance(lexer);
          }
          if (eat_count(lexer, '\'', 3) != 3) {
            // treat less than 3 `'`s as normal characters
            continue;
          }
          if (eat_count(lexer, '#', hash_count) == hash_count) {
            // end of string
            return true;
          } else {
            // treat as normal characters
            continue;
          }
        } else {
          // skip not new lines
          advance(lexer);
        }
      }
    } else if (lexer->lookahead == '"') {
      advance(lexer);
      while (!lexer->eof(lexer)) {
        if (lexer->lookahead == '\\') {
          advance(lexer);
          if (eat_count(lexer, '#', hash_count) == hash_count) {
            // treat next character as not special
            // but \n is not allowed in simple string
            if (lexer->lookahead != '\n') {
              advance(lexer);
            }
          } else {
            // not an escape
            continue;
          }
        } else if (lexer->lookahead == '"') {
          advance(lexer);
          if (eat_count(lexer, '#', hash_count) == hash_count) {
            // end of string
            return true;
          } else {
            continue;
          }
        } else if (lexer->lookahead == '\n') {
          // new line is not allowed in simple string
          return false;
        } else {
          advance(lexer);
        }
      }
    } else {
      // # not followed by ' or "
      return false;
    }
  }
  return false;
}
