// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_SYNTAX_LEX_SCAN_HELPER_H_
#define CARBON_EXPLORER_SYNTAX_LEX_SCAN_HELPER_H_

#include <string>

#include "explorer/syntax/parse_and_lex_context.h"
#include "explorer/syntax/parser.h"

// Exposes yyinput; defined in lexer.lpp.
extern auto YyinputWrapper(yyscan_t yyscanner) -> int;

namespace Carbon {

class StringLexHelper {
 public:
  StringLexHelper(const char* text, yyscan_t yyscanner,
                  Carbon::ParseAndLexContext& context)
      : str_(text), yyscanner_(yyscanner), context_(context), is_eof_(false) {}
  // Advances yyscanner by one char. Sets is_eof to true and returns false on
  // EOF.
  auto Advance() -> bool;
  // Returns the last scanned char.
  auto last_char() -> char { return str_.back(); };
  // Returns the scanned string.
  auto str() -> const std::string& { return str_; };

  auto is_eof() -> bool { return is_eof_; };

 private:
  std::string str_;
  yyscan_t yyscanner_;
  Carbon::ParseAndLexContext& context_;
  // Skips reading next char.
  bool is_eof_;
};

// Tries to Read `hashtag_num` hashtags. Returns true on success.
// Reads `hashtag_num` characters on success, and number of consecutive hashtags
// (< `hashtag_num`) + 1 characters on failure.
auto ReadHashTags(Carbon::StringLexHelper& scan_helper, size_t hashtag_num)
    -> bool;

// Removes quotes and escapes a single line string. Reports an error on
// invalid escaping.
auto ProcessSingleLineString(llvm::StringRef str,
                             Carbon::ParseAndLexContext& context,
                             size_t hashtag_num) -> Carbon::Parser::symbol_type;
auto ProcessMultiLineString(llvm::StringRef str,
                            Carbon::ParseAndLexContext& context,
                            size_t hashtag_num) -> Carbon::Parser::symbol_type;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_SYNTAX_LEX_SCAN_HELPER_H_
