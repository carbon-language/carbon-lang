// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/syntax/lex_scan_helper.h"

#include "common/string_helpers.h"
#include "explorer/syntax/lex_helper.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon {

auto StringLexHelper::Advance() -> bool {
  CARBON_CHECK(is_eof_ == false);
  const char c = YyinputWrapper(yyscanner_);
  if (c <= 0) {
    context_.RecordSyntaxError("Unexpected end of file");
    is_eof_ = true;
    return false;
  }
  str_.push_back(c);
  return true;
}

auto ReadHashTags(Carbon::StringLexHelper& scan_helper,
                  const size_t hashtag_num) -> bool {
  for (size_t i = 0; i < hashtag_num; ++i) {
    if (!scan_helper.Advance() || scan_helper.last_char() != '#') {
      return false;
    }
  }
  return true;
}

auto ProcessSingleLineString(llvm::StringRef str,
                             Carbon::ParseAndLexContext& context,
                             const size_t hashtag_num)
    -> Carbon::Parser::symbol_type {
  std::string hashtags(hashtag_num, '#');
  const auto str_with_quote = str;
  CARBON_CHECK(str.consume_front(hashtags + "\"") &&
               str.consume_back("\"" + hashtags));

  std::optional<std::string> unescaped =
      Carbon::UnescapeStringLiteral(str, hashtag_num);
  if (unescaped == std::nullopt) {
    return context.RecordSyntaxError(
        llvm::formatv("Invalid escaping in string: {0}", str_with_quote));
  }
  return CARBON_ARG_TOKEN(string_literal, *unescaped);
}

auto ProcessMultiLineString(llvm::StringRef str,
                            Carbon::ParseAndLexContext& context,
                            const size_t hashtag_num)
    -> Carbon::Parser::symbol_type {
  std::string hashtags(hashtag_num, '#');
  CARBON_CHECK(str.consume_front(hashtags) && str.consume_back(hashtags));
  Carbon::ErrorOr<std::string> block_string =
      Carbon::ParseBlockStringLiteral(str, hashtag_num);
  if (!block_string.ok()) {
    return context.RecordSyntaxError(llvm::formatv(
        "Invalid block string: {0}", block_string.error().message()));
  }
  return CARBON_ARG_TOKEN(string_literal, *block_string);
}

}  // namespace Carbon
