// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEXER_TOKENIZED_BUFFER_TEST_HELPERS_H_
#define CARBON_TOOLCHAIN_LEXER_TOKENIZED_BUFFER_TEST_HELPERS_H_

#include <gmock/gmock.h>

#include "common/check.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/YAMLParser.h"
#include "toolchain/lexer/tokenized_buffer.h"

namespace Carbon {

inline void PrintTo(const TokenizedBuffer& buffer, std::ostream* output) {
  std::string message;
  llvm::raw_string_ostream message_stream(message);
  message_stream << "\n";
  buffer.Print(message_stream);
  *output << message_stream.str();
}

namespace Testing {

struct ExpectedToken {
  friend auto operator<<(std::ostream& output, const ExpectedToken& expected)
      -> std::ostream& {
    output << "\ntoken: { kind: '" << expected.kind.name().str() << "'";
    if (expected.line != -1) {
      output << ", line: " << expected.line;
    }
    if (expected.column != -1) {
      output << ", column " << expected.column;
    }
    if (expected.indent_column != -1) {
      output << ", indent: " << expected.indent_column;
    }
    if (!expected.text.empty()) {
      output << ", spelling: '" << expected.text.str() << "'";
    }
    if (expected.string_contents) {
      output << ", string contents: '" << expected.string_contents->str()
             << "'";
    }
    if (expected.recovery) {
      output << ", recovery: true";
    }
    output << " }";
    return output;
  }

  TokenKind kind;
  int line = -1;
  int column = -1;
  int indent_column = -1;
  bool recovery = false;
  llvm::StringRef text = "";
  std::optional<llvm::StringRef> string_contents = std::nullopt;
};

// TODO: Consider rewriting this into a `TokenEq` matcher which is used inside
// `ElementsAre`. If that isn't easily done, potentially worth checking for size
// mismatches first.
// NOLINTNEXTLINE: Expands from GoogleTest.
MATCHER_P(HasTokens, raw_all_expected, "") {
  const TokenizedBuffer& buffer = arg;
  llvm::ArrayRef<ExpectedToken> all_expected = raw_all_expected;

  bool matches = true;
  auto buffer_it = buffer.tokens().begin();
  for (const ExpectedToken& expected : all_expected) {
    if (buffer_it == buffer.tokens().end()) {
      // The size check outside the loop will fail and print useful info.
      break;
    }

    int index = buffer_it - buffer.tokens().begin();
    auto token = *buffer_it++;

    TokenKind actual_kind = buffer.GetKind(token);
    if (actual_kind != expected.kind) {
      *result_listener << "\nToken " << index << " is a "
                       << actual_kind.name().str() << ", expected a "
                       << expected.kind.name().str() << ".";
      matches = false;
    }

    int actual_line = buffer.GetLineNumber(token);
    if (expected.line != -1 && actual_line != expected.line) {
      *result_listener << "\nToken " << index << " is at line " << actual_line
                       << ", expected " << expected.line << ".";
      matches = false;
    }

    int actual_column = buffer.GetColumnNumber(token);
    if (expected.column != -1 && actual_column != expected.column) {
      *result_listener << "\nToken " << index << " is at column "
                       << actual_column << ", expected " << expected.column
                       << ".";
      matches = false;
    }

    int actual_indent_column =
        buffer.GetIndentColumnNumber(buffer.GetLine(token));
    if (expected.indent_column != -1 &&
        actual_indent_column != expected.indent_column) {
      *result_listener << "\nToken " << index << " has column indent "
                       << actual_indent_column << ", expected "
                       << expected.indent_column << ".";
      matches = false;
    }

    int actual_recovery = buffer.IsRecoveryToken(token);
    if (expected.recovery != actual_recovery) {
      *result_listener << "\nToken " << index << " is "
                       << (actual_recovery ? "recovery" : "non-recovery")
                       << ", expected "
                       << (expected.recovery ? "recovery" : "non-recovery")
                       << ".";
      matches = false;
    }

    llvm::StringRef actual_text = buffer.GetTokenText(token);
    if (!expected.text.empty() && actual_text != expected.text) {
      *result_listener << "\nToken " << index << " has spelling `"
                       << actual_text.str() << "`, expected `"
                       << expected.text.str() << "`.";
      matches = false;
    }

    CARBON_CHECK(!expected.string_contents ||
                 expected.kind == TokenKind::StringLiteral);
    if (expected.string_contents && actual_kind == TokenKind::StringLiteral) {
      llvm::StringRef actual_contents = buffer.GetStringLiteral(token);
      if (actual_contents != *expected.string_contents) {
        *result_listener << "\nToken " << index << " has contents `"
                         << actual_contents.str() << "`, expected `"
                         << expected.string_contents->str() << "`.";
        matches = false;
      }
    }
  }

  int actual_size = buffer.tokens().end() - buffer.tokens().begin();
  if (static_cast<int>(all_expected.size()) != actual_size) {
    *result_listener << "\nExpected " << all_expected.size()
                     << " tokens but found " << actual_size << ".";
    matches = false;
  }
  return matches;
}

}  // namespace Testing
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LEXER_TOKENIZED_BUFFER_TEST_HELPERS_H_
