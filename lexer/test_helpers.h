// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LEXER_TEST_HELPERS_H_
#define LEXER_TEST_HELPERS_H_

#include <string>

#include "diagnostics/diagnostic_emitter.h"
#include "gmock/gmock.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon {
namespace Testing {

// A diagnostic translator for tests that lex a single token. Produces
// locations such as "`12.5`:1:3" to refer to the third character in the token.
class SingleTokenDiagnosticTranslator
    : public DiagnosticLocationTranslator<const char*> {
 public:
  // Form a translator for a given token. The string provided here must refer
  // to the same character array that we are going to lex.
  SingleTokenDiagnosticTranslator(llvm::StringRef token) : token(token) {}

  auto GetLocation(const char* pos) -> Diagnostic::Location override {
    assert(pos >= token.begin() && pos <= token.end() &&
           "invalid diagnostic location");
    llvm::StringRef prefix = token.take_front(pos - token.begin());
    auto [before_last_newline, this_line] = prefix.rsplit('\n');
    if (before_last_newline.size() == prefix.size()) {
      // On first line.
      return {.file_name = file_name,
              .line_number = 1,
              .column_number = static_cast<int32_t>(pos - token.begin() + 1)};
    } else {
      // On second or subsequent lines.
      return {.file_name = file_name,
              .line_number =
                  static_cast<int32_t>(before_last_newline.count('\n') + 2),
              .column_number = static_cast<int32_t>(this_line.size() + 1)};
    }
  }

 private:
  llvm::StringRef token;
  std::string file_name = llvm::formatv("`{0}`", token);
};

}  // namespace Testing
}  // namespace Carbon

#endif  // LEXER_TOKENIZED_BUFFER_TEST_HELPERS_H_
