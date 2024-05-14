// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_TEST_HELPERS_H_
#define CARBON_TOOLCHAIN_LEX_TEST_HELPERS_H_

#include <gmock/gmock.h>

#include <array>

#include "common/check.h"
#include "common/string_helpers.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon::Testing {

// A diagnostic converter for tests that lex a single token. Produces
// locations such as "`12.5`:1:3" to refer to the third character in the token.
class SingleTokenDiagnosticConverter : public DiagnosticConverter<const char*> {
 public:
  // Form a converter for a given token. The string provided here must refer
  // to the same character array that we are going to lex.
  explicit SingleTokenDiagnosticConverter(llvm::StringRef token)
      : token_(token) {}

  auto ConvertLoc(const char* pos, ContextFnT /*context_fn*/) const
      -> DiagnosticLoc override {
    CARBON_CHECK(StringRefContainsPointer(token_, pos))
        << "invalid diagnostic location";
    llvm::StringRef prefix = token_.take_front(pos - token_.begin());
    auto [before_last_newline, this_line] = prefix.rsplit('\n');
    if (before_last_newline.size() == prefix.size()) {
      // On first line.
      return {.line_number = 1,
              .column_number = static_cast<int32_t>(pos - token_.begin() + 1)};
    } else {
      // On second or subsequent lines. Note that the line number here is 2
      // more than the number of newlines because `rsplit` removed one newline
      // and `line_number` is 1-based.
      return {.line_number =
                  static_cast<int32_t>(before_last_newline.count('\n') + 2),
              .column_number = static_cast<int32_t>(this_line.size() + 1)};
    }
  }

 private:
  llvm::StringRef token_;
};

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_LEX_TEST_HELPERS_H_
