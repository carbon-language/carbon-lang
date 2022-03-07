// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_DIAGNOSTICS_MOCKS_H_
#define TOOLCHAIN_DIAGNOSTICS_MOCKS_H_

#include <gmock/gmock.h>

#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon::Testing {

class MockDiagnosticConsumer : public DiagnosticConsumer {
 public:
  MOCK_METHOD(void, HandleDiagnostic, (const Diagnostic& diagnostic),
              (override));
};

// Matcher `DiagnosticAt` matches the location of a diagnostic.
MATCHER_P2(DiagnosticAt, line, column, "") {
  const Diagnostic& diag = arg;
  const Diagnostic::Location& loc = diag.location;
  if (loc.line_number != line) {
    *result_listener << "\nExpected diagnostic on line " << line
                     << " but diagnostic is on line " << loc.line_number << ".";
    return false;
  }
  if (loc.column_number != column) {
    *result_listener << "\nExpected diagnostic on column " << column
                     << " but diagnostic is on column " << loc.column_number
                     << ".";
    return false;
  }
  return true;
}

inline auto DiagnosticLevel(Diagnostic::Level level) -> auto {
  return testing::Field(&Diagnostic::level, level);
}

template <typename Matcher>
auto DiagnosticMessage(Matcher&& inner_matcher) -> auto {
  return testing::Field(&Diagnostic::message,
                        std::forward<Matcher&&>(inner_matcher));
}

template <typename Matcher>
auto DiagnosticShortName(Matcher&& inner_matcher) -> auto {
  return testing::Field(&Diagnostic::short_name,
                        std::forward<Matcher&&>(inner_matcher));
}

}  // namespace Carbon::Testing

namespace Carbon {

// Printing helper for tests.
void PrintTo(const Diagnostic& diagnostic, std::ostream* os);

}  // namespace Carbon

#endif  // TOOLCHAIN_DIAGNOSTICS_MOCKS_H_
