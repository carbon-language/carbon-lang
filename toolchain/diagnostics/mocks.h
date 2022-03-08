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
  MOCK_METHOD(void, HandleDiagnostic,
              (const Diagnostic& diagnostic, const DiagnosticLocation& loc),
              (override));
};

// Matcher `DiagnosticAt` matches the location of a diagnostic.
inline auto DiagnosticAt(int line, int column) {
  return testing::AllOf(
      testing::Field(&DiagnosticLocation::line_number, line),
      testing::Field(&DiagnosticLocation::column_number, column));
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
void PrintTo(const DiagnosticLocation& loc, std::ostream* os);

}  // namespace Carbon

#endif  // TOOLCHAIN_DIAGNOSTICS_MOCKS_H_
