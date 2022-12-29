// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_MOCKS_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_MOCKS_H_

#include <gmock/gmock.h>

#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon::Testing {

class MockDiagnosticConsumer : public DiagnosticConsumer {
 public:
  MOCK_METHOD(void, HandleDiagnostic, (Diagnostic diagnostic), (override));
};

MATCHER_P(IsDiagnosticMessage, matcher, "") {
  const Diagnostic& diag = arg;
  return testing::ExplainMatchResult(
      matcher, diag.message.format_fn(diag.message), result_listener);
}

inline auto IsDiagnostic(testing::Matcher<DiagnosticKind> kind,
                         testing::Matcher<DiagnosticLevel> level,
                         testing::Matcher<int> line_number,
                         testing::Matcher<int> column_number,
                         testing::Matcher<std::string> message) {
  return testing::AllOf(
      testing::Field("level", &Diagnostic::level, level),
      testing::Field(
          "message", &Diagnostic::message,
          testing::AllOf(
              testing::Field("kind", &DiagnosticMessage::kind, kind),
              testing::Field(
                  &DiagnosticMessage::location,
                  testing::AllOf(
                      testing::Field("line_number",
                                     &DiagnosticLocation::line_number,
                                     line_number),
                      testing::Field("column_number",
                                     &DiagnosticLocation::column_number,
                                     column_number))))),
      IsDiagnosticMessage(message));
}

}  // namespace Carbon::Testing

namespace Carbon {

// Printing helpers for tests.
void PrintTo(const Diagnostic& diagnostic, std::ostream* os);
void PrintTo(DiagnosticLevel level, std::ostream* os);

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_MOCKS_H_
