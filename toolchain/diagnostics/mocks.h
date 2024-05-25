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

// NOLINTNEXTLINE(modernize-use-trailing-return-type): From the macro.
MATCHER_P(IsDiagnosticMessageString, matcher, "") {
  const DiagnosticMessage& message = arg;
  return testing::ExplainMatchResult(matcher, message.format_fn(message),
                                     result_listener);
}

inline auto IsDiagnosticMessage(testing::Matcher<DiagnosticKind> kind,
                                testing::Matcher<DiagnosticLevel> level,
                                testing::Matcher<int> line_number,
                                testing::Matcher<int> column_number,
                                testing::Matcher<std::string> message)
    -> testing::Matcher<DiagnosticMessage> {
  using testing::AllOf;
  using testing::Field;
  return AllOf(Field("kind", &DiagnosticMessage::kind, kind),
               Field("level", &DiagnosticMessage::level, level),
               Field(&DiagnosticMessage::loc,
                     AllOf(Field("line_number", &DiagnosticLoc::line_number,
                                 line_number),
                           Field("column_number", &DiagnosticLoc::column_number,
                                 column_number))),
               IsDiagnosticMessageString(message));
}

inline auto IsDiagnostic(
    testing::Matcher<DiagnosticLevel> level,
    testing::Matcher<llvm::SmallVector<DiagnosticMessage>> elements)
    -> testing::Matcher<Diagnostic> {
  return testing::AllOf(
      testing::Field("level", &Diagnostic::level, level),
      testing::Field("messages", &Diagnostic::messages, elements));
}

inline auto IsSingleDiagnostic(testing::Matcher<DiagnosticKind> kind,
                               testing::Matcher<DiagnosticLevel> level,
                               testing::Matcher<int> line_number,
                               testing::Matcher<int> column_number,
                               testing::Matcher<std::string> message)
    -> testing::Matcher<Diagnostic> {
  return IsDiagnostic(
      level, testing::ElementsAre(IsDiagnosticMessage(kind, level, line_number,
                                                      column_number, message)));
}

}  // namespace Carbon::Testing

namespace Carbon {

// Printing helpers for tests.
void PrintTo(const Diagnostic& diagnostic, std::ostream* os);
void PrintTo(DiagnosticLevel level, std::ostream* os);

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_MOCKS_H_
