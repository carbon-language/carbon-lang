// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_MOCKS_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_MOCKS_H_

#include <gmock/gmock.h>

#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon::Testing {

class MockDiagnosticConsumer : public Diagnostics::Consumer {
 public:
  MOCK_METHOD(void, HandleDiagnostic, (Diagnostics::Diagnostic diagnostic),
              (override));
};

// NOLINTNEXTLINE(modernize-use-trailing-return-type): From the macro.
MATCHER_P(IsDiagnosticsMessageString, matcher, "") {
  const Diagnostics::Message& message = arg;
  return testing::ExplainMatchResult(matcher, message.Format(),
                                     result_listener);
}

inline auto IsDiagnosticMessage(testing::Matcher<Diagnostics::Kind> kind,
                                testing::Matcher<Diagnostics::Level> level,
                                testing::Matcher<int> line_number,
                                testing::Matcher<int> column_number,
                                testing::Matcher<std::string> message)
    -> testing::Matcher<Diagnostics::Message> {
  using testing::AllOf;
  using testing::Field;
  return AllOf(
      Field("kind", &Diagnostics::Message::kind, kind),
      Field("level", &Diagnostics::Message::level, level),
      Field(&Diagnostics::Message::loc,
            AllOf(Field("line_number", &Diagnostics::Loc::line_number,
                        line_number),
                  Field("column_number", &Diagnostics::Loc::column_number,
                        column_number))),
      IsDiagnosticsMessageString(message));
}

inline auto IsDiagnostic(
    testing::Matcher<Diagnostics::Level> level,
    testing::Matcher<llvm::SmallVector<Diagnostics::Message>> elements)
    -> testing::Matcher<Diagnostics::Diagnostic> {
  return testing::AllOf(
      testing::Field("level", &Diagnostics::Diagnostic::level, level),
      testing::Field("messages", &Diagnostics::Diagnostic::messages, elements));
}

inline auto IsSingleDiagnostic(testing::Matcher<Diagnostics::Kind> kind,
                               testing::Matcher<Diagnostics::Level> level,
                               testing::Matcher<int> line_number,
                               testing::Matcher<int> column_number,
                               testing::Matcher<std::string> message)
    -> testing::Matcher<Diagnostics::Diagnostic> {
  return IsDiagnostic(
      level, testing::ElementsAre(IsDiagnosticMessage(kind, level, line_number,
                                                      column_number, message)));
}

}  // namespace Carbon::Testing

namespace Carbon::Diagnostics {

// Printing helpers for tests.
auto PrintTo(const Diagnostic& diagnostic, std::ostream* os) -> void;
auto PrintTo(Level level, std::ostream* os) -> void;

}  // namespace Carbon::Diagnostics

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_MOCKS_H_
