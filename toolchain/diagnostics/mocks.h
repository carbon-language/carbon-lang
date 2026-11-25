// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_MOCKS_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_MOCKS_H_

#include <gmock/gmock.h>

#include "toolchain/diagnostics/emitter.h"

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

// NOLINTNEXTLINE(modernize-use-trailing-return-type): From the macro.
MATCHER_P(IsDiagnosticsLabelString, matcher, "") {
  const Diagnostics::Label& label = arg;
  return testing::ExplainMatchResult(matcher, label.Format(), result_listener);
}

// Matches the location fields both a message and a label carry.
template <typename T>
auto IsAtLoc(testing::Matcher<int> line_number,
             testing::Matcher<int> column_number) -> testing::Matcher<T> {
  using testing::AllOf;
  using testing::Field;
  return Field(
      &T::loc,
      AllOf(Field("line_number", &Diagnostics::Loc::line_number, line_number),
            Field("column_number", &Diagnostics::Loc::column_number,
                  column_number)));
}

inline auto IsDiagnosticMessage(testing::Matcher<Diagnostics::Kind> kind,
                                testing::Matcher<Diagnostics::Level> level,
                                testing::Matcher<int> line_number,
                                testing::Matcher<int> column_number,
                                testing::Matcher<std::string> message)
    -> testing::Matcher<Diagnostics::Message> {
  using testing::AllOf;
  using testing::Field;
  return AllOf(Field("kind", &Diagnostics::Message::kind, kind),
               Field("level", &Diagnostics::Message::level, level),
               IsAtLoc<Diagnostics::Message>(line_number, column_number),
               IsDiagnosticsMessageString(message));
}

inline auto IsDiagnosticLabel(testing::Matcher<int> line_number,
                              testing::Matcher<int> column_number,
                              testing::Matcher<std::string> text)
    -> testing::Matcher<Diagnostics::Label> {
  return testing::AllOf(IsAtLoc<Diagnostics::Label>(line_number, column_number),
                        IsDiagnosticsLabelString(text));
}

inline auto IsDiagnostic(
    testing::Matcher<Diagnostics::Level> level,
    testing::Matcher<Diagnostics::Message> message,
    testing::Matcher<llvm::SmallVector<Diagnostics::Label, 0>> labels)
    -> testing::Matcher<Diagnostics::Diagnostic> {
  return testing::AllOf(
      testing::Field("level", &Diagnostics::Diagnostic::level, level),
      testing::Field("message", &Diagnostics::Diagnostic::message, message),
      testing::Field("labels", &Diagnostics::Diagnostic::labels, labels));
}

// The start and length expected of a location, in `IsDiagnosticWithRange`.
struct ExpectedLoc {
  testing::Matcher<int> line_number;
  testing::Matcher<int> column_number;
  testing::Matcher<int> length;
};

// Matches a diagnostic whose message has one extra range, checking the lengths
// of the message's location and of the range as well as their starts, and that
// has no labels.
inline auto IsDiagnosticWithRange(testing::Matcher<Diagnostics::Kind> kind,
                                  testing::Matcher<Diagnostics::Level> level,
                                  ExpectedLoc loc,
                                  testing::Matcher<std::string> message,
                                  ExpectedLoc range)
    -> testing::Matcher<Diagnostics::Diagnostic> {
  using testing::AllOf;
  using testing::Field;
  return IsDiagnostic(
      level,
      AllOf(IsDiagnosticMessage(kind, level, loc.line_number, loc.column_number,
                                message),
            Field("loc", &Diagnostics::Message::loc,
                  Field("length", &Diagnostics::Loc::length, loc.length)),
            Field("ranges", &Diagnostics::Message::ranges,
                  testing::ElementsAre(AllOf(
                      Field("line_number", &Diagnostics::Loc::line_number,
                            range.line_number),
                      Field("column_number", &Diagnostics::Loc::column_number,
                            range.column_number),
                      Field("length", &Diagnostics::Loc::length,
                            range.length))))),
      testing::IsEmpty());
}

// Matches a diagnostic with a message and nothing else: no extra ranges and no
// labels.
inline auto IsSingleDiagnostic(testing::Matcher<Diagnostics::Kind> kind,
                               testing::Matcher<Diagnostics::Level> level,
                               testing::Matcher<int> line_number,
                               testing::Matcher<int> column_number,
                               testing::Matcher<std::string> message)
    -> testing::Matcher<Diagnostics::Diagnostic> {
  return IsDiagnostic(
      level,
      testing::AllOf(
          IsDiagnosticMessage(kind, level, line_number, column_number, message),
          testing::Field("ranges", &Diagnostics::Message::ranges,
                         testing::IsEmpty())),
      testing::IsEmpty());
}

// Matches a diagnostic with one message, located at the point just before
// `column_number`, and nothing attached to it.
inline auto IsPointDiagnostic(testing::Matcher<Diagnostics::Kind> kind,
                              testing::Matcher<Diagnostics::Level> level,
                              testing::Matcher<int> line_number,
                              testing::Matcher<int> column_number,
                              testing::Matcher<std::string> message)
    -> testing::Matcher<Diagnostics::Diagnostic> {
  return IsDiagnostic(
      level,
      testing::AllOf(
          IsDiagnosticMessage(kind, level, line_number, column_number, message),
          testing::Field(
              "loc", &Diagnostics::Message::loc,
              testing::Field("length", &Diagnostics::Loc::length, 0)),
          testing::Field("ranges", &Diagnostics::Message::ranges,
                         testing::IsEmpty())),
      testing::IsEmpty());
}

}  // namespace Carbon::Testing

namespace Carbon::Diagnostics {

// Printing helpers for tests.
auto PrintTo(const Diagnostic& diagnostic, std::ostream* os) -> void;
auto PrintTo(Level level, std::ostream* os) -> void;

}  // namespace Carbon::Diagnostics

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_MOCKS_H_
