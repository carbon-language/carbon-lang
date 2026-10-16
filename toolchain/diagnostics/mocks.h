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

// NOLINTNEXTLINE(modernize-use-trailing-return-type): From the macro.
MATCHER_P(IsDiagnosticsContextString, matcher, "") {
  const Diagnostics::Context& context = arg;
  return testing::ExplainMatchResult(matcher, context.Format(),
                                     result_listener);
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

inline auto IsDiagnosticLabel(
    testing::Matcher<Diagnostics::LabelCategory> category,
    testing::Matcher<int> line_number, testing::Matcher<int> column_number,
    testing::Matcher<std::string> text)
    -> testing::Matcher<Diagnostics::Label> {
  using testing::AllOf;
  using testing::Field;
  return AllOf(Field("category", &Diagnostics::Label::category, category),
               IsAtLoc<Diagnostics::Label>(line_number, column_number),
               IsDiagnosticsLabelString(text));
}

inline auto IsDiagnosticContext(testing::Matcher<int> line_number,
                                testing::Matcher<int> column_number,
                                testing::Matcher<std::string> text)
    -> testing::Matcher<Diagnostics::Context> {
  return testing::AllOf(
      IsAtLoc<Diagnostics::Context>(line_number, column_number),
      IsDiagnosticsContextString(text));
}

inline auto IsDiagnostic(
    testing::Matcher<Diagnostics::Level> level,
    testing::Matcher<Diagnostics::Message> message,
    testing::Matcher<llvm::SmallVector<Diagnostics::Context, 0>> contexts,
    testing::Matcher<llvm::SmallVector<Diagnostics::Label, 0>> labels)
    -> testing::Matcher<Diagnostics::Diagnostic> {
  return testing::AllOf(
      testing::Field("level", &Diagnostics::Diagnostic::level, level),
      testing::Field("message", &Diagnostics::Diagnostic::message, message),
      testing::Field("contexts", &Diagnostics::Diagnostic::contexts, contexts),
      testing::Field("labels", &Diagnostics::Diagnostic::labels, labels));
}

inline auto IsSingleDiagnostic(testing::Matcher<Diagnostics::Kind> kind,
                               testing::Matcher<Diagnostics::Level> level,
                               testing::Matcher<int> line_number,
                               testing::Matcher<int> column_number,
                               testing::Matcher<std::string> message)
    -> testing::Matcher<Diagnostics::Diagnostic> {
  return IsDiagnostic(
      level,
      IsDiagnosticMessage(kind, level, line_number, column_number, message),
      testing::IsEmpty(), testing::IsEmpty());
}

}  // namespace Carbon::Testing

namespace Carbon::Diagnostics {

// Printing helpers for tests.
auto PrintTo(const Diagnostic& diagnostic, std::ostream* os) -> void;
auto PrintTo(Level level, std::ostream* os) -> void;
auto PrintTo(LabelCategory category, std::ostream* os) -> void;

}  // namespace Carbon::Diagnostics

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_MOCKS_H_
