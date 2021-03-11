// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef DIAGNOSTICS_TEST_HELPERS_H_
#define DIAGNOSTICS_TEST_HELPERS_H_

#include "diagnostics/diagnostic_emitter.h"
#include "gmock/gmock.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/YAMLParser.h"

namespace Carbon {

template <typename LocationT>
inline auto NullDiagnosticLocationTranslator()
    -> DiagnosticLocationTranslator<LocationT>& {
  struct Translator : DiagnosticLocationTranslator<LocationT> {
    auto GetLocation(LocationT) -> Diagnostic::Location override { return {}; }
  };
  static Translator* translator = new Translator;
  return *translator;
}

inline auto NullDiagnosticConsumer() -> DiagnosticConsumer& {
  struct Consumer : DiagnosticConsumer {
    auto HandleDiagnostic(const Diagnostic& d) -> void override {}
  };
  static auto* consumer = new Consumer;
  return *consumer;
}

template <typename LocationT>
inline auto NullDiagnosticEmitter() -> DiagnosticEmitter<LocationT>& {
  static auto* emitter = new DiagnosticEmitter<LocationT>(
      NullDiagnosticLocationTranslator<LocationT>(), NullDiagnosticConsumer());
  return *emitter;
}

namespace Testing {

class MockDiagnosticConsumer : public DiagnosticConsumer {
 public:
  MOCK_METHOD1(HandleDiagnostic, void(const Diagnostic& diagnostic));
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

}  // namespace Testing
}  // namespace Carbon

#endif  // DIAGNOSTICS_TEST_HELPERS_H_
