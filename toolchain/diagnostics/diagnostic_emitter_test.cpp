// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/diagnostic_emitter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/diagnostics/mocks.h"

namespace Carbon {
namespace {

using Testing::DiagnosticAt;
using Testing::DiagnosticLevel;
using Testing::DiagnosticMessage;
using Testing::DiagnosticShortName;
using ::testing::ElementsAre;
using ::testing::Eq;

struct FakeDiagnostic {
  static constexpr llvm::StringLiteral ShortName = "fake-diagnostic";
  // TODO: consider ways to put the Message into `format` to allow dynamic
  // selection of the message.
  static constexpr llvm::StringLiteral Message = "{0}";

  std::string message;

  auto Format() -> std::string {
    // Work around a bug in Clang's unused const variable warning by marking it
    // used here with a no-op.
    static_cast<void>(ShortName);

    return llvm::formatv(Message.data(), message).str();
  }
};

struct FakeDiagnosticLocationTranslator : DiagnosticLocationTranslator<int> {
  auto GetLocation(int n) -> Diagnostic::Location override {
    return {.file_name = "test", .line_number = 1, .column_number = n};
  }
};

TEST(DiagTest, EmitErrors) {
  FakeDiagnosticLocationTranslator translator;
  Testing::MockDiagnosticConsumer consumer;
  DiagnosticEmitter<int> emitter(translator, consumer);

  EXPECT_CALL(consumer, HandleDiagnostic(
                            AllOf(DiagnosticLevel(Diagnostic::Error),
                                  DiagnosticAt(1, 1), DiagnosticMessage("M1"),
                                  DiagnosticShortName("fake-diagnostic"))));
  EXPECT_CALL(consumer, HandleDiagnostic(
                            AllOf(DiagnosticLevel(Diagnostic::Error),
                                  DiagnosticAt(1, 2), DiagnosticMessage("M2"),
                                  DiagnosticShortName("fake-diagnostic"))));

  emitter.EmitError<FakeDiagnostic>(1, {.message = "M1"});
  emitter.EmitError<FakeDiagnostic>(2, {.message = "M2"});
}

TEST(DiagTest, EmitWarnings) {
  std::vector<std::string> reported;

  FakeDiagnosticLocationTranslator translator;
  Testing::MockDiagnosticConsumer consumer;
  DiagnosticEmitter<int> emitter(translator, consumer);

  EXPECT_CALL(consumer, HandleDiagnostic(
                            AllOf(DiagnosticLevel(Diagnostic::Warning),
                                  DiagnosticAt(1, 3), DiagnosticMessage("M1"),
                                  DiagnosticShortName("fake-diagnostic"))));
  EXPECT_CALL(consumer, HandleDiagnostic(
                            AllOf(DiagnosticLevel(Diagnostic::Warning),
                                  DiagnosticAt(1, 5), DiagnosticMessage("M3"),
                                  DiagnosticShortName("fake-diagnostic"))));

  emitter.EmitWarningIf<FakeDiagnostic>(3, [](FakeDiagnostic& diagnostic) {
    diagnostic.message = "M1";
    return true;
  });
  emitter.EmitWarningIf<FakeDiagnostic>(4, [](FakeDiagnostic& diagnostic) {
    diagnostic.message = "M2";
    return false;
  });
  emitter.EmitWarningIf<FakeDiagnostic>(5, [](FakeDiagnostic& diagnostic) {
    diagnostic.message = "M3";
    return true;
  });
}

}  // namespace
}  // namespace Carbon
