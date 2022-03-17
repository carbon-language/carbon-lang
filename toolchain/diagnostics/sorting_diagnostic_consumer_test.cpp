// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/sorting_diagnostic_consumer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/mocks.h"

namespace Carbon::Testing {
namespace {

using ::testing::InSequence;

struct FakeDiagnostic : DiagnosticBase<FakeDiagnostic> {
  static constexpr llvm::StringLiteral ShortName = "fake-diagnostic";
  // TODO: consider ways to put the Message into `format` to allow dynamic
  // selection of the message.
  static constexpr llvm::StringLiteral Message = "{0}";

  auto Format() -> std::string {
    // Work around a bug in Clang's unused const variable warning by marking it
    // used here with a no-op.
    static_cast<void>(ShortName);

    return llvm::formatv(Message.data(), message).str();
  }

  std::string message;
};

struct FakeDiagnosticLocationTranslator
    : DiagnosticLocationTranslator<Diagnostic::Location> {
  auto GetLocation(Diagnostic::Location loc) -> Diagnostic::Location override {
    return loc;
  }
};

static auto MakeLoc(int line, int col) -> Diagnostic::Location {
  return {.file_name = "test", .line_number = line, .column_number = col};
}

TEST(SortedDiagnosticEmitterTest, EmitErrors) {
  FakeDiagnosticLocationTranslator translator;
  Testing::MockDiagnosticConsumer consumer;
  SortingDiagnosticConsumer sorting_consumer(consumer);
  DiagnosticEmitter<Diagnostic::Location> emitter(translator, sorting_consumer);

  emitter.EmitError<FakeDiagnostic>(MakeLoc(2, 1), {.message = "M1"});
  emitter.EmitError<FakeDiagnostic>(MakeLoc(1, 1), {.message = "M2"});
  emitter.EmitError<FakeDiagnostic>(MakeLoc(1, 3), {.message = "M3"});
  emitter.EmitError<FakeDiagnostic>(MakeLoc(3, 4), {.message = "M4"});
  emitter.EmitError<FakeDiagnostic>(MakeLoc(3, 2), {.message = "M5"});

  InSequence s;
  EXPECT_CALL(consumer, HandleDiagnostic(
                            AllOf(DiagnosticLevel(Diagnostic::Error),
                                  DiagnosticAt(1, 1), DiagnosticMessage("M2"),
                                  DiagnosticShortName("fake-diagnostic"))));
  EXPECT_CALL(consumer, HandleDiagnostic(
                            AllOf(DiagnosticLevel(Diagnostic::Error),
                                  DiagnosticAt(1, 3), DiagnosticMessage("M3"),
                                  DiagnosticShortName("fake-diagnostic"))));
  EXPECT_CALL(consumer, HandleDiagnostic(
                            AllOf(DiagnosticLevel(Diagnostic::Error),
                                  DiagnosticAt(2, 1), DiagnosticMessage("M1"),
                                  DiagnosticShortName("fake-diagnostic"))));
  EXPECT_CALL(consumer, HandleDiagnostic(
                            AllOf(DiagnosticLevel(Diagnostic::Error),
                                  DiagnosticAt(3, 2), DiagnosticMessage("M5"),
                                  DiagnosticShortName("fake-diagnostic"))));
  EXPECT_CALL(consumer, HandleDiagnostic(
                            AllOf(DiagnosticLevel(Diagnostic::Error),
                                  DiagnosticAt(3, 4), DiagnosticMessage("M4"),
                                  DiagnosticShortName("fake-diagnostic"))));

  sorting_consumer.SortAndFlush();
}

}  // namespace
}  // namespace Carbon::Testing
