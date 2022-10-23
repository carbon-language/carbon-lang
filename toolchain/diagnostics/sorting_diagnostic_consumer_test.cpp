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

CARBON_DIAGNOSTIC(TestDiagnostic, Error, "{0}", llvm::StringRef);

struct FakeDiagnosticLocationTranslator
    : DiagnosticLocationTranslator<DiagnosticLocation> {
  auto GetLocation(DiagnosticLocation loc) -> DiagnosticLocation override {
    return loc;
  }
};

TEST(SortedDiagnosticEmitterTest, SortErrors) {
  FakeDiagnosticLocationTranslator translator;
  Testing::MockDiagnosticConsumer consumer;
  SortingDiagnosticConsumer sorting_consumer(consumer);
  DiagnosticEmitter<DiagnosticLocation> emitter(translator, sorting_consumer);

  emitter.Emit({"f", 2, 1}, TestDiagnostic, "M1");
  emitter.Emit({"f", 1, 1}, TestDiagnostic, "M2");
  emitter.Emit({"f", 1, 3}, TestDiagnostic, "M3");
  emitter.Emit({"f", 3, 4}, TestDiagnostic, "M4");
  emitter.Emit({"f", 3, 2}, TestDiagnostic, "M5");
  emitter.Emit({"f", 3, 2}, TestDiagnostic, "M6");

  InSequence s;
  EXPECT_CALL(consumer, HandleDiagnostic(
                            IsDiagnostic(DiagnosticKind::TestDiagnostic,
                                         DiagnosticLevel::Error, 1, 1, "M2")));
  EXPECT_CALL(consumer, HandleDiagnostic(
                            IsDiagnostic(DiagnosticKind::TestDiagnostic,
                                         DiagnosticLevel::Error, 1, 3, "M3")));
  EXPECT_CALL(consumer, HandleDiagnostic(
                            IsDiagnostic(DiagnosticKind::TestDiagnostic,
                                         DiagnosticLevel::Error, 2, 1, "M1")));
  EXPECT_CALL(consumer, HandleDiagnostic(
                            IsDiagnostic(DiagnosticKind::TestDiagnostic,
                                         DiagnosticLevel::Error, 3, 2, "M5")));
  EXPECT_CALL(consumer, HandleDiagnostic(
                            IsDiagnostic(DiagnosticKind::TestDiagnostic,
                                         DiagnosticLevel::Error, 3, 2, "M6")));
  EXPECT_CALL(consumer, HandleDiagnostic(
                            IsDiagnostic(DiagnosticKind::TestDiagnostic,
                                         DiagnosticLevel::Error, 3, 4, "M4")));
  sorting_consumer.Flush();
}

}  // namespace
}  // namespace Carbon::Testing
