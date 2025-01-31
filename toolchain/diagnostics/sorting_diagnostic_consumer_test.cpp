// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/sorting_diagnostic_consumer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/StringRef.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/mocks.h"

namespace Carbon {
namespace {

using ::Carbon::Testing::IsSingleDiagnostic;
using ::testing::_;
using ::testing::InSequence;

CARBON_DIAGNOSTIC(TestDiagnostic, Error, "M{0}", int);

class FakeDiagnosticEmitter : public DiagnosticEmitter<int32_t> {
 public:
  using DiagnosticEmitter::DiagnosticEmitter;

 protected:
  auto ConvertLoc(int32_t last_byte_offset, ContextFnT /*context_fn*/) const
      -> ConvertedDiagnosticLoc override {
    return {.loc = {}, .last_byte_offset = last_byte_offset};
  }
};

TEST(SortedDiagnosticEmitterTest, SortErrors) {
  Testing::MockDiagnosticConsumer consumer;
  SortingDiagnosticConsumer sorting_consumer(consumer);
  FakeDiagnosticEmitter emitter(&sorting_consumer);

  emitter.Emit(1, TestDiagnostic, 1);
  emitter.Emit(-1, TestDiagnostic, 2);
  emitter.Emit(0, TestDiagnostic, 3);
  emitter.Emit(4, TestDiagnostic, 4);
  emitter.Emit(3, TestDiagnostic, 5);
  emitter.Emit(3, TestDiagnostic, 6);

  InSequence s;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            DiagnosticKind::TestDiagnostic,
                            DiagnosticLevel::Error, _, _, "M2")));
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            DiagnosticKind::TestDiagnostic,
                            DiagnosticLevel::Error, _, _, "M3")));
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            DiagnosticKind::TestDiagnostic,
                            DiagnosticLevel::Error, _, _, "M1")));
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            DiagnosticKind::TestDiagnostic,
                            DiagnosticLevel::Error, _, _, "M5")));
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            DiagnosticKind::TestDiagnostic,
                            DiagnosticLevel::Error, _, _, "M6")));
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            DiagnosticKind::TestDiagnostic,
                            DiagnosticLevel::Error, _, _, "M4")));
  sorting_consumer.Flush();
}

}  // namespace
}  // namespace Carbon
