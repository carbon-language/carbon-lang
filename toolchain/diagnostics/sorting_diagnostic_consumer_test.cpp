// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/sorting_diagnostic_consumer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/StringRef.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/mocks.h"

namespace Carbon::Diagnostics {
namespace {

using ::Carbon::Testing::IsSingleDiagnostic;
using ::testing::_;
using ::testing::InSequence;

CARBON_DIAGNOSTIC(TestDiagnostic, Error, "M{0}", int);

class FakeEmitter : public Emitter<int32_t> {
 public:
  using Emitter::Emitter;

 protected:
  auto ConvertLoc(int32_t last_byte_offset, ContextFnT /*context_fn*/) const
      -> ConvertedLoc override {
    return {.loc = {}, .last_byte_offset = last_byte_offset};
  }
};

TEST(SortedEmitterTest, SortErrors) {
  Testing::MockDiagnosticConsumer consumer;
  SortingConsumer sorting_consumer(consumer);
  FakeEmitter emitter(&sorting_consumer);

  emitter.Emit(1, TestDiagnostic, 1);
  emitter.Emit(-1, TestDiagnostic, 2);
  emitter.Emit(0, TestDiagnostic, 3);
  emitter.Emit(4, TestDiagnostic, 4);
  emitter.Emit(3, TestDiagnostic, 5);
  emitter.Emit(3, TestDiagnostic, 6);

  InSequence s;
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            Kind::TestDiagnostic, Level::Error, _, _, "M2")));
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            Kind::TestDiagnostic, Level::Error, _, _, "M3")));
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            Kind::TestDiagnostic, Level::Error, _, _, "M1")));
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            Kind::TestDiagnostic, Level::Error, _, _, "M5")));
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            Kind::TestDiagnostic, Level::Error, _, _, "M6")));
  EXPECT_CALL(consumer, HandleDiagnostic(IsSingleDiagnostic(
                            Kind::TestDiagnostic, Level::Error, _, _, "M4")));
  sorting_consumer.Flush();
}

}  // namespace
}  // namespace Carbon::Diagnostics
