// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/diagnostic_emitter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/diagnostics/mocks.h"

namespace Carbon::Testing {
namespace {

struct FakeDiagnosticLocationTranslator : DiagnosticLocationTranslator<int> {
  auto GetLocation(int n) -> DiagnosticLocation override {
    return {.line_number = 1, .column_number = n};
  }
};

class DiagnosticEmitterTest : public ::testing::Test {
 protected:
  DiagnosticEmitterTest() : emitter_(translator_, consumer_) {}

  FakeDiagnosticLocationTranslator translator_;
  Testing::MockDiagnosticConsumer consumer_;
  DiagnosticEmitter<int> emitter_;
};

TEST_F(DiagnosticEmitterTest, EmitSimpleError) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Error, "simple error");
  EXPECT_CALL(consumer_, HandleDiagnostic(IsDiagnostic(
                             DiagnosticKind::TestDiagnostic,
                             DiagnosticLevel::Error, 1, 1, "simple error")));
  EXPECT_CALL(consumer_, HandleDiagnostic(IsDiagnostic(
                             DiagnosticKind::TestDiagnostic,
                             DiagnosticLevel::Error, 1, 2, "simple error")));
  emitter_.Emit(1, TestDiagnostic);
  emitter_.Emit(2, TestDiagnostic);
}

TEST_F(DiagnosticEmitterTest, EmitSimpleWarning) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  EXPECT_CALL(consumer_,
              HandleDiagnostic(IsDiagnostic(DiagnosticKind::TestDiagnostic,
                                            DiagnosticLevel::Warning, 1, 1,
                                            "simple warning")));
  emitter_.Emit(1, TestDiagnostic);
}

TEST_F(DiagnosticEmitterTest, EmitOneArgDiagnostic) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Error, "arg: `{0}`", llvm::StringRef);
  EXPECT_CALL(consumer_, HandleDiagnostic(IsDiagnostic(
                             DiagnosticKind::TestDiagnostic,
                             DiagnosticLevel::Error, 1, 1, "arg: `str`")));
  emitter_.Emit(1, TestDiagnostic, "str");
}

TEST_F(DiagnosticEmitterTest, EmitNote) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  CARBON_DIAGNOSTIC(TestDiagnosticNote, Note, "note");
  EXPECT_CALL(consumer_,
              HandleDiagnostic(IsDiagnostic(DiagnosticKind::TestDiagnostic,
                                            DiagnosticLevel::Warning, 1, 1,
                                            "simple warning")));
  emitter_.Build(1, TestDiagnostic).Note(2, TestDiagnosticNote).Emit();
}

}  // namespace
}  // namespace Carbon::Testing
