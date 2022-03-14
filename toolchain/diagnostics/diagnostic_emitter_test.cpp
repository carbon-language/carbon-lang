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
  DIAGNOSTIC(TestDiagnostic, Error, "simple error");
  EXPECT_CALL(consumer_, HandleDiagnostic(IsDiagnostic(
                             DiagnosticKind::TestDiagnostic(),
                             DiagnosticLevel::Error, 1, 1, "simple error")));
  EXPECT_CALL(consumer_, HandleDiagnostic(IsDiagnostic(
                             DiagnosticKind::TestDiagnostic(),
                             DiagnosticLevel::Error, 1, 2, "simple error")));
  emitter_.Emit(1, TestDiagnostic);
  emitter_.Emit(2, TestDiagnostic);
}

TEST_F(DiagnosticEmitterTest, EmitSimpleWarning) {
  DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  EXPECT_CALL(consumer_,
              HandleDiagnostic(IsDiagnostic(DiagnosticKind::TestDiagnostic(),
                                            DiagnosticLevel::Warning, 1, 1,
                                            "simple warning")));
  emitter_.Emit(1, TestDiagnostic);
}

TEST_F(DiagnosticEmitterTest, EmitOneArgDiagnostic) {
  DIAGNOSTIC(TestDiagnostic, Error, "arg: `{0}`", llvm::StringRef);
  EXPECT_CALL(consumer_, HandleDiagnostic(IsDiagnostic(
                             DiagnosticKind::TestDiagnostic(),
                             DiagnosticLevel::Error, 1, 1, "arg: `str`")));
  emitter_.Emit(1, TestDiagnostic, "str");
}

TEST_F(DiagnosticEmitterTest, EmitCustomFormat) {
  DIAGNOSTIC_WITH_FORMAT_FN(
      TestDiagnostic, Error, "unused",
      [](llvm::StringLiteral) -> std::string { return "custom format"; });
  EXPECT_CALL(consumer_, HandleDiagnostic(IsDiagnostic(
                             DiagnosticKind::TestDiagnostic(),
                             DiagnosticLevel::Error, 1, 1, "custom format")));
  emitter_.Emit(1, TestDiagnostic);
}

}  // namespace
}  // namespace Carbon::Testing
