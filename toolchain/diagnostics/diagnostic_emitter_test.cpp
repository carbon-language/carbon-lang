// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/diagnostic_emitter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/StringRef.h"
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
  TestDiagnostic.Emit(emitter_, 1);
  TestDiagnostic.Emit(emitter_, 2);
}

TEST_F(DiagnosticEmitterTest, EmitSimpleWarning) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  EXPECT_CALL(consumer_,
              HandleDiagnostic(IsDiagnostic(DiagnosticKind::TestDiagnostic,
                                            DiagnosticLevel::Warning, 1, 1,
                                            "simple warning")));
  TestDiagnostic.Emit(emitter_, 1);
}

TEST_F(DiagnosticEmitterTest, EmitOneArgDiagnostic) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Error, "arg: `{0}`", llvm::StringRef);
  EXPECT_CALL(consumer_, HandleDiagnostic(IsDiagnostic(
                             DiagnosticKind::TestDiagnostic,
                             DiagnosticLevel::Error, 1, 1, "arg: `str`")));
  TestDiagnostic.Emit(emitter_, 1, "str");
}

TEST_F(DiagnosticEmitterTest, EmitNote) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  CARBON_DIAGNOSTIC(TestDiagnosticNote, Note, "note");
  EXPECT_CALL(consumer_,
              HandleDiagnostic(IsDiagnostic(DiagnosticKind::TestDiagnostic,
                                            DiagnosticLevel::Warning, 1, 1,
                                            "simple warning")));
  auto diag = TestDiagnostic.Build(emitter_, 1);
  TestDiagnosticNote.Note(diag, 2);
  diag.Emit();
}

TEST_F(DiagnosticEmitterTest, EmitStringByRef) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "{0}", std::string);
  EXPECT_CALL(consumer_,
              HandleDiagnostic(IsDiagnostic(DiagnosticKind::TestDiagnostic,
                                            DiagnosticLevel::Warning, 1, 1,
                                            "simple warning")));
  std::string s = "simple warning";
  auto ref = [&]() -> std::string& { return s; };
  TestDiagnostic.Emit(emitter_, 1, ref());
}

TEST_F(DiagnosticEmitterTest, EmitString) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "{0}", std::string);
  EXPECT_CALL(consumer_,
              HandleDiagnostic(IsDiagnostic(DiagnosticKind::TestDiagnostic,
                                            DiagnosticLevel::Warning, 1, 1,
                                            "simple warning")));
  auto ref = [&]() -> std::string { return "simple warning"; };
  TestDiagnostic.Emit(emitter_, 1, ref());
}

}  // namespace
}  // namespace Carbon::Testing
