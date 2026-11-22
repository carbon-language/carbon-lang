// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/emitter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

#include "llvm/ADT/StringRef.h"
#include "toolchain/diagnostics/mocks.h"

namespace Carbon::Testing {
namespace {

using testing::_;
using testing::ElementsAre;

class FakeEmitter : public Diagnostics::Emitter<int> {
 public:
  using Emitter::Emitter;

 protected:
  auto ConvertLoc(int n, ContextFnT /*context_fn*/) const
      -> Diagnostics::ConvertedLoc override {
    return {.loc = {.line_number = 1, .column_number = n},
            .last_byte_offset = -1};
  }
};

class EmitterTest : public ::testing::Test {
 public:
  EmitterTest() : emitter_(&consumer_) {}

  Testing::MockDiagnosticConsumer consumer_;
  FakeEmitter emitter_;
};

TEST_F(EmitterTest, EmitSimpleError) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Error, "simple error");
  EXPECT_CALL(consumer_, HandleDiagnostic(IsSingleDiagnostic(
                             Diagnostics::Kind::TestDiagnostic,
                             Diagnostics::Level::Error, 1, 1, "simple error")));
  EXPECT_CALL(consumer_, HandleDiagnostic(IsSingleDiagnostic(
                             Diagnostics::Kind::TestDiagnostic,
                             Diagnostics::Level::Error, 1, 2, "simple error")));
  emitter_.Emit(1, TestDiagnostic);
  emitter_.Emit(2, TestDiagnostic);
}

TEST_F(EmitterTest, EmitSimpleWarning) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  EXPECT_CALL(consumer_,
              HandleDiagnostic(IsSingleDiagnostic(
                  Diagnostics::Kind::TestDiagnostic,
                  Diagnostics::Level::Warning, 1, 1, "simple warning")));
  emitter_.Emit(1, TestDiagnostic);
}

TEST_F(EmitterTest, EmitOneArgDiagnostic) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Error, "arg: `{0}`", std::string);
  EXPECT_CALL(consumer_, HandleDiagnostic(IsSingleDiagnostic(
                             Diagnostics::Kind::TestDiagnostic,
                             Diagnostics::Level::Error, 1, 1, "arg: `str`")));
  emitter_.Emit(1, TestDiagnostic, "str");
}

TEST_F(EmitterTest, AttachLabel) {
  // The label has arguments and a format of its own, both validated.
  CARBON_DIAGNOSTIC(TestDiagnostic, Error, "expected {0}, found {1}",
                    std::string, std::string);
  CARBON_DIAGNOSTIC_LABEL(TestLabel, "this is {0}", std::string);
  EXPECT_CALL(consumer_,
              HandleDiagnostic(IsDiagnostic(
                  Diagnostics::Level::Error,
                  IsDiagnosticMessage(Diagnostics::Kind::TestDiagnostic,
                                      Diagnostics::Level::Error, 1, 1,
                                      "expected i32, found String"),
                  ElementsAre(IsDiagnosticLabel(1, 2, "this is String")))));
  emitter_.Build(1, TestDiagnostic, "i32", "String")
      .Attach(2, TestLabel, "String")
      .Emit();
}

TEST_F(EmitterTest, ExtraRange) {
  // `ExtraRange` adds to the message's ranges and attaches no label.
  CARBON_DIAGNOSTIC(TestDiagnostic, Error, "simple error");
  EXPECT_CALL(
      consumer_,
      HandleDiagnostic(IsDiagnosticWithRange(
          Diagnostics::Kind::TestDiagnostic, Diagnostics::Level::Error,
          {.line_number = 1, .column_number = 1, .length = _}, "simple error",
          {.line_number = 1, .column_number = 2, .length = _})));
  emitter_.Build(1, TestDiagnostic).ExtraRange(2).Emit();
}

TEST_F(EmitterTest, AttachInfo) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  CARBON_DIAGNOSTIC_LABEL(TestInfo, "note");
  EXPECT_CALL(consumer_,
              HandleDiagnostic(IsDiagnostic(
                  Diagnostics::Level::Warning,
                  IsDiagnosticMessage(Diagnostics::Kind::TestDiagnostic,
                                      Diagnostics::Level::Warning, 1, 1,
                                      "simple warning"),
                  ElementsAre(IsDiagnosticLabel(1, 2, "note")))));
  emitter_.Build(1, TestDiagnostic).Attach(2, TestInfo).Emit();
}

TEST_F(EmitterTest, ContextScopesAttachLabelsInnermostFirst) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  CARBON_DIAGNOSTIC_LABEL(TestLabel, "in {0}", std::string);
  EXPECT_CALL(consumer_,
              HandleDiagnostic(IsDiagnostic(
                  Diagnostics::Level::Warning,
                  IsDiagnosticMessage(Diagnostics::Kind::TestDiagnostic,
                                      Diagnostics::Level::Warning, 1, 1,
                                      "simple warning"),
                  ElementsAre(IsDiagnosticLabel(1, 2, "in inner"),
                              IsDiagnosticLabel(1, 3, "in outer")))));
  Diagnostics::ContextScope outer(
      &emitter_, [&](auto& builder) { builder.Attach(3, TestLabel, "outer"); });
  Diagnostics::ContextScope inner(
      &emitter_, [&](auto& builder) { builder.Attach(2, TestLabel, "inner"); });
  emitter_.Emit(1, TestDiagnostic);
}

TEST_F(EmitterTest, Flush) {
  bool flushed = false;
  auto flush_fn = [&]() { flushed = true; };

  {
    FakeEmitter emitter(&consumer_);
    emitter.AddFlushFn(flush_fn);

    // Registering the function does not flush.
    EXPECT_FALSE(flushed);

    // Explicit calls to `Flush` should flush.
    emitter.Flush();
    EXPECT_TRUE(flushed);
    flushed = false;

    {
      Diagnostics::ContextScope scope(&emitter, [](auto&) {});

      // Entering a context scope should flush.
      EXPECT_TRUE(flushed);
      flushed = false;
    }

    // Leaving a context scope should flush.
    EXPECT_TRUE(flushed);
    flushed = false;
  }

  // Destroying the emitter should not flush, as that could call back into the
  // base class emitter after the derived-class emitter has been destroyed.
  EXPECT_FALSE(flushed);
}

}  // namespace
}  // namespace Carbon::Testing
