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

using testing::ElementsAre;
using testing::IsEmpty;

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
  CARBON_DIAGNOSTIC_LABEL(TestLabel, Primary, "this is {0}", std::string);
  EXPECT_CALL(
      consumer_,
      HandleDiagnostic(IsDiagnostic(
          Diagnostics::Level::Error,
          IsDiagnosticMessage(Diagnostics::Kind::TestDiagnostic,
                              Diagnostics::Level::Error, 1, 1,
                              "expected i32, found String"),
          IsEmpty(),
          ElementsAre(IsDiagnosticLabel(Diagnostics::LabelCategory::Primary, 1,
                                        2, "this is String")))));
  emitter_.Build(1, TestDiagnostic, "i32", "String")
      .Attach(2, TestLabel, "String")
      .Emit();
}

TEST_F(EmitterTest, AttachRangeWithNoText) {
  // A label with nothing to say marks its range and no more, which is how a
  // diagnostic points at the code its message is about.
  CARBON_DIAGNOSTIC(TestDiagnostic, Error, "simple error");
  EXPECT_CALL(
      consumer_,
      HandleDiagnostic(IsDiagnostic(
          Diagnostics::Level::Error,
          IsDiagnosticMessage(Diagnostics::Kind::TestDiagnostic,
                              Diagnostics::Level::Error, 1, 1, "simple error"),
          IsEmpty(),
          ElementsAre(IsDiagnosticLabel(Diagnostics::LabelCategory::Primary, 1,
                                        2, "")))));
  emitter_.Build(1, TestDiagnostic).Attach(2).Emit();
}

TEST_F(EmitterTest, AttachRangeWithNoTextAsInfo) {
  // A range that only explains the problem can say nothing about itself too,
  // which is what the category on the wordless `Attach` is for: there is no
  // phrase to declare, so there is nowhere else to say what the range is.
  CARBON_DIAGNOSTIC(TestDiagnostic, Error, "simple error");
  EXPECT_CALL(
      consumer_,
      HandleDiagnostic(IsDiagnostic(
          Diagnostics::Level::Error,
          IsDiagnosticMessage(Diagnostics::Kind::TestDiagnostic,
                              Diagnostics::Level::Error, 1, 1, "simple error"),
          IsEmpty(),
          ElementsAre(
              IsDiagnosticLabel(Diagnostics::LabelCategory::Info, 1, 2, "")))));
  emitter_.Build(1, TestDiagnostic)
      .Attach(2, Diagnostics::LabelCategory::Info)
      .Emit();
}

TEST_F(EmitterTest, AttachInfo) {
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  CARBON_DIAGNOSTIC_LABEL(TestInfo, Info, "note");
  EXPECT_CALL(consumer_,
              HandleDiagnostic(IsDiagnostic(
                  Diagnostics::Level::Warning,
                  IsDiagnosticMessage(Diagnostics::Kind::TestDiagnostic,
                                      Diagnostics::Level::Warning, 1, 1,
                                      "simple warning"),
                  IsEmpty(),
                  ElementsAre(IsDiagnosticLabel(
                      Diagnostics::LabelCategory::Info, 1, 2, "note")))));
  emitter_.Build(1, TestDiagnostic).Attach(2, TestInfo).Emit();
}

TEST_F(EmitterTest, EmitContext) {
  CARBON_DIAGNOSTIC_CONTEXT(TestContext, "context");
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  EXPECT_CALL(
      consumer_,
      HandleDiagnostic(IsDiagnostic(
          Diagnostics::Level::Warning,
          IsDiagnosticMessage(Diagnostics::Kind::TestDiagnostic,
                              Diagnostics::Level::Warning, 1, 1,
                              "simple warning"),
          ElementsAre(IsDiagnosticContext(1, 2, "context")), IsEmpty())));
  Diagnostics::ContextScope scope(
      &emitter_, [&](auto& builder) { builder.Attach(2, TestContext); });
  emitter_.Emit(1, TestDiagnostic);
}

TEST_F(EmitterTest, EmitSoftContext) {
  CARBON_DIAGNOSTIC_SOFT_CONTEXT(TestSoftContext, "soft context");
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  EXPECT_CALL(
      consumer_,
      HandleDiagnostic(IsDiagnostic(
          Diagnostics::Level::Warning,
          IsDiagnosticMessage(Diagnostics::Kind::TestDiagnostic,
                              Diagnostics::Level::Warning, 1, 1,
                              "simple warning"),
          ElementsAre(IsDiagnosticContext(1, 2, "soft context")), IsEmpty())));
  Diagnostics::ContextScope soft_scope(
      &emitter_, [&](auto& builder) { builder.Attach(2, TestSoftContext); });
  emitter_.Emit(1, TestDiagnostic);
}

TEST_F(EmitterTest, EmitSoftContextAndContext) {
  CARBON_DIAGNOSTIC_SOFT_CONTEXT(TestSoftContext, "soft context");
  CARBON_DIAGNOSTIC_CONTEXT(TestContext, "context");
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  EXPECT_CALL(consumer_,
              HandleDiagnostic(IsDiagnostic(
                  Diagnostics::Level::Warning,
                  IsDiagnosticMessage(Diagnostics::Kind::TestDiagnostic,
                                      Diagnostics::Level::Warning, 1, 1,
                                      "simple warning"),
                  ElementsAre(IsDiagnosticContext(1, 3, "soft context"),
                              IsDiagnosticContext(1, 2, "context")),
                  IsEmpty())));
  Diagnostics::ContextScope soft_scope(
      &emitter_, [&](auto& builder) { builder.Attach(3, TestSoftContext); });
  Diagnostics::ContextScope scope(
      &emitter_, [&](auto& builder) { builder.Attach(2, TestContext); });
  emitter_.Emit(1, TestDiagnostic);
}

TEST_F(EmitterTest, EmitContextAndSoftContext) {
  CARBON_DIAGNOSTIC_CONTEXT(TestContext, "context");
  CARBON_DIAGNOSTIC_SOFT_CONTEXT(TestSoftContext, "soft context");
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  EXPECT_CALL(
      consumer_,
      HandleDiagnostic(IsDiagnostic(
          Diagnostics::Level::Warning,
          IsDiagnosticMessage(Diagnostics::Kind::TestDiagnostic,
                              Diagnostics::Level::Warning, 1, 1,
                              "simple warning"),
          ElementsAre(IsDiagnosticContext(1, 3, "context")), IsEmpty())));
  Diagnostics::ContextScope scope(
      &emitter_, [&](auto& builder) { builder.Attach(3, TestContext); });
  // This soft context is dropped, since the earlier context supersedes it.
  Diagnostics::ContextScope soft_scope(
      &emitter_, [&](auto& builder) { builder.Attach(2, TestSoftContext); });
  emitter_.Emit(1, TestDiagnostic);
}

TEST_F(EmitterTest, EmitTwoContext) {
  CARBON_DIAGNOSTIC_CONTEXT(TestContext, "context");
  CARBON_DIAGNOSTIC_CONTEXT(TestContext2, "context 2");
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  EXPECT_CALL(consumer_,
              HandleDiagnostic(IsDiagnostic(
                  Diagnostics::Level::Warning,
                  IsDiagnosticMessage(Diagnostics::Kind::TestDiagnostic,
                                      Diagnostics::Level::Warning, 1, 1,
                                      "simple warning"),
                  ElementsAre(IsDiagnosticContext(1, 3, "context"),
                              IsDiagnosticContext(1, 2, "context 2")),
                  IsEmpty())));
  Diagnostics::ContextScope scope(
      &emitter_, [&](auto& builder) { builder.Attach(3, TestContext); });
  Diagnostics::ContextScope scope2(
      &emitter_, [&](auto& builder) { builder.Attach(2, TestContext2); });
  emitter_.Emit(1, TestDiagnostic);
}

TEST_F(EmitterTest, EmitTwoSoftContext) {
  CARBON_DIAGNOSTIC_SOFT_CONTEXT(TestSoftContext, "soft context");
  CARBON_DIAGNOSTIC_SOFT_CONTEXT(TestSoftContext2, "soft context 2");
  CARBON_DIAGNOSTIC(TestDiagnostic, Warning, "simple warning");
  EXPECT_CALL(
      consumer_,
      HandleDiagnostic(IsDiagnostic(
          Diagnostics::Level::Warning,
          IsDiagnosticMessage(Diagnostics::Kind::TestDiagnostic,
                              Diagnostics::Level::Warning, 1, 1,
                              "simple warning"),
          ElementsAre(IsDiagnosticContext(1, 3, "soft context")), IsEmpty())));
  Diagnostics::ContextScope scope(
      &emitter_, [&](auto& builder) { builder.Attach(3, TestSoftContext); });
  // This soft context is dropped, since the earlier one supersedes it.
  Diagnostics::ContextScope soft_scope(
      &emitter_, [&](auto& builder) { builder.Attach(2, TestSoftContext2); });
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
      Diagnostics::AnnotationScope annot(&emitter, [](auto&) {});

      // Registering an annotation scope should flush.
      EXPECT_TRUE(flushed);
      flushed = false;
    }

    // Unregistering an annotation scope should flush.
    EXPECT_TRUE(flushed);
    flushed = false;
  }

  // Destroying the emitter should not flush, as that could call back into the
  // base class emitter after the derived-class emitter has been destroyed.
  EXPECT_FALSE(flushed);
}

}  // namespace
}  // namespace Carbon::Testing
