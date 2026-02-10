// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/sorting_consumer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/StringRef.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/diagnostics/mocks.h"

namespace Carbon::Diagnostics {
namespace {

using ::Carbon::Testing::IsSingleDiagnostic;
using ::testing::_;
using ::testing::ElementsAreArray;

CARBON_DIAGNOSTIC(TestDiagnostic, Error, "Diag{0}", int);
CARBON_DIAGNOSTIC_ON_SCOPE(TestDiagnosticOnScope, Error, "DiagOnScope{0}", int);

// Sorting-related locations, consumed by `FakeEmitter::ConvertLoc`.
struct TestLoc {
  int32_t last_byte_offset;
  int32_t line_number;
  int32_t column_number;
};

// Emits a diagnostic with the requested location.
class FakeEmitter : public Emitter<TestLoc> {
 public:
  using Emitter::Emitter;

 protected:
  auto ConvertLoc(TestLoc test_loc, ContextFnT /*context_fn*/) const
      -> ConvertedLoc override {
    return {.loc = {.line_number = test_loc.line_number,
                    .column_number = test_loc.column_number},
            .last_byte_offset = test_loc.last_byte_offset};
  }
};

// Stores diagnostics in a vector for `ElementsArray` matching.
class VectorConsumer : public Consumer {
 public:
  auto HandleDiagnostic(Diagnostic diagnostic) -> void override {
    diagnostics_.push_back(std::move(diagnostic));
  }

  auto Flush() -> void override {}

  auto diagnostics() const -> llvm::ArrayRef<Diagnostic> {
    return diagnostics_;
  }

 private:
  llvm::SmallVector<Diagnostic> diagnostics_;
};

TEST(SortedEmitterTest, SortErrors) {
  VectorConsumer consumer;
  SortingConsumer sorting_consumer(consumer);
  FakeEmitter emitter(&sorting_consumer);

  emitter.Emit({1, 10, 1}, TestDiagnostic, 1);
  emitter.Emit({-1, 11, 1}, TestDiagnostic, 2);
  emitter.Emit({0, 12, 1}, TestDiagnostic, 3);
  emitter.Emit({4, 13, 1}, TestDiagnostic, 4);
  emitter.Emit({3, 14, 1}, TestDiagnostic, 5);
  emitter.Emit({3, 15, 1}, TestDiagnostic, 6);

  EXPECT_TRUE(consumer.diagnostics().empty());
  sorting_consumer.Flush();
  EXPECT_THAT(
      consumer.diagnostics(),
      ElementsAreArray({
          IsSingleDiagnostic(Kind::TestDiagnostic, Level::Error, _, _, "Diag2"),
          IsSingleDiagnostic(Kind::TestDiagnostic, Level::Error, _, _, "Diag3"),
          IsSingleDiagnostic(Kind::TestDiagnostic, Level::Error, _, _, "Diag1"),
          IsSingleDiagnostic(Kind::TestDiagnostic, Level::Error, _, _, "Diag5"),
          IsSingleDiagnostic(Kind::TestDiagnostic, Level::Error, _, _, "Diag6"),
          IsSingleDiagnostic(Kind::TestDiagnostic, Level::Error, _, _, "Diag4"),
      }));
}

TEST(SortedEmitterTest, SortOnScope) {
  VectorConsumer consumer;
  SortingConsumer sorting_consumer(consumer);
  FakeEmitter emitter(&sorting_consumer);

  emitter.Emit({2, 15, 1}, TestDiagnosticOnScope, 1);
  emitter.Emit({4, 1, 1}, TestDiagnosticOnScope, 2);
  // Same `last_byte_offset`, should sort by line/column.
  emitter.Emit({3, 10, 5}, TestDiagnosticOnScope, 3);
  emitter.Emit({3, 10, 1}, TestDiagnosticOnScope, 4);
  emitter.Emit({3, 5, 20}, TestDiagnosticOnScope, 5);
  emitter.Emit({3, 10, 3}, TestDiagnosticOnScope, 6);

  EXPECT_TRUE(consumer.diagnostics().empty());
  sorting_consumer.Flush();

  EXPECT_THAT(consumer.diagnostics(),
              ElementsAreArray({
                  IsSingleDiagnostic(Kind::TestDiagnosticOnScope, Level::Error,
                                     _, _, "DiagOnScope1"),
                  IsSingleDiagnostic(Kind::TestDiagnosticOnScope, Level::Error,
                                     _, _, "DiagOnScope5"),
                  IsSingleDiagnostic(Kind::TestDiagnosticOnScope, Level::Error,
                                     _, _, "DiagOnScope4"),
                  IsSingleDiagnostic(Kind::TestDiagnosticOnScope, Level::Error,
                                     _, _, "DiagOnScope6"),
                  IsSingleDiagnostic(Kind::TestDiagnosticOnScope, Level::Error,
                                     _, _, "DiagOnScope3"),
                  IsSingleDiagnostic(Kind::TestDiagnosticOnScope, Level::Error,
                                     _, _, "DiagOnScope2"),
              }));
}

TEST(SortedEmitterTest, MixedScope) {
  VectorConsumer consumer;
  SortingConsumer sorting_consumer(consumer);
  FakeEmitter emitter(&sorting_consumer);

  // Difference in on-scope means only `last_byte_offset` is used.
  emitter.Emit({3, 3, 1}, TestDiagnosticOnScope, 1);
  emitter.Emit({3, 10, 1}, TestDiagnostic, 2);

  EXPECT_TRUE(consumer.diagnostics().empty());
  sorting_consumer.Flush();

  EXPECT_THAT(
      consumer.diagnostics(),
      ElementsAreArray({
          IsSingleDiagnostic(Kind::TestDiagnostic, Level::Error, _, _, "Diag2"),
          IsSingleDiagnostic(Kind::TestDiagnosticOnScope, Level::Error, _, _,
                             "DiagOnScope1"),
      }));
}

}  // namespace
}  // namespace Carbon::Diagnostics
