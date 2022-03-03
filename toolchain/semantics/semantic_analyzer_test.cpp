// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantic_analyzer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>

#include "toolchain/diagnostics/mocks.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::Testing {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::Eq;

class SemanticAnalyzerTest : public ::testing::Test {
 protected:
  auto Analyze(llvm::Twine t) -> Semantics {
    source_buffer.emplace(std::move(*SourceBuffer::CreateFromText(t.str())));
    tokenized_buffer = TokenizedBuffer::Lex(*source_buffer, consumer);
    EXPECT_FALSE(tokenized_buffer->HasErrors());
    parse_tree = ParseTree::Parse(*tokenized_buffer, consumer);
    EXPECT_FALSE(parse_tree->HasErrors());
    return SemanticAnalyzer::Analyze(*parse_tree, consumer);
  }

  std::optional<SourceBuffer> source_buffer;
  std::optional<TokenizedBuffer> tokenized_buffer;
  std::optional<ParseTree> parse_tree;
  MockDiagnosticConsumer consumer;
};

TEST_F(SemanticAnalyzerTest, Empty) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Analyze("");
}

TEST_F(SemanticAnalyzerTest, FunctionBasic) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Analyze("fn Foo() {}");
}

TEST_F(SemanticAnalyzerTest, FunctionDuplicate) {
  EXPECT_CALL(
      consumer,
      HandleDiagnostic(AllOf(
          DiagnosticAt(1, 4),
          DiagnosticMessage(Eq(
              "Name conflict for `Foo`; previously declared at /text:2:17.")))))
      .Times(1);
  Analyze(R"(fn Foo() {}
             fn Foo() {}
            )");
}

}  // namespace
}  // namespace Carbon::Testing
