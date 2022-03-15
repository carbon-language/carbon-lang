// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/Optional.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::Testing {
namespace {

class ParseTreeTest : public ::testing::Test {
 protected:
  auto Analyze(llvm::Twine t) -> Semantics {
    source_buffer.emplace(std::move(*SourceBuffer::CreateFromText(t.str())));
    tokenized_buffer = TokenizedBuffer::Lex(*source_buffer, consumer);
    parse_tree = ParseTree::Parse(*tokenized_buffer, consumer);
    return Semantics::Analyze(*parse_tree, consumer);
  }

  llvm::Optional<SourceBuffer> source_buffer;
  llvm::Optional<TokenizedBuffer> tokenized_buffer;
  llvm::Optional<ParseTree> parse_tree;
  DiagnosticConsumer& consumer = ConsoleDiagnosticConsumer();
};

TEST_F(ParseTreeTest, Empty) {
  // TODO: Validate the returned Semantics object.
  Analyze("");
  ASSERT_FALSE(parse_tree->has_errors());
}

}  // namespace
}  // namespace Carbon::Testing
