// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir_factory.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "toolchain/diagnostics/mocks.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_ir_test_helpers.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::Testing {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

class SemanticsIRFactoryTest : public ::testing::Test {
 protected:
  void Build(llvm::Twine t) {
    source_buffer.emplace(std::move(*SourceBuffer::CreateFromText(t)));
    tokenized_buffer = TokenizedBuffer::Lex(*source_buffer, consumer);
    EXPECT_FALSE(tokenized_buffer->has_errors());
    parse_tree = ParseTree::Parse(*tokenized_buffer, consumer);
    EXPECT_FALSE(parse_tree->has_errors());
    llvm::errs() << *parse_tree;
    g_semantics_ir = SemanticsIRFactory::Build(*parse_tree);
  }

  ~SemanticsIRFactoryTest() override { g_semantics_ir = llvm::None; }

  llvm::Optional<SourceBuffer> source_buffer;
  llvm::Optional<TokenizedBuffer> tokenized_buffer;
  llvm::Optional<ParseTree> parse_tree;
  MockDiagnosticConsumer consumer;
};

TEST_F(SemanticsIRFactoryTest, SimpleProgram) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build(R"(// package FactoryTest api;

           fn Add(x: i32, y: i32) {
             return x + y;
           }

           fn Main() -> i32 {
             var x: i32 = Add(3, 10);
             x *= 5;
             return x;
           }
          )");
}

TEST_F(SemanticsIRFactoryTest, Empty) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build("");
  EXPECT_THAT(g_semantics_ir->root_block(), Block(IsEmpty(), IsEmpty()));
}

TEST_F(SemanticsIRFactoryTest, FunctionBasic) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build("fn Foo() {}");
  EXPECT_THAT(
      g_semantics_ir->root_block(),
      Block(ElementsAre(FunctionName("Foo")),
            UnorderedElementsAre(MappedNode("Foo", FunctionName("Foo")))));
}

TEST_F(SemanticsIRFactoryTest, FunctionDuplicate) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build(R"(fn Foo() {}
           fn Foo() {}
          )");
  EXPECT_THAT(
      g_semantics_ir->root_block(),
      Block(ElementsAre(FunctionName("Foo"), FunctionName("Foo")),
            UnorderedElementsAre(MappedNode("Foo", FunctionName("Foo")))));
}

TEST_F(SemanticsIRFactoryTest, FunctionOrder) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build(R"(fn Foo() {}
           fn Bar() {}
          )");
  EXPECT_THAT(
      g_semantics_ir->root_block(),
      Block(ElementsAre(FunctionName("Bar"), FunctionName("Foo")),
            UnorderedElementsAre(MappedNode("Foo", FunctionName("Foo")),
                                 MappedNode("Bar", FunctionName("Bar")))));
}

TEST_F(SemanticsIRFactoryTest, FunctionBody) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build("fn Foo() { var x: i32 = 0; }");
  EXPECT_THAT(g_semantics_ir->root_block(),
              Block(ElementsAre(FunctionName("Foo")),
                    ElementsAre(MappedNode("Foo", FunctionName("Foo")))));
}

}  // namespace
}  // namespace Carbon::Testing
