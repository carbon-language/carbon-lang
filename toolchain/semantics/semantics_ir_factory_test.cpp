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
using ::testing::Optional;
using ::testing::StrEq;

class SemanticsIRFactoryTest : public ::testing::Test {
 protected:
  void Build(llvm::Twine t) {
    source_buffer.emplace(std::move(*SourceBuffer::CreateFromText(t)));
    tokenized_buffer = TokenizedBuffer::Lex(*source_buffer, consumer);
    EXPECT_FALSE(tokenized_buffer->has_errors());
    parse_tree = ParseTree::Parse(*tokenized_buffer, consumer);
    EXPECT_FALSE(parse_tree->has_errors());
    SemanticsIRForTest::set_semantics(
        SemanticsIRFactory::Build(*tokenized_buffer, *parse_tree));
  }

  ~SemanticsIRFactoryTest() override { SemanticsIRForTest::clear(); }

  auto root_block() const -> llvm::ArrayRef<Semantics::NodeRef> {
    return SemanticsIRForTest::semantics().root_block();
  }

  llvm::Optional<SourceBuffer> source_buffer;
  llvm::Optional<TokenizedBuffer> tokenized_buffer;
  llvm::Optional<ParseTree> parse_tree;
  MockDiagnosticConsumer consumer;
};

/*
TEST_F(SemanticsIRFactoryTest, SimpleProgram) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build(R"(// package FactoryTest api;

           fn Add(x: i32, y: i32) -> i32 {
             return x + y;
           }

           fn Main() -> i32 {
             var x: i32 = Add(3, 10);
             x *= 5;
             return x;
           }
          )");
  EXPECT_THAT(
      SemanticsIRForTest::semantics().root_block(),
      ElementsAre(Function(Eq("Add"),
                           ElementsAre(PatternBinding(Eq("x"), Literal("i32")),
                                       PatternBinding(Eq("y"), Literal("i32"))),
                           Optional(Literal("i32"))),
                  Function(Eq("Main"), IsEmpty(), Optional(Literal("i32")))));
}
*/

TEST_F(SemanticsIRFactoryTest, Empty) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build("");
  EXPECT_THAT(root_block(), IsEmpty());
}

TEST_F(SemanticsIRFactoryTest, FunctionBasic) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build("fn Foo() {}");
  EXPECT_THAT(root_block(),
              ElementsAre(Function(0, IsEmpty()), SetName(StrEq("Foo"), 0)));
}

/*
TEST_F(SemanticsIRFactoryTest, FunctionParams) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build("fn Foo(x: i32, y: i64) {}");
  ExpectRootBlock(
      ElementsAre(Function(Eq("Foo"),
                           ElementsAre(PatternBinding(Eq("x"), Literal("i32")),
                                       PatternBinding(Eq("y"), Literal("i64"))),
                           IsNone(), StatementBlock(IsEmpty(), IsEmpty()))),
      UnorderedElementsAre(MappedNode("Foo", FunctionName("Foo"))));
}
*/

/*
TEST_F(SemanticsIRFactoryTest, FunctionReturnType) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build("fn Foo() -> i32 {}");
  EXPECT_THAT(root_block(), ElementsAre(Function(0, IsEmpty()),
                                        SetName(StrEq("Foo"), 0)));
}
*/

TEST_F(SemanticsIRFactoryTest, FunctionOrder) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build(R"(fn Foo() {}
           fn Bar() {}
           fn Bar() {}
          )");
  EXPECT_THAT(root_block(),
              ElementsAre(Function(2, IsEmpty()), SetName(StrEq("Foo"), 2),
                          Function(1, IsEmpty()), SetName(StrEq("Bar"), 1),
                          Function(0, IsEmpty()), SetName(StrEq("Bar"), 0)));
}

TEST_F(SemanticsIRFactoryTest, TrivialReturn) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build(R"(fn Main() {
             return;
           }
          )");
  EXPECT_THAT(root_block(),
              ElementsAre(Function(0, ElementsAre(Return(IsNone()))),
                          SetName(StrEq("Main"), 0)));
}

TEST_F(SemanticsIRFactoryTest, ReturnLiteral) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build(R"(fn Main() {
             return 12;
           }
          )");
  EXPECT_THAT(root_block(),
              ElementsAre(Function(0, ElementsAre(IntegerLiteral(1, 12),
                                                  Return(Optional(1)))),
                          SetName(StrEq("Main"), 0)));
}

TEST_F(SemanticsIRFactoryTest, ReturnArithmetic) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build(R"(fn Main() {
             return 12 + 34;
           }
          )");
  EXPECT_THAT(
      root_block(),
      ElementsAre(
          Function(0,
                   ElementsAre(IntegerLiteral(3, 12), IntegerLiteral(2, 34),
                               BinaryOperator(
                                   1, Semantics::BinaryOperator::Op::Add, 3, 2),
                               Return(Optional(1)))),
          SetName(StrEq("Main"), 0)));
}

}  // namespace
}  // namespace Carbon::Testing
