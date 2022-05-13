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
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Optional;
using ::testing::UnorderedElementsAre;

class SemanticsIRFactoryTest : public ::testing::Test {
 protected:
  void Build(llvm::Twine t) {
    source_buffer.emplace(std::move(*SourceBuffer::CreateFromText(t)));
    tokenized_buffer = TokenizedBuffer::Lex(*source_buffer, consumer);
    EXPECT_FALSE(tokenized_buffer->has_errors());
    parse_tree = ParseTree::Parse(*tokenized_buffer, consumer);
    EXPECT_FALSE(parse_tree->has_errors());
    SemanticsIRForTest::set_semantics(SemanticsIRFactory::Build(*parse_tree));
  }

  ~SemanticsIRFactoryTest() override { SemanticsIRForTest::clear(); }

  void ExpectRootBlock(
      ::testing::Matcher<llvm::ArrayRef<Semantics::Declaration>> decls,
      ::testing::Matcher<llvm::StringMap<Semantics::Declaration>> name_lookup) {
    EXPECT_THAT(SemanticsIRForTest::semantics().root_block().nodes(), decls);
    EXPECT_THAT(SemanticsIRForTest::semantics().root_block().name_lookup(),
                name_lookup);
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
  ExpectRootBlock(
      ElementsAre(
          Function(
              Eq("Add"),
              ElementsAre(PatternBinding(Eq("x"), ExpressionLiteral("i32")),
                          PatternBinding(Eq("y"), ExpressionLiteral("i32"))),
              Optional(ExpressionLiteral("i32"))),
          Function(Eq("Main"), IsEmpty(), Optional(ExpressionLiteral("i32")))),
      UnorderedElementsAre(MappedNode("Add", FunctionName("Add")),
                           MappedNode("Main", FunctionName("Main"))));
}
*/

TEST_F(SemanticsIRFactoryTest, Empty) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build("");
  ExpectRootBlock(IsEmpty(), IsEmpty());
}

TEST_F(SemanticsIRFactoryTest, FunctionBasic) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build("fn Foo() {}");
  ExpectRootBlock(ElementsAre(Function(Eq("Foo"), IsEmpty(), IsNone(),
                                       StatementBlock(IsEmpty(), IsEmpty()))),
                  UnorderedElementsAre(MappedNode("Foo", FunctionName("Foo"))));
}

TEST_F(SemanticsIRFactoryTest, FunctionParams) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build("fn Foo(x: i32, y: i64) {}");
  ExpectRootBlock(
      ElementsAre(Function(
          Eq("Foo"),
          ElementsAre(PatternBinding(Eq("x"), ExpressionLiteral("i32")),
                      PatternBinding(Eq("y"), ExpressionLiteral("i64"))),
          IsNone(), StatementBlock(IsEmpty(), IsEmpty()))),
      UnorderedElementsAre(MappedNode("Foo", FunctionName("Foo"))));
}

TEST_F(SemanticsIRFactoryTest, FunctionReturnType) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build("fn Foo() -> i32 {}");
  ExpectRootBlock(ElementsAre(Function(Eq("Foo"), IsEmpty(),
                                       Optional(ExpressionLiteral("i32")),
                                       StatementBlock(IsEmpty(), IsEmpty()))),
                  UnorderedElementsAre(MappedNode("Foo", FunctionName("Foo"))));
}

TEST_F(SemanticsIRFactoryTest, FunctionDuplicate) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build(R"(fn Foo() {}
           fn Foo() {}
          )");
  ExpectRootBlock(ElementsAre(FunctionName("Foo"), FunctionName("Foo")),
                  UnorderedElementsAre(MappedNode("Foo", FunctionName("Foo"))));
}

TEST_F(SemanticsIRFactoryTest, FunctionOrder) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build(R"(fn Foo() {}
           fn Bar() {}
          )");
  ExpectRootBlock(ElementsAre(FunctionName("Foo"), FunctionName("Bar")),
                  UnorderedElementsAre(MappedNode("Bar", FunctionName("Bar")),
                                       MappedNode("Foo", FunctionName("Foo"))));
}

TEST_F(SemanticsIRFactoryTest, TrivialReturn) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build(R"(fn Main() {
             return;
           }
          )");
  ExpectRootBlock(
      ElementsAre(
          Function(Eq("Main"), IsEmpty(), IsNone(),
                   StatementBlock(ElementsAre(Return(IsNone())), IsEmpty()))),
      UnorderedElementsAre(MappedNode("Main", FunctionName("Main"))));
}

TEST_F(SemanticsIRFactoryTest, ReturnLiteral) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build(R"(fn Main() {
             return 1;
           }
          )");
  ExpectRootBlock(
      ElementsAre(Function(
          Eq("Main"), IsEmpty(), IsNone(),
          StatementBlock(ElementsAre(Return(Optional(ExpressionLiteral("1")))),
                         IsEmpty()))),
      UnorderedElementsAre(MappedNode("Main", FunctionName("Main"))));
}

/*
TEST_F(SemanticsIRFactoryTest, ReturnArithmetic) {
  EXPECT_CALL(consumer, HandleDiagnostic(_)).Times(0);
  Build(R"(fn Main() -> i32 {
             return 1 + 2;
           }
          )");
  ExpectRootBlock(
      ElementsAre(
          Function(Eq("Main"), IsEmpty(), Optional(ExpressionLiteral("i32")))),
      UnorderedElementsAre(MappedNode("Main", FunctionName("Main"))));
}
*/

}  // namespace
}  // namespace Carbon::Testing
