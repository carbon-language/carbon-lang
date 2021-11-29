// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "executable_semantics/ast/ast_test_matchers.h"
#include "executable_semantics/syntax/parse.h"
#include "executable_semantics/syntax/parse_test_matchers.h"

namespace Carbon {
namespace {

using ::testing::ElementsAre;

TEST(UnimplementedExampleTest, VerifyPrecedence) {
  static constexpr std::string_view Program = R"(
    package ExecutableSemanticsTest api;
    fn Main() -> i32 {
      return 1 __unimplemented_example_infix 2 + 3;
    }
  )";
  Arena arena;
  EXPECT_THAT(ParseFromString(&arena, "dummy.carbon", Program, false),
              ParsedAs(ASTDeclarations(
                  ElementsAre(MatchesFunctionDeclaration().WithBody(
                      BlockContentsAre(ElementsAre(MatchesReturn(MatchesAdd(
                          MatchesUnimplementedExpression(
                              "ExampleInfix", ElementsAre(MatchesLiteral(1),
                                                          MatchesLiteral(2))),
                          MatchesLiteral(3))))))))));
}

}  // namespace
}  // namespace Carbon
