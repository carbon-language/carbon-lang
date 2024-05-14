// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "explorer/ast/ast_test_matchers.h"
#include "explorer/syntax/parse.h"
#include "explorer/syntax/parse_test_matchers.h"

namespace Carbon {
namespace {

using ::testing::ElementsAre;

TEST(UnimplementedExampleTest, VerifyPrecedence) {
  static constexpr std::string_view Program = R"(
    package ExplorerTest api;
    fn Main() -> i32 {
      return 1 __unimplemented_example_infix 2 == 3;
    }
  )";
  Arena arena;
  EXPECT_THAT(
      ParseFromString(&arena, "dummy.carbon", FileKind::Main, Program, false),
      ParsedAs(
          ASTDeclarations(ElementsAre(MatchesFunctionDeclaration().WithBody(
              BlockContentsAre(ElementsAre(MatchesReturn(MatchesEq(
                  MatchesUnimplementedExpression(
                      "ExampleInfix",
                      ElementsAre(MatchesLiteral(1), MatchesLiteral(2))),
                  MatchesLiteral(3))))))))));
}

}  // namespace
}  // namespace Carbon
