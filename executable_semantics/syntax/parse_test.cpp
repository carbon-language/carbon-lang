// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/parse.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <variant>

#include "executable_semantics/common/arena.h"

namespace Carbon {
namespace {

static constexpr std::string_view FileContents = R"(
package ExecutableSemanticsTest api;

fn Foo() {}
)";

TEST(ParseTest, ParseFromString) {
  Arena arena;
  std::variant<AST, SyntaxErrorCode> parse_result =
      ParseFromString(&arena, "file.carbon", FileContents, /*trace=*/false);
  ASSERT_TRUE(std::holds_alternative<AST>(parse_result));
  EXPECT_EQ(std::get<AST>(parse_result).declarations.size(), 1);
}

}  // namespace
}  // namespace Carbon
