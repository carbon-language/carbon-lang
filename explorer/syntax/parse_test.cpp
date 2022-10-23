// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/syntax/parse.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <variant>

#include "explorer/common/arena.h"

namespace Carbon::Testing {
namespace {

static constexpr std::string_view FileContents = R"(
package ExplorerTest api;

fn Foo() {}
)";

TEST(ParseTest, ParseFromString) {
  Arena arena;
  ErrorOr<AST> parse_result = ParseFromString(
      &arena, "file.carbon", FileContents, /*parser_debug=*/false);
  ASSERT_TRUE(parse_result.ok());
  EXPECT_EQ(parse_result->declarations.size(), 1);
}

}  // namespace
}  // namespace Carbon::Testing
