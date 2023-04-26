// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/parse_and_execute.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon::Testing {
namespace {

using ::testing::Eq;

class ParseAndExecuteTest : public ::testing::Test {
 protected:
  auto ParseAndExecuteHelper(const std::string& source,
                             testing::Matcher<std::string> err_matcher)
      -> void {
    auto err = ParseAndExecute("explorer/data/prelude.carbon", source);
    ASSERT_FALSE(err.ok());
    EXPECT_THAT(err.error().message(), err_matcher);
  }
};

TEST_F(ParseAndExecuteTest, Recursion) {
  std::string source = R"(
    package Test api;
    fn Main() -> i32 {
      return
  )";
  // A high depth that's expected to complete in ~10s.
  static constexpr int Depth = 2500;
  for (int i = 0; i < Depth; ++i) {
    source += "if true then\n";
  }
  source += "1\n";
  for (int i = 0; i < Depth; ++i) {
    source += "else 0\n";
  }
  source += R"(
        ;
    }
  )";
  ParseAndExecuteHelper(
      source, Eq("stack overflow: too many interpreter actions on stack"));
}

}  // namespace
}  // namespace Carbon::Testing
