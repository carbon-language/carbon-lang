// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/parse_and_execute/parse_and_execute.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon::Testing {
namespace {

using ::testing::EndsWith;
using ::testing::Eq;

TEST(ParseAndExecuteTest, Recursion) {
  std::string source = R"(
    package Test api;
    fn Main() -> i32 {
      return
  )";
  // A high depth that's expected to complete in a few seconds.
  static constexpr int Depth = 50000;
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
  TraceStream trace_stream;
  auto err =
      ParseAndExecute("explorer/data/prelude.carbon", "test.carbon", source,
                      /*parser_debug=*/false, &trace_stream, &llvm::nulls());
  ASSERT_FALSE(err.ok());
  EXPECT_THAT(
      err.error().message(),
      EndsWith("stack overflow: too many interpreter actions on stack"));
}

}  // namespace
}  // namespace Carbon::Testing
