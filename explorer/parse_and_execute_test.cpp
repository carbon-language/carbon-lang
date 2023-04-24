// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/check.h"
#include "common/error.h"
#include "explorer/interpreter/exec_program.h"
#include "explorer/interpreter/trace_stream.h"
#include "explorer/syntax/parse.h"
#include "explorer/syntax/prelude.h"
#include "gtest/gtest.h"

namespace Carbon::Testing {
namespace {

class ParseAndExecuteTest : public ::testing::Test {
 protected:
  auto ParseAndExecute(const std::string& source,
                       std::optional<testing::Matcher<std::string>> err_matcher)
      -> void {
    auto err = ParseAndExecuteImpl(source);
    if (err_matcher) {
      ASSERT_FALSE(err.ok());
      EXPECT_THAT(err.error().message(), *err_matcher);
    } else {
      EXPECT_TRUE(err.ok()) << err.error();
    }
  }

  auto ParseAndExecuteImpl(const std::string& source) -> ErrorOr<int> {
    Arena arena;
    CARBON_ASSIGN_OR_RETURN(AST ast,
                            ParseFromString(&arena, "test.carbon", source,
                                            /*parser_debug=*/false));

    AddPrelude("explorer/data/prelude.carbon", &arena, &ast.declarations,
               &ast.num_prelude_declarations);
    TraceStream trace_stream;

    // Use llvm::nulls() to suppress output from the Print intrinsic.
    CARBON_ASSIGN_OR_RETURN(
        ast, AnalyzeProgram(&arena, ast, &trace_stream, &llvm::nulls()));
    return ExecProgram(&arena, ast, &trace_stream, &llvm::nulls());
  }
};

TEST_F(ParseAndExecuteTest, Recursion) {
  std::string source = R"(
    package Test api;
    fn Main() -> i32 {
      return
  )";
  for (int i = 0; i < 1000; ++i) {
    source += "if true then\n";
  }
  source += "1\n";
  for (int i = 0; i < 1000; ++i) {
    source += "else 0\n";
  }
  source += R"(
        ;
    }
  )";
  ParseAndExecute(source,
                  testing::MatchesRegex(R"(Exceeded recursion limit \(\d+\))"));
}

}  // namespace
}  // namespace Carbon::Testing
