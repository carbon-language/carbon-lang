// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/parse_and_execute/parse_and_execute.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon {
namespace {

using ::testing::MatchesRegex;

TEST(ParseAndExecuteTest, Recursion) {
  llvm::vfs::InMemoryFileSystem fs;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> prelude =
      llvm::MemoryBuffer::getFile("explorer/data/prelude.carbon");
  ASSERT_FALSE(prelude.getError()) << prelude.getError().message();
  ASSERT_TRUE(fs.addFile("prelude.carbon", /*ModificationTime=*/0,
                         std::move(*prelude)));

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
  ASSERT_TRUE(fs.addFile("test.carbon", /*ModificationTime=*/0,
                         llvm::MemoryBuffer::getMemBuffer(source)));

  TraceStream trace_stream;
  auto err =
      ParseAndExecute(fs, "prelude.carbon", "test.carbon",
                      /*parser_debug=*/false, &trace_stream, &llvm::nulls());
  ASSERT_FALSE(err.ok());
  // Don't expect any particular source location for the error.
  EXPECT_THAT(err.error().message(),
              MatchesRegex("RUNTIME ERROR:.* stack overflow: too many "
                           "interpreter actions on stack"));
}

}  // namespace
}  // namespace Carbon
