// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/parse_and_execute/parse_and_execute.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>

namespace Carbon::Testing {
namespace {

using ::testing::Eq;
using ::testing::ValuesIn;

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
  auto err = ParseAndExecute("explorer/data/prelude.carbon", source);
  ASSERT_FALSE(err.ok());
  EXPECT_THAT(err.error().message(),
              Eq("RUNTIME ERROR: overflow:1: stack overflow: too many "
                 "interpreter actions on stack"));
}

class ParseAndExecuteTestFile : public ::testing::Test {
 public:
  explicit ParseAndExecuteTestFile(llvm::StringRef file) : file_(file) {}

  auto TestBody() -> void override { llvm::errs() << file_ << "\n"; }

 private:
  llvm::StringRef file_;
};

}  // namespace
}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);

  // Explicitly registers instead of INSTANTIATE_TEST_CASE_P because of ordering
  // issues between container initialization and test instantiation by
  // InitGoogleTest.
  for (int i = 1; i < argc; ++i) {
    const char* file = argv[i];
    ::testing::RegisterTest(
        "ParseAndExecuteTestForFile", file, nullptr, file, __FILE__, __LINE__,
        [=]() { return new Carbon::Testing::ParseAndExecuteTestFile(file); });
  }

  // gtest should remove flags, leaving just input files.
  return RUN_ALL_TESTS();
}
