// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "testing/util/test_raw_ostream.h"
#include "toolchain/common/yaml_test_helpers.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon::Testing {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::StrEq;

class DriverTest : public testing::Test {
 protected:
  DriverTest() : driver_(fs_, test_output_stream_, test_error_stream_) {}

  auto CreateTestFile(llvm::StringRef text) -> llvm::StringRef {
    static constexpr llvm::StringLiteral TestFileName = "test_file.carbon";
    fs_.addFile(TestFileName, /*ModificationTime=*/0,
                llvm::MemoryBuffer::getMemBuffer(text));
    return TestFileName;
  }

  llvm::vfs::InMemoryFileSystem fs_;
  TestRawOstream test_output_stream_;
  TestRawOstream test_error_stream_;
  Driver driver_;
};

TEST_F(DriverTest, FullCommandErrors) {
  EXPECT_FALSE(driver_.RunCommand({}));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver_.RunCommand({"foo"}));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver_.RunCommand({"foo --bar --baz"}));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));
}

TEST_F(DriverTest, DumpTokens) {
  auto file = CreateTestFile("Hello World");
  EXPECT_TRUE(driver_.RunCommand(
      {"compile", "--phase=tokenize", "--dump-tokens", file}));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  auto tokenized_text = test_output_stream_.TakeStr();

  EXPECT_THAT(Yaml::Value::FromText(tokenized_text),
              ElementsAre(Yaml::SequenceValue{
                  Yaml::MappingValue{{"index", "0"},
                                     {"kind", "Identifier"},
                                     {"line", "1"},
                                     {"column", "1"},
                                     {"indent", "1"},
                                     {"spelling", "Hello"},
                                     {"identifier", "0"},
                                     {"has_trailing_space", "true"}},
                  Yaml::MappingValue{{"index", "1"},
                                     {"kind", "Identifier"},
                                     {"line", "1"},
                                     {"column", "7"},
                                     {"indent", "1"},
                                     {"spelling", "World"},
                                     {"identifier", "1"},
                                     {"has_trailing_space", "true"}},
                  Yaml::MappingValue{{"index", "2"},
                                     {"kind", "EndOfFile"},
                                     {"line", "1"},
                                     {"column", "12"},
                                     {"indent", "1"},
                                     {"spelling", ""}}}));
}

TEST_F(DriverTest, DumpParseTree) {
  auto file = CreateTestFile("var v: i32 = 42;");
  EXPECT_TRUE(driver_.RunCommand(
      {"compile", "--phase=parse", "--dump-parse-tree", file}));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // Verify there is output without examining it.
  EXPECT_FALSE(test_output_stream_.TakeStr().empty());
}

}  // namespace
}  // namespace Carbon::Testing
