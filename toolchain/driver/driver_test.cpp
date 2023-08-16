// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "testing/base/test_raw_ostream.h"
#include "toolchain/base/yaml_test_helpers.h"
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
  EXPECT_FALSE(driver_.RunFullCommand({}));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver_.RunFullCommand({"foo"}));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver_.RunFullCommand({"foo --bar --baz"}));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));
}

TEST_F(DriverTest, Help) {
  EXPECT_TRUE(driver_.RunHelpSubcommand(ConsoleDiagnosticConsumer(), {}));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  auto help_text = test_output_stream_.TakeStr();

  // Help text should mention each subcommand.
#define CARBON_SUBCOMMAND(Name, Spelling, ...) \
  EXPECT_THAT(help_text, HasSubstr(Spelling));
#include "toolchain/driver/flags.def"

  // Check that the subcommand dispatch works.
  EXPECT_TRUE(driver_.RunFullCommand({"help"}));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  EXPECT_THAT(test_output_stream_.TakeStr(), StrEq(help_text));
}

TEST_F(DriverTest, HelpErrors) {
  EXPECT_FALSE(driver_.RunHelpSubcommand(ConsoleDiagnosticConsumer(), {"foo"}));
  EXPECT_THAT(test_output_stream_.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(
      driver_.RunHelpSubcommand(ConsoleDiagnosticConsumer(), {"help"}));
  EXPECT_THAT(test_output_stream_.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(
      driver_.RunHelpSubcommand(ConsoleDiagnosticConsumer(), {"--xyz"}));
  EXPECT_THAT(test_output_stream_.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));
}

TEST_F(DriverTest, DumpTokens) {
  auto file = CreateTestFile("Hello World");
  EXPECT_TRUE(
      driver_.RunDumpSubcommand(ConsoleDiagnosticConsumer(), {"tokens", file}));
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

  // Check that the subcommand dispatch works.
  EXPECT_TRUE(driver_.RunFullCommand({"dump", "tokens", file}));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  EXPECT_THAT(test_output_stream_.TakeStr(), StrEq(tokenized_text));
}

TEST_F(DriverTest, DumpErrors) {
  EXPECT_FALSE(driver_.RunDumpSubcommand(ConsoleDiagnosticConsumer(), {"foo"}));
  EXPECT_THAT(test_output_stream_.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(
      driver_.RunDumpSubcommand(ConsoleDiagnosticConsumer(), {"--xyz"}));
  EXPECT_THAT(test_output_stream_.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(
      driver_.RunDumpSubcommand(ConsoleDiagnosticConsumer(), {"tokens"}));
  EXPECT_THAT(test_output_stream_.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver_.RunDumpSubcommand(ConsoleDiagnosticConsumer(),
                                         {"tokens", "/not/a/real/file/name"}));
  EXPECT_THAT(test_output_stream_.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));
}

TEST_F(DriverTest, DumpParseTree) {
  auto file = CreateTestFile("var v: Int = 42;");
  EXPECT_TRUE(driver_.RunDumpSubcommand(ConsoleDiagnosticConsumer(),
                                        {"parse-tree", file}));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // Verify there is output without examining it.
  EXPECT_FALSE(test_output_stream_.TakeStr().empty());

  // Check that the subcommand dispatch works.
  EXPECT_TRUE(driver_.RunFullCommand({"dump", "parse-tree", file}));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // Verify there is output without examining it.
  EXPECT_FALSE(test_output_stream_.TakeStr().empty());
}

}  // namespace
}  // namespace Carbon::Testing
