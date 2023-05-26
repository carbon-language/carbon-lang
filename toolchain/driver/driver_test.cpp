// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/test_raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "toolchain/common/yaml_test_helpers.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon::Testing {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::StrEq;

TEST(DriverTest, CommandErrors) {
  TestRawOstream test_output_stream;
  TestRawOstream test_error_stream;
  Driver driver = Driver(test_output_stream, test_error_stream);

  EXPECT_FALSE(driver.RunCommand({}));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver.RunCommand({"foo"}));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver.RunCommand({"foo --bar --baz"}));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));
}

#if 0
TEST(DriverTest, Help) {
  TestRawOstream test_output_stream;
  TestRawOstream test_error_stream;
  Driver driver = Driver(test_output_stream, test_error_stream);

  EXPECT_TRUE(driver.RunHelpSubcommand(ConsoleDiagnosticConsumer(), {}));
  EXPECT_THAT(test_error_stream.TakeStr(), StrEq(""));
  auto help_text = test_output_stream.TakeStr();

  // Help text should mention each subcommand.
#define CARBON_SUBCOMMAND(Name, Spelling, ...) \
  EXPECT_THAT(help_text, HasSubstr(Spelling));
#include "toolchain/driver/flags.def"

  // Check that the subcommand dispatch works.
  EXPECT_TRUE(driver.RunFullCommand({"help"}));
  EXPECT_THAT(test_error_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(help_text));
}

TEST(DriverTest, HelpErrors) {
  TestRawOstream test_output_stream;
  TestRawOstream test_error_stream;
  Driver driver = Driver(test_output_stream, test_error_stream);

  EXPECT_FALSE(driver.RunHelpSubcommand(ConsoleDiagnosticConsumer(), {"foo"}));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver.RunHelpSubcommand(ConsoleDiagnosticConsumer(), {"help"}));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(
      driver.RunHelpSubcommand(ConsoleDiagnosticConsumer(), {"--xyz"}));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));
}
#endif

auto CreateTestFile(llvm::StringRef text) -> std::string {
  int fd = -1;
  llvm::SmallString<1024> path;
  auto ec = llvm::sys::fs::createTemporaryFile("test_file", ".txt", fd, path);
  if (ec) {
    llvm::report_fatal_error(llvm::Twine("Failed to create temporary file: ") +
                             ec.message());
  }

  llvm::raw_fd_ostream s(fd, /*shouldClose=*/true);
  s << text;
  s.close();

  return path.str().str();
}

TEST(DriverTest, DumpTokens) {
  TestRawOstream test_output_stream;
  TestRawOstream test_error_stream;
  Driver driver = Driver(test_output_stream, test_error_stream);

  auto test_file_path = CreateTestFile("Hello World");
  EXPECT_TRUE(driver.RunCommand(
      {"compile", "--phase=tokenize", "--dump-tokens", test_file_path}));
  EXPECT_THAT(test_error_stream.TakeStr(), StrEq(""));
  auto tokenized_text = test_output_stream.TakeStr();

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

#if 0
TEST(DriverTest, DumpErrors) {
  TestRawOstream test_output_stream;
  TestRawOstream test_error_stream;
  Driver driver = Driver(test_output_stream, test_error_stream);

  EXPECT_FALSE(driver.RunDumpSubcommand(ConsoleDiagnosticConsumer(), {"foo"}));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(
      driver.RunDumpSubcommand(ConsoleDiagnosticConsumer(), {"--xyz"}));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(
      driver.RunDumpSubcommand(ConsoleDiagnosticConsumer(), {"tokens"}));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver.RunDumpSubcommand(ConsoleDiagnosticConsumer(),
                                        {"tokens", "/not/a/real/file/name"}));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));
}
#endif

TEST(DriverTest, DumpParseTree) {
  TestRawOstream test_output_stream;
  TestRawOstream test_error_stream;
  Driver driver = Driver(test_output_stream, test_error_stream);

  auto test_file_path = CreateTestFile("var v: Int = 42;");
  EXPECT_TRUE(driver.RunCommand(
      {"compile", "--phase=parse", "--dump-parse-tree", test_file_path}));
  EXPECT_THAT(test_error_stream.TakeStr(), StrEq(""));
  // Verify there is output without examining it.
  EXPECT_FALSE(test_output_stream.TakeStr().empty());
}

}  // namespace
}  // namespace Carbon::Testing
