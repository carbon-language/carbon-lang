// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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

/// A raw_ostream that makes it easy to repeatedly check streamed output.
class RawTestOstream : public llvm::raw_ostream {
 public:
  ~RawTestOstream() override {
    flush();
    if (!buffer_.empty()) {
      ADD_FAILURE() << "Unchecked output:\n" << buffer_;
    }
  }

  /// Flushes the stream and returns the contents so far, clearing the stream
  /// back to empty.
  auto TakeStr() -> std::string {
    flush();
    std::string result = std::move(buffer_);
    buffer_.clear();
    return result;
  }

 private:
  void write_impl(const char* ptr, size_t size) override {
    buffer_.append(ptr, ptr + size);
  }

  [[nodiscard]] auto current_pos() const -> uint64_t override {
    return buffer_.size();
  }

  std::string buffer_;
};

TEST(DriverTest, FullCommandErrors) {
  RawTestOstream test_output_stream;
  RawTestOstream test_error_stream;
  Driver driver = Driver(test_output_stream, test_error_stream);

  EXPECT_FALSE(driver.RunFullCommand({}));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver.RunFullCommand({"foo"}));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver.RunFullCommand({"foo --bar --baz"}));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));
}

TEST(DriverTest, Help) {
  RawTestOstream test_output_stream;
  RawTestOstream test_error_stream;
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
  RawTestOstream test_output_stream;
  RawTestOstream test_error_stream;
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
  RawTestOstream test_output_stream;
  RawTestOstream test_error_stream;
  Driver driver = Driver(test_output_stream, test_error_stream);

  auto test_file_path = CreateTestFile("Hello World");
  EXPECT_TRUE(driver.RunDumpSubcommand(ConsoleDiagnosticConsumer(),
                                       {"tokens", test_file_path}));
  EXPECT_THAT(test_error_stream.TakeStr(), StrEq(""));
  auto tokenized_text = test_output_stream.TakeStr();

  EXPECT_THAT(Yaml::Value::FromText(tokenized_text),
              ElementsAre(Yaml::MappingValue{
                  {"token", Yaml::MappingValue{{"index", "0"},
                                               {"kind", "Identifier"},
                                               {"line", "1"},
                                               {"column", "1"},
                                               {"indent", "1"},
                                               {"spelling", "Hello"},
                                               {"identifier", "0"},
                                               {"has_trailing_space", "true"}}},
                  {"token", Yaml::MappingValue{{"index", "1"},
                                               {"kind", "Identifier"},
                                               {"line", "1"},
                                               {"column", "7"},
                                               {"indent", "1"},
                                               {"spelling", "World"},
                                               {"identifier", "1"},
                                               {"has_trailing_space", "true"}}},
                  {"token", Yaml::MappingValue{{"index", "2"},
                                               {"kind", "EndOfFile"},
                                               {"line", "1"},
                                               {"column", "12"},
                                               {"indent", "1"},
                                               {"spelling", ""}}}}));

  // Check that the subcommand dispatch works.
  EXPECT_TRUE(driver.RunFullCommand({"dump", "tokens", test_file_path}));
  EXPECT_THAT(test_error_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(tokenized_text));
}

TEST(DriverTest, DumpErrors) {
  RawTestOstream test_output_stream;
  RawTestOstream test_error_stream;
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

TEST(DriverTest, DumpParseTree) {
  RawTestOstream test_output_stream;
  RawTestOstream test_error_stream;
  Driver driver = Driver(test_output_stream, test_error_stream);

  auto test_file_path = CreateTestFile("var v: Int = 42;");
  EXPECT_TRUE(driver.RunDumpSubcommand(ConsoleDiagnosticConsumer(),
                                       {"parse-tree", test_file_path}));
  EXPECT_THAT(test_error_stream.TakeStr(), StrEq(""));
  auto tokenized_text = test_output_stream.TakeStr();

  EXPECT_THAT(
      Yaml::Value::FromText(tokenized_text),
      ElementsAre(Yaml::SequenceValue{
          Yaml::MappingValue{
              {"node_index", "6"},
              {"kind", "VariableDeclaration"},
              {"text", "var"},
              {"subtree_size", "7"},
              {"children",
               Yaml::SequenceValue{
                   Yaml::MappingValue{
                       {"node_index", "2"},
                       {"kind", "PatternBinding"},
                       {"text", ":"},
                       {"subtree_size", "3"},
                       {"children",
                        Yaml::SequenceValue{
                            Yaml::MappingValue{{"node_index", "0"},
                                               {"kind", "DeclaredName"},
                                               {"text", "v"}},
                            Yaml::MappingValue{{"node_index", "1"},
                                               {"kind", "NameReference"},
                                               {"text", "Int"}}}}},
                   Yaml::MappingValue{{"node_index", "4"},
                                      {"kind", "VariableInitializer"},
                                      {"text", "="},
                                      {"subtree_size", "2"},
                                      {"children",  //
                                       Yaml::SequenceValue{Yaml::MappingValue{
                                           {"node_index", "3"},
                                           {"kind", "Literal"},
                                           {"text", "42"}}}}},
                   Yaml::MappingValue{{"node_index", "5"},
                                      {"kind", "DeclarationEnd"},
                                      {"text", ";"}}}}},
          Yaml::MappingValue{{"node_index", "7"},  //
                             {"kind", "FileEnd"},
                             {"text", ""}}}));

  // Check that the subcommand dispatch works.
  EXPECT_TRUE(driver.RunFullCommand({"dump", "parse-tree", test_file_path}));
  EXPECT_THAT(test_error_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(tokenized_text));
}

}  // namespace
}  // namespace Carbon::Testing
