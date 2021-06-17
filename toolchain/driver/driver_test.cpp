// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "toolchain/common/yaml_test_helpers.h"

namespace Carbon {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::Pair;
using ::testing::StrEq;
namespace Yaml = Carbon::Testing::Yaml;

/// A raw_ostream that makes it easy to repeatedly check streamed output.
class RawTestOstream : public llvm::raw_ostream {
  std::string buffer;

  void write_impl(const char* ptr, size_t size) override {
    buffer.append(ptr, ptr + size);
  }

  [[nodiscard]] auto current_pos() const -> uint64_t override {
    return buffer.size();
  }

 public:
  ~RawTestOstream() override {
    flush();
    if (!buffer.empty()) {
      ADD_FAILURE() << "Unchecked output:\n" << buffer;
    }
  }

  /// Flushes the stream and returns the contents so far, clearing the stream
  /// back to empty.
  auto TakeStr() -> std::string {
    flush();
    std::string result = std::move(buffer);
    buffer.clear();
    return result;
  }
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

  EXPECT_TRUE(driver.RunHelpSubcommand({}));
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

  EXPECT_FALSE(driver.RunHelpSubcommand({"foo"}));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver.RunHelpSubcommand({"help"}));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver.RunHelpSubcommand({"--xyz"}));
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
  EXPECT_TRUE(driver.RunDumpTokensSubcommand({test_file_path}));
  EXPECT_THAT(test_error_stream.TakeStr(), StrEq(""));
  auto tokenized_text = test_output_stream.TakeStr();

  EXPECT_THAT(
      Yaml::Value::FromText(tokenized_text),
      ElementsAre(Yaml::Mapping(ElementsAre(
          Pair("token",
               Yaml::Mapping(ElementsAre(
                   Pair("index", "0"), Pair("kind", "Identifier"),
                   Pair("line", "1"), Pair("column", "1"), Pair("indent", "1"),
                   Pair("spelling", "Hello"), Pair("identifier", "0"),
                   Pair("has_trailing_space", "true")))),
          Pair("token",
               Yaml::Mapping(ElementsAre(
                   Pair("index", "1"), Pair("kind", "Identifier"),
                   Pair("line", "1"), Pair("column", "7"), Pair("indent", "1"),
                   Pair("spelling", "World"), Pair("identifier", "1"),
                   Pair("has_trailing_space", "true")))),
          Pair("token", Yaml::Mapping(ElementsAre(
                            Pair("index", "2"), Pair("kind", "EndOfFile"),
                            Pair("line", "1"), Pair("column", "12"),
                            Pair("indent", "1"), Pair("spelling", ""))))))));

  // Check that the subcommand dispatch works.
  EXPECT_TRUE(driver.RunFullCommand({"dump-tokens", test_file_path}));
  EXPECT_THAT(test_error_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(tokenized_text));
}

TEST(DriverTest, DumpTokenErrors) {
  RawTestOstream test_output_stream;
  RawTestOstream test_error_stream;
  Driver driver = Driver(test_output_stream, test_error_stream);

  EXPECT_FALSE(driver.RunDumpTokensSubcommand({}));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver.RunDumpTokensSubcommand({"--xyz"}));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver.RunDumpTokensSubcommand({"/not/a/real/file/name"}));
  EXPECT_THAT(test_output_stream.TakeStr(), StrEq(""));
  EXPECT_THAT(test_error_stream.TakeStr(), HasSubstr("ERROR"));
}

}  // namespace
}  // namespace Carbon
