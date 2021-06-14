// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "toolchain/lexer/tokenized_buffer_test_helpers.h"

namespace Carbon {
namespace {

using Carbon::Testing::IsKeyValueScalars;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::StrEq;

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

  // Parse the output into a YAML stream. This will print errors to stderr and
  // is the most stable view of the textual dumping API.
  llvm::SourceMgr sm;
  llvm::yaml::Stream yaml_stream(tokenized_text, sm);
  auto yaml_it = yaml_stream.begin();
  auto* root_node = llvm::dyn_cast<llvm::yaml::MappingNode>(yaml_it->getRoot());
  ASSERT_THAT(root_node, NotNull());

  // Walk the top-level mapping of tokens, dig out the sub-mapping of data for
  // each taken, and then verify those entries.
  auto mapping_it = llvm::cast<llvm::yaml::MappingNode>(root_node)->begin();
  auto* token_node = llvm::dyn_cast<llvm::yaml::KeyValueNode>(&*mapping_it);
  ASSERT_THAT(token_node, NotNull());
  auto* token_key_node =
      llvm::dyn_cast<llvm::yaml::ScalarNode>(token_node->getKey());
  ASSERT_THAT(token_key_node, NotNull());
  EXPECT_THAT(token_key_node->getRawValue(), StrEq("token"));
  auto* token_value_node =
      llvm::dyn_cast<llvm::yaml::MappingNode>(token_node->getValue());
  ASSERT_THAT(token_value_node, NotNull());
  auto token_it = token_value_node->begin();
  EXPECT_THAT(&*token_it, IsKeyValueScalars("index", "0"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("kind", "Identifier"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("line", "1"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("column", "1"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("indent", "1"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("spelling", "Hello"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("identifier", "0"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("has_trailing_space", "true"));
  EXPECT_THAT(++token_it, Eq(token_value_node->end()));

  ++mapping_it;
  token_node = llvm::dyn_cast<llvm::yaml::KeyValueNode>(&*mapping_it);
  ASSERT_THAT(token_node, NotNull());
  token_key_node = llvm::dyn_cast<llvm::yaml::ScalarNode>(token_node->getKey());
  ASSERT_THAT(token_key_node, NotNull());
  EXPECT_THAT(token_key_node->getRawValue(), StrEq("token"));
  token_value_node =
      llvm::dyn_cast<llvm::yaml::MappingNode>(token_node->getValue());
  ASSERT_THAT(token_value_node, NotNull());
  token_it = token_value_node->begin();
  EXPECT_THAT(&*token_it, IsKeyValueScalars("index", "1"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("kind", "Identifier"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("line", "1"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("column", "7"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("indent", "1"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("spelling", "World"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("identifier", "1"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("has_trailing_space", "true"));
  EXPECT_THAT(++token_it, Eq(token_value_node->end()));

  ++mapping_it;
  token_node = llvm::dyn_cast<llvm::yaml::KeyValueNode>(&*mapping_it);
  ASSERT_THAT(token_node, NotNull());
  token_key_node = llvm::dyn_cast<llvm::yaml::ScalarNode>(token_node->getKey());
  ASSERT_THAT(token_key_node, NotNull());
  EXPECT_THAT(token_key_node->getRawValue(), StrEq("token"));
  token_value_node =
      llvm::dyn_cast<llvm::yaml::MappingNode>(token_node->getValue());
  ASSERT_THAT(token_value_node, NotNull());
  token_it = token_value_node->begin();
  EXPECT_THAT(&*token_it, IsKeyValueScalars("index", "2"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("kind", "EndOfFile"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("line", "1"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("column", "12"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("indent", "1"));
  ++token_it;
  EXPECT_THAT(&*token_it, IsKeyValueScalars("spelling", ""));
  EXPECT_THAT(++token_it, Eq(token_value_node->end()));

  ASSERT_THAT(++mapping_it, Eq(root_node->end()));
  ASSERT_THAT(++yaml_it, Eq(yaml_stream.end()));

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
