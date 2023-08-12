// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <utility>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "testing/util/test_raw_ostream.h"
#include "toolchain/common/yaml_test_helpers.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon::Testing {
namespace {

using ::testing::ElementsAre;
using ::testing::ContainsRegex;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::StrEq;

// Reads a file to string.
// TODO: Extract this to a helper and share it with other tests.
static auto ReadFile(std::filesystem::path path) -> std::string {
  std::ifstream proto_file(path);
  std::stringstream buffer;
  buffer << proto_file.rdbuf();
  proto_file.close();
  return buffer.str();
}

class DriverTest : public testing::Test {
 protected:
  DriverTest() : driver_(fs_, test_output_stream_, test_error_stream_) {
    char* tmpdir_env = getenv("TEST_TMPDIR");
    CARBON_CHECK(tmpdir_env != nullptr);
    test_tmpdir_ = tmpdir_env;
  }

  auto CreateTestFile(llvm::StringRef text,
                      llvm::StringRef file_name = "test_file.carbon")
      -> llvm::StringRef {
    fs_.addFile(file_name, /*ModificationTime=*/0,
                llvm::MemoryBuffer::getMemBuffer(text));
    return file_name;
  }
  
  auto ScopedTempWorkingDir() {
    // Save our current working directory.
    std::error_code ec;
    auto original_dir = std::filesystem::current_path(ec);
    CARBON_CHECK(!ec) << ec.message();

    const auto* unit_test = ::testing::UnitTest::GetInstance();
    const auto* test_info = unit_test->current_test_info();
    std::filesystem::path test_dir =
        test_tmpdir_.append(llvm::formatv("{0}_{1}", test_info->test_suite_name(),
                                    test_info->name())
                          .str());
    std::filesystem::create_directory(test_dir, ec);
    CARBON_CHECK(!ec) << "Could not create test working dir '"
                      << test_dir << "': " << ec.message();
    std::filesystem::current_path(test_dir, ec);
    CARBON_CHECK(!ec) << "Could not change the current working dir to '"
                      << test_dir << "': " << ec.message();
    return llvm::make_scope_exit([original_dir, test_dir] {
      std::error_code ec;
      std::filesystem::current_path(original_dir, ec);
      CARBON_CHECK(!ec) << "Could not change the current working dir to '"
                        << original_dir << "': " << ec.message();
      std::filesystem::remove_all(test_dir, ec);
      CARBON_CHECK(!ec) << "Could not remove the test working dir '"
                        << test_dir << "': " << ec.message();

    });
  }

  llvm::vfs::InMemoryFileSystem fs_;
  TestRawOstream test_output_stream_;
  TestRawOstream test_error_stream_;

  // Some tests work directly with files in the test temporary directory.
  std::filesystem::path test_tmpdir_;

  Driver driver_;
};

TEST_F(DriverTest, BadCommandErrors) {
  EXPECT_FALSE(driver_.RunCommand({}));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver_.RunCommand({"foo"}));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver_.RunCommand({"foo --bar --baz"}));
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));
}

TEST_F(DriverTest, CompileCommandErrors) {
  // No input file. This error message is important so check all of it.
  EXPECT_FALSE(driver_.RunCommand({"compile"}));
  EXPECT_THAT(
      test_error_stream_.TakeStr(),
      StrEq("ERROR: Not all required positional arguments were provided. First "
            "missing and required positional argument: 'FILE'\n"));

  // Invalid output filename. No reliably error message here.
  // TODO: Likely want a different filename on Windows.
  auto empty_file = CreateTestFile("");
  EXPECT_FALSE(
      driver_.RunCommand({"compile", "--output=/dev/empty", empty_file}));
  EXPECT_THAT(test_error_stream_.TakeStr(), ContainsRegex("ERROR: .*/dev/empty.*"));
}

TEST_F(DriverTest, DumpTokens) {
  auto file = CreateTestFile("Hello World");
  EXPECT_TRUE(driver_.RunCommand(
      {"compile", "--phase=lex", "--dump-tokens", file}));
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

TEST_F(DriverTest, StdoutOutput) {
  // Use explicit filenames so we can look for those to validate output.
  CreateTestFile("fn Main() -> i32 { return 0; }", "test.carbon");

  EXPECT_TRUE(driver_.RunCommand({"compile", "--output=-", "test.carbon"}));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // The default is textual assembly.
  EXPECT_THAT(test_output_stream_.TakeStr(),
              ContainsRegex("\\.file\\s+\"test.carbon\""));

  EXPECT_TRUE(driver_.RunCommand({"compile", "--output=-", "--force-obj-output", "test.carbon"}));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  std::string output = test_output_stream_.TakeStr();
  auto result = llvm::object::createBinary(llvm::MemoryBufferRef(output, "test_output"));
  if (auto error = result.takeError()) {
    FAIL() << toString(std::move(error));
  }
  EXPECT_TRUE(result->get()->isObject());
}

TEST_F(DriverTest, FileOutput) {
  auto scope = ScopedTempWorkingDir();

  // Use explicit filenames as the default output filename is computed from
  // this, and we can use this to validate output.
  CreateTestFile("fn Main() -> i32 { return 0; }", "test.carbon");

  // Object output (the default) uses `.o`.
  // TODO: This should actually reflect the platform defaults.
  EXPECT_TRUE(driver_.RunCommand({"compile", "test.carbon"}));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // Ensure we wrote an object file of some form with the correct name.
  auto result = llvm::object::createBinary("test.o");
  if (auto error = result.takeError()) {
    FAIL() << toString(std::move(error));
  }
  EXPECT_TRUE(result->getBinary()->isObject());

  // Assembly output uses `.s`.
  // TODO: This should actually reflect the platform defaults.
  EXPECT_TRUE(driver_.RunCommand({"compile", "--asm-output", "test.carbon"}));
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // TODO: This may need to be tailored to other assembly formats.
  EXPECT_THAT(ReadFile("test.s"), ContainsRegex("\\.file\\s+\"test.carbon\""));
}

}  // namespace
}  // namespace Carbon::Testing
