// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <utility>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/FormatVariadic.h"
#include "testing/base/test_raw_ostream.h"
#include "toolchain/testing/yaml_test_helpers.h"

namespace Carbon {
namespace {

using ::Carbon::Testing::TestRawOstream;
using ::testing::_;
using ::testing::ContainsRegex;
using ::testing::HasSubstr;
using ::testing::StrEq;

namespace Yaml = ::Carbon::Testing::Yaml;

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
  DriverTest() : driver_(fs_, "", test_output_stream_, test_error_stream_) {
    char* tmpdir_env = getenv("TEST_TMPDIR");
    CARBON_CHECK(tmpdir_env != nullptr);
    test_tmpdir_ = tmpdir_env;
  }

  auto MakeTestFile(llvm::StringRef text,
                    llvm::StringRef filename = "test_file.carbon")
      -> llvm::StringRef {
    fs_.addFile(filename, /*ModificationTime=*/0,
                llvm::MemoryBuffer::getMemBuffer(text));
    return filename;
  }

  // Makes a temp directory and changes the working directory to it. Returns an
  // LLVM `scope_exit` that will restore the working directory and remove the
  // temporary directory (and everything it contains) when destroyed.
  auto ScopedTempWorkingDir() {
    // Save our current working directory.
    std::error_code ec;
    auto original_dir = std::filesystem::current_path(ec);
    CARBON_CHECK(!ec) << ec.message();

    const auto* unit_test = ::testing::UnitTest::GetInstance();
    const auto* test_info = unit_test->current_test_info();
    std::filesystem::path test_dir = test_tmpdir_.append(
        llvm::formatv("{0}_{1}", test_info->test_suite_name(),
                      test_info->name())
            .str());
    std::filesystem::create_directory(test_dir, ec);
    CARBON_CHECK(!ec) << "Could not create test working dir '" << test_dir
                      << "': " << ec.message();
    std::filesystem::current_path(test_dir, ec);
    CARBON_CHECK(!ec) << "Could not change the current working dir to '"
                      << test_dir << "': " << ec.message();
    return llvm::make_scope_exit([original_dir, test_dir] {
      std::error_code ec;
      std::filesystem::current_path(original_dir, ec);
      CARBON_CHECK(!ec) << "Could not change the current working dir to '"
                        << original_dir << "': " << ec.message();
      std::filesystem::remove_all(test_dir, ec);
      CARBON_CHECK(!ec) << "Could not remove the test working dir '" << test_dir
                        << "': " << ec.message();
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
  EXPECT_FALSE(driver_.RunCommand({}).success);
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver_.RunCommand({"foo"}).success);
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));

  EXPECT_FALSE(driver_.RunCommand({"foo --bar --baz"}).success);
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("ERROR"));
}

TEST_F(DriverTest, CompileCommandErrors) {
  // No input file. This error message is important so check all of it.
  EXPECT_FALSE(driver_.RunCommand({"compile"}).success);
  EXPECT_THAT(
      test_error_stream_.TakeStr(),
      StrEq("ERROR: Not all required positional arguments were provided. First "
            "missing and required positional argument: 'FILE'\n"));

  // Invalid output filename. No reliably error message here.
  // TODO: Likely want a different filename on Windows.
  auto empty_file = MakeTestFile("");
  EXPECT_FALSE(driver_
                   .RunCommand({"compile", "--no-prelude-import",
                                "--output=/dev/empty", empty_file})
                   .success);
  EXPECT_THAT(test_error_stream_.TakeStr(),
              ContainsRegex("ERROR: .*/dev/empty.*"));
}

TEST_F(DriverTest, DumpTokens) {
  auto file = MakeTestFile("Hello World");
  EXPECT_TRUE(driver_
                  .RunCommand({"compile", "--no-prelude-import", "--phase=lex",
                               "--dump-tokens", file})
                  .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // Verify there is output without examining it.
  EXPECT_THAT(Yaml::Value::FromText(test_output_stream_.TakeStr()),
              Yaml::IsYaml(_));
}

TEST_F(DriverTest, DumpParseTree) {
  auto file = MakeTestFile("var v: () = ();");
  EXPECT_TRUE(driver_
                  .RunCommand({"compile", "--no-prelude-import",
                               "--phase=parse", "--dump-parse-tree", file})
                  .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // Verify there is output without examining it.
  EXPECT_THAT(Yaml::Value::FromText(test_output_stream_.TakeStr()),
              Yaml::IsYaml(_));
}

TEST_F(DriverTest, StdoutOutput) {
  // Use explicit filenames so we can look for those to validate output.
  MakeTestFile("fn Main() {}", "test.carbon");

  EXPECT_TRUE(driver_
                  .RunCommand({"compile", "--no-prelude-import", "--output=-",
                               "test.carbon"})
                  .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // The default is textual assembly.
  EXPECT_THAT(test_output_stream_.TakeStr(), ContainsRegex("Main:"));

  EXPECT_TRUE(driver_
                  .RunCommand({"compile", "--no-prelude-import", "--output=-",
                               "--force-obj-output", "test.carbon"})
                  .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  std::string output = test_output_stream_.TakeStr();
  auto result =
      llvm::object::createBinary(llvm::MemoryBufferRef(output, "test_output"));
  if (auto error = result.takeError()) {
    FAIL() << toString(std::move(error));
  }
  EXPECT_TRUE(result->get()->isObject());
}

TEST_F(DriverTest, FileOutput) {
  auto scope = ScopedTempWorkingDir();

  // Use explicit filenames as the default output filename is computed from
  // this, and we can use this to validate output.
  MakeTestFile("fn Main() {}", "test.carbon");

  // Object output (the default) uses `.o`.
  // TODO: This should actually reflect the platform defaults.
  EXPECT_TRUE(
      driver_.RunCommand({"compile", "--no-prelude-import", "test.carbon"})
          .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // Ensure we wrote an object file of some form with the correct name.
  auto result = llvm::object::createBinary("test.o");
  if (auto error = result.takeError()) {
    FAIL() << toString(std::move(error));
  }
  EXPECT_TRUE(result->getBinary()->isObject());

  // Assembly output uses `.s`.
  // TODO: This should actually reflect the platform defaults.
  EXPECT_TRUE(driver_
                  .RunCommand({"compile", "--no-prelude-import", "--asm-output",
                               "test.carbon"})
                  .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // TODO: This may need to be tailored to other assembly formats.
  EXPECT_THAT(ReadFile("test.s"), ContainsRegex("Main:"));
}

}  // namespace
}  // namespace Carbon
