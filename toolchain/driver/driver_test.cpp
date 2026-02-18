// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "common/error_test_helpers.h"
#include "common/filesystem.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "testing/base/file_helpers.h"
#include "testing/base/global_exe_path.h"
#include "toolchain/testing/yaml_test_helpers.h"

namespace Carbon {
namespace {

using ::testing::_;
using ::testing::ContainsRegex;
using ::testing::HasSubstr;
using Testing::IsSuccess;
using ::testing::Ne;
using ::testing::NotNull;
using ::testing::StrEq;

namespace Yaml = ::Carbon::Testing::Yaml;

class DriverTest : public testing::Test {
 public:
  DriverTest()
      : installation_(
            InstallPaths::MakeForBazelRunfiles(Testing::GetExePath())),
        driver_(fs_, &installation_, /*input_stream=*/nullptr,
                &test_output_stream_, &test_error_stream_) {
    test_tmpdir_ = Testing::GetTempDirectory();
  }

  auto MakeTestFile(llvm::StringRef text,
                    llvm::StringRef filename = "test_file.carbon")
      -> llvm::StringRef {
    fs_->addFile(filename, /*ModificationTime=*/0,
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
    CARBON_CHECK(!ec, "{0}", ec.message());
    Driver original_driver = std::move(driver_);

    const auto* unit_test = ::testing::UnitTest::GetInstance();
    const auto* test_info = unit_test->current_test_info();
    std::filesystem::path test_dir = test_tmpdir_.append(
        llvm::formatv("{0}_{1}", test_info->test_suite_name(),
                      test_info->name())
            .str());
    std::filesystem::create_directory(test_dir, ec);
    CARBON_CHECK(!ec, "Could not create test working dir '{0}': {1}", test_dir,
                 ec.message());
    std::filesystem::current_path(test_dir, ec);
    CARBON_CHECK(!ec, "Could not change the current working dir to '{0}': {1}",
                 test_dir, ec.message());

    // Build an overlay filesystem between the in-memory one and this new
    // directory.
    llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlay_fs =
        new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem());
    overlay_fs->pushOverlay(fs_);

    // Rebuild the driver around this filesystem.
    driver_ = Driver(overlay_fs, &installation_, /*input_stream=*/nullptr,
                     &test_output_stream_, &test_error_stream_);

    return llvm::scope_exit([this, original_dir, original_driver, test_dir] {
      std::error_code ec;
      std::filesystem::current_path(original_dir, ec);
      CARBON_CHECK(!ec,
                   "Could not change the current working dir to '{0}': {1}",
                   original_dir, ec.message());
      driver_ = original_driver;
      std::filesystem::remove_all(test_dir, ec);
      CARBON_CHECK(!ec, "Could not remove the test working dir '{0}': {1}",
                   test_dir, ec.message());
    });
  }

  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs_ =
      new llvm::vfs::InMemoryFileSystem;
  const InstallPaths installation_;
  RawStringOstream test_output_stream_;
  RawStringOstream test_error_stream_;

  // Some tests work directly with files in the test temporary directory.
  std::filesystem::path test_tmpdir_;

  Driver driver_;
};

TEST_F(DriverTest, BadCommandErrors) {
  EXPECT_FALSE(driver_.RunCommand({}).success);
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("error:"));

  EXPECT_FALSE(driver_.RunCommand({"foo"}).success);
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("error:"));

  EXPECT_FALSE(driver_.RunCommand({"foo --bar --baz"}).success);
  EXPECT_THAT(test_error_stream_.TakeStr(), HasSubstr("error:"));
}

TEST_F(DriverTest, CompileCommandErrors) {
  // No input file. This error message is important so check all of it.
  EXPECT_FALSE(driver_.RunCommand({"compile"}).success);
  EXPECT_THAT(
      test_error_stream_.TakeStr(),
      StrEq("error: not all required positional arguments were provided; first "
            "missing and required positional argument: `FILE`\n"));

  // Pass non-existing file
  EXPECT_FALSE(driver_
                   .RunCommand({"compile", "--dump-mem-usage",
                                "non-existing-file.carbon"})
                   .success);
  EXPECT_THAT(
      test_error_stream_.TakeStr(),
      ContainsRegex("No such file or directory[\\n]*non-existing-file.carbon"));
  // Flush output stream, because it's ok that it's not empty here.
  test_output_stream_.TakeStr();

  // Invalid output filename. No reliably error message here.
  // TODO: Likely want a different filename on Windows.
  auto empty_file = MakeTestFile("");
  EXPECT_FALSE(driver_
                   .RunCommand({"compile", "--no-prelude-import",
                                "--output=/dev/empty", empty_file})
                   .success);
  EXPECT_THAT(test_error_stream_.TakeStr(),
              ContainsRegex("error: .*/dev/empty.*"));
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
  MakeTestFile("fn Run() {}", "test.carbon");

  EXPECT_TRUE(driver_
                  .RunCommand({"compile", "--no-prelude-import", "--output=-",
                               "test.carbon"})
                  .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));
  // The default is textual assembly.
  EXPECT_THAT(test_output_stream_.TakeStr(), ContainsRegex("main:"));

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
  MakeTestFile("fn Run() {}", "test.carbon");

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
  EXPECT_THAT(*Testing::ReadFile("test.s"), ContainsRegex("main:"));
}

TEST_F(DriverTest, Link) {
  auto scope = ScopedTempWorkingDir();

  // First compile a file to get a linkable object.
  MakeTestFile("fn Run() {}", "test.carbon");
  ASSERT_TRUE(
      driver_.RunCommand({"compile", "--no-prelude-import", "test.carbon"})
          .success)
      << test_error_stream_.TakeStr();

  // Now link this into a binary. Note that we suppress building runtimes on
  // demand here as no runtimes should be needed for the empty program.
  EXPECT_TRUE(driver_
                  .RunCommand({"--no-build-runtimes", "link", "--output=test",
                               "test.o"})
                  .success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));

  // Ensure we wrote an executable file of some form with the correct name.
  // TODO: We may need to update this if we implicitly synthesize a
  // platform-specific `.exe` suffix or something similar.
  auto result = llvm::object::createBinary("test");
  if (auto error = result.takeError()) {
    FAIL() << toString(std::move(error));
  }
  // Executables are also classified as object files.
  EXPECT_TRUE(result->getBinary()->isObject());
}

TEST_F(DriverTest, ConfigJson) {
  EXPECT_TRUE(driver_.RunCommand({"config", "--json"}).success);
  EXPECT_THAT(test_error_stream_.TakeStr(), StrEq(""));

  // Make sure the output parses as JSON.
  std::string output = test_output_stream_.TakeStr();
  llvm::Expected<llvm::json::Value> json_value = llvm::json::parse(output);
  if (auto error = json_value.takeError()) {
    FAIL() << "Unable to parse to JSON: " << toString(std::move(error))
           << "\nOriginal text:\n"
           << output << "\n";
  }
  llvm::json::Object* json_obj = json_value->getAsObject();
  ASSERT_THAT(json_obj, NotNull());

  // Check relevant paths in the output point to existing directories.
  std::optional<llvm::StringRef> install_root =
      json_obj->getString("INSTALL_ROOT");
  ASSERT_THAT(install_root, Ne(std::nullopt));
  EXPECT_THAT(Filesystem::Cwd().OpenDir(install_root->str()), IsSuccess(_));

  std::optional<llvm::StringRef> clang_sysroot =
      json_obj->getString("CLANG_SYSROOT");
  ASSERT_THAT(clang_sysroot, Ne(std::nullopt));
  EXPECT_THAT(Filesystem::Cwd().OpenDir(clang_sysroot->str()), IsSuccess(_));
}

}  // namespace
}  // namespace Carbon
