// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_runner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <utility>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"
#include "testing/base/test_raw_ostream.h"

namespace Carbon {
namespace {

using ::Carbon::Testing::TestRawOstream;
using ::testing::HasSubstr;
using ::testing::StrEq;

// While these are marked as "internal" APIs, they seem to work and be pretty
// widely used for their exact documented behavior.
using ::testing::internal::CaptureStderr;
using ::testing::internal::CaptureStdout;
using ::testing::internal::GetCapturedStderr;
using ::testing::internal::GetCapturedStdout;

// Calls the provided lambda with `stderr` and `stdout` captured and saved into
// the provided output parameters. The lambda's result is returned. It is
// important to not put anything inside the lambda whose output would be useful
// in interpreting test errors such as Google Test assertions as their output
// will end up captured as well.
template <typename CallableT>
static auto RunWithCapturedOutput(std::string& out, std::string& err,
                                  CallableT callable) {
  CaptureStderr();
  CaptureStdout();
  auto result = callable();
  // No need to flush stderr.
  err = GetCapturedStderr();
  llvm::outs().flush();
  out = GetCapturedStdout();
  return result;
}

TEST(ClangRunnerTest, Version) {
  TestRawOstream test_os;
  std::string target = llvm::sys::getDefaultTargetTriple();
  ClangRunner runner("./toolchain/driver/run_clang_test", target, &test_os);

  std::string out;
  std::string err;
  EXPECT_TRUE(RunWithCapturedOutput(out, err,
                                    [&] { return runner.Run({"--version"}); }));
  // The arguments to Clang should be part of the verbose log.
  EXPECT_THAT(test_os.TakeStr(), HasSubstr("--version"));

  // No need to flush stderr, just check its contents.
  EXPECT_THAT(err, StrEq(""));

  // Flush and get the captured stdout to test that this command worked.
  // We don't care about any particular version, just that it is printed.
  EXPECT_THAT(out, HasSubstr("clang version"));
  // The target should match what we provided.
  EXPECT_THAT(out, HasSubstr((llvm::Twine("Target: ") + target).str()));
  // The installation should come from the above path of the test binary.
  EXPECT_THAT(out, HasSubstr("InstalledDir: ./toolchain/driver"));
}

// Utility to write a test file. We don't need the full power provided here yet,
// but we anticipate adding more tests such as compiling basic C++ code in the
// future and this provides a basis for building those tests.
static auto WriteTestFile(llvm::StringRef name_suffix, llvm::Twine contents)
    -> std::filesystem::path {
  std::filesystem::path test_tmpdir;
  if (char* tmpdir_env = getenv("TEST_TMPDIR"); tmpdir_env != nullptr) {
    test_tmpdir = std::string(tmpdir_env);
  } else {
    test_tmpdir = std::filesystem::temp_directory_path();
  }

  const auto* unit_test = ::testing::UnitTest::GetInstance();
  const auto* test_info = unit_test->current_test_info();
  std::filesystem::path test_file =
      test_tmpdir / llvm::formatv("{0}_{1}_{2}", test_info->test_suite_name(),
                                  test_info->name(), name_suffix)
                        .str();
  // Make debugging a bit easier by cleaning up any files from previous runs.
  // This is only necessary when not run in Bazel's test environment.
  std::filesystem::remove(test_file);
  CARBON_CHECK(!std::filesystem::exists(test_file));

  {
    std::error_code ec;
    llvm::raw_fd_ostream test_file_stream(test_file.string(), ec);
    CARBON_CHECK(!ec) << "Test file error: " << ec.message();
    test_file_stream << contents;
  }
  return test_file;
}

// It's hard to write a portable and reliable unittest for all the layers of the
// Clang driver because they work hard to interact with the underlying
// filesystem and operating system. For now, we just check that a link command
// is echoed back with plausible contents.
//
// TODO: We should eventually strive to have a more complete setup that lets us
// test more complete Clang functionality here.
TEST(ClangRunnerTest, LinkCommandEcho) {
  // Just create some empty files to use in a synthetic link command below.
  std::filesystem::path foo_file = WriteTestFile("foo.o", "");
  std::filesystem::path bar_file = WriteTestFile("bar.o", "");

  std::string verbose_out;
  llvm::raw_string_ostream verbose_os(verbose_out);
  std::string target = llvm::sys::getDefaultTargetTriple();
  ClangRunner runner("./toolchain/driver/run_clang_test", target, &verbose_os);
  std::string out;
  std::string err;
  EXPECT_TRUE(RunWithCapturedOutput(out, err,
                                    [&] {
                                      return runner.Run({"-###", "-o", "binary",
                                                         foo_file.string(),
                                                         bar_file.string()});
                                    }))
      << "Verbose output from runner:\n"
      << verbose_out << "\n";

  // Because we use `-###' above, we should just see the command that the Clang
  // driver would have run in a subprocess. This will be very architecture
  // dependent and have lots of variety, but we expect to see both file strings
  // in it the command at least.
  EXPECT_THAT(err, HasSubstr(foo_file.string())) << err;
  EXPECT_THAT(err, HasSubstr(bar_file.string())) << err;

  // And no non-stderr output should be produced.
  EXPECT_THAT(out, StrEq(""));
}

}  // namespace
}  // namespace Carbon
