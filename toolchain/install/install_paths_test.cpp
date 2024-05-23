// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/install/install_paths.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "testing/base/gtest_main.h"

namespace Carbon {
namespace {

using ::testing::StartsWith;

static auto GetTestSrcdir() -> std::optional<std::string> {
  if (char* test_srcdir = getenv("TEST_SRCDIR")) {
    return std::string(test_srcdir);
  }
  return std::nullopt;
}

class InstallationTest : public ::testing::Test {
 protected:
  InstallationTest() : test_srcdir(GetTestSrcdir()) {}

  // Compute the test runfiles path, either using Bazel's `TEST_SRCDIR`
  // environment variable or the name of the test executable. Trying both of
  // these allows tests to be run directly out of the Bazel build tree, for
  // example by a debugger, and still find their runfiles.
  //
  // TODO: Extract this to a testing helper library and use it more broadly.
  auto FindTestRunfiles() -> std::string {
    if (test_srcdir && llvm::sys::fs::is_directory(*test_srcdir)) {
      return *test_srcdir;
    }

    return Testing::GetTestExePath().str() + ".runfiles";
  }

  // Test the install paths found with the given `exe_path`. Will check that the
  // detected install prefix path starts with `prefix_startswith`, and then
  // check that the path accessors point to the right kind of file or directory.
  auto TestInstallPaths(llvm::StringRef exe_path,
                        llvm::StringRef prefix_startswith) -> void {
    SCOPED_TRACE(llvm::formatv("Executable path: '%s'", exe_path));
    InstallPaths paths(exe_path, &llvm::errs());

    // Grab a the prefix into a string to make it easier to use in the test.
    std::string prefix = paths.prefix().str();
    EXPECT_THAT(prefix, StartsWith(prefix_startswith));
    SCOPED_TRACE(llvm::formatv("Install prefix path: '%s'", prefix));
    EXPECT_TRUE(llvm::sys::fs::exists(prefix));
    EXPECT_TRUE(llvm::sys::fs::is_directory(prefix));

    // Now check that all the expected parts of the toolchain's install are in
    // fact found using the API.
    std::string driver_path = paths.driver();
    ASSERT_THAT(driver_path, StartsWith(prefix));
    EXPECT_TRUE(llvm::sys::fs::exists(driver_path)) << "path: " << driver_path;
    EXPECT_TRUE(llvm::sys::fs::can_execute(driver_path))
        << "path: " << driver_path;

    std::string llvm_bin_path = paths.llvm_install_bin();
    ASSERT_THAT(llvm_bin_path, StartsWith(prefix));
    EXPECT_TRUE(llvm::sys::fs::exists(llvm_bin_path))
        << "path: " << llvm_bin_path;
    EXPECT_TRUE(llvm::sys::fs::is_directory(llvm_bin_path))
        << "path: " << llvm_bin_path;

    for (llvm::StringRef llvm_bin :
         {"lld", "ld.lld", "ld64.lld", "lld-link", "wasm-ld"}) {
      llvm::SmallString<128> bin_path;
      bin_path.assign(llvm_bin_path);
      llvm::sys::path::append(bin_path, llvm_bin);

      EXPECT_TRUE(llvm::sys::fs::exists(bin_path)) << "path: " << bin_path;
      EXPECT_TRUE(llvm::sys::fs::can_execute(bin_path)) << "path: " << bin_path;
    }
  }

  // When run as a Bazel test, the `TEST_SRCDIR` environment variable.
  std::optional<std::string> test_srcdir;
};

TEST_F(InstallationTest, Installations) {
  // Use synthetic install trees to test detection of various patterns.
  // Each of these trees is identified by a specific executable path.

  // First, test a simulated install using the driver's executable path.
  std::string runfiles = FindTestRunfiles();
  llvm::SmallString<128> test_installed_root = llvm::StringRef(runfiles);
  llvm::sys::path::append(test_installed_root, llvm::sys::path::Style::posix,
                          "_main/toolchain/install/test_installed_root/");
  llvm::SmallString<128> installed_driver = test_installed_root;
  llvm::sys::path::append(installed_driver, llvm::sys::path::Style::posix,
                          "bin/carbon");
  ASSERT_TRUE(llvm::sys::fs::can_execute(installed_driver))
      << "Driver path: " << installed_driver;
  TestInstallPaths(installed_driver, test_installed_root);

  // We simulate direct execution of a just-built binary by synthesizing a
  // similar layout to `bazel-bin` and the runfiles tree path used there.
  llvm::SmallString<128> test_binary = llvm::StringRef(runfiles);
  llvm::sys::path::append(
      test_binary, llvm::sys::path::Style::posix,
      "_main/toolchain/install/test_direct_exec_root/test_binary");
  ASSERT_TRUE(llvm::sys::fs::can_execute(test_binary))
      << "Test binary path: " << test_binary;
  // We expect the just-built binary path to be a string prefix of the detected
  // install prefix, so we pass the path twice here.
  TestInstallPaths(test_binary, test_binary);

  // If we have `TEST_SRCDIR`, also check that it works by using a nonsense
  // executable path.
  if (test_srcdir) {
    TestInstallPaths("test_exe", *test_srcdir);
  }
}

}  // namespace
}  // namespace Carbon
