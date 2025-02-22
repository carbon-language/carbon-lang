// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/install/install_paths.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/check.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "testing/base/global_exe_path.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace Carbon {

class InstallPathsTestPeer {
 public:
  static auto GetPrefix(const InstallPaths& paths) -> llvm::StringRef {
    return paths.prefix_;
  }
};

namespace {

using ::bazel::tools::cpp::runfiles::Runfiles;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::testing::StartsWith;

class InstallPathsTest : public ::testing::Test {
 public:
  InstallPathsTest() {
    std::string error;
    test_runfiles_.reset(Runfiles::Create(Testing::GetExePath().str(), &error));
    CARBON_CHECK(test_runfiles_ != nullptr, "{0}", error);
  }

  // Test the install paths found with the given `exe_path`. Will check that
  // the detected install prefix path starts with `prefix_startswith`, and then
  // check that the path accessors point to the right kind of file or
  // directory.
  auto TestInstallPaths(const InstallPaths& paths) -> void {
    auto prefix = InstallPathsTestPeer::GetPrefix(paths);

    SCOPED_TRACE(llvm::formatv("Install prefix: '{0}'", prefix));

    // Grab a the prefix into a string to make it easier to use in the test.
    EXPECT_TRUE(llvm::sys::fs::exists(prefix));
    EXPECT_TRUE(llvm::sys::fs::is_directory(prefix));

    // Now check that all the expected parts of the toolchain's install are in
    // fact found using the API.
    llvm::SmallString<256> driver_path(prefix);
    // TODO: Adjust this to work equally well on Windows.
    llvm::sys::path::append(driver_path, llvm::sys::path::Style::posix,
                            "bin/carbon");
    EXPECT_TRUE(llvm::sys::fs::exists(driver_path)) << "path: " << driver_path;
    EXPECT_TRUE(llvm::sys::fs::can_execute(driver_path))
        << "path: " << driver_path;

    std::string core_package_path = paths.core_package();
    ASSERT_THAT(core_package_path, StartsWith(prefix));
    EXPECT_TRUE(llvm::sys::fs::exists(core_package_path + "/prelude.carbon"))
        << "path: " << core_package_path;

    std::string llvm_bin_path = paths.llvm_install_bin();
    ASSERT_THAT(llvm_bin_path, StartsWith(prefix));
    EXPECT_TRUE(llvm::sys::fs::exists(llvm_bin_path))
        << "path: " << llvm_bin_path;
    EXPECT_TRUE(llvm::sys::fs::is_directory(llvm_bin_path))
        << "path: " << llvm_bin_path;

    for (llvm::StringRef llvm_bin : {"ld.lld", "ld64.lld"}) {
      llvm::SmallString<128> bin_path;
      bin_path.assign(llvm_bin_path);
      llvm::sys::path::append(bin_path, llvm_bin);

      EXPECT_TRUE(llvm::sys::fs::exists(bin_path)) << "path: " << bin_path;
      EXPECT_TRUE(llvm::sys::fs::can_execute(bin_path)) << "path: " << bin_path;
    }
  }

  std::unique_ptr<Runfiles> test_runfiles_;
};

TEST_F(InstallPathsTest, PrefixRootBusybox) {
  std::string installed_driver_path = test_runfiles_->Rlocation(
      "carbon/toolchain/install/prefix_root/lib/carbon/carbon-busybox");

  auto paths = InstallPaths::MakeExeRelative(installed_driver_path);
  ASSERT_THAT(paths.error(), Eq(std::nullopt)) << *paths.error();
  TestInstallPaths(paths);
}

TEST_F(InstallPathsTest, PrefixRootExplicit) {
  std::string marker_path = test_runfiles_->Rlocation(
      "carbon/toolchain/install/prefix_root/lib/carbon/carbon_install.txt");

  llvm::StringRef prefix_path = marker_path;
  CARBON_CHECK(prefix_path.consume_back("lib/carbon/carbon_install.txt"),
               "Unexpected suffix of the marker path: {0}", marker_path);

  auto paths = InstallPaths::Make(prefix_path);
  ASSERT_THAT(paths.error(), Eq(std::nullopt)) << *paths.error();
  TestInstallPaths(paths);
}

TEST_F(InstallPathsTest, TestRunfiles) {
  auto paths = InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());
  ASSERT_THAT(paths.error(), Eq(std::nullopt)) << *paths.error();
  TestInstallPaths(paths);
}

TEST_F(InstallPathsTest, BinaryRunfiles) {
  std::string test_binary_path =
      test_runfiles_->Rlocation("carbon/toolchain/install/test_binary");
  CARBON_CHECK(llvm::sys::fs::can_execute(test_binary_path), "{0}",
               test_binary_path);

  auto paths = InstallPaths::MakeForBazelRunfiles(test_binary_path);
  ASSERT_THAT(paths.error(), Eq(std::nullopt)) << *paths.error();
  TestInstallPaths(paths);
}

TEST_F(InstallPathsTest, Errors) {
  auto paths = InstallPaths::Make("/foo/bar/baz");
  EXPECT_THAT(paths.error(), Optional(HasSubstr("foo/bar/baz")));
  EXPECT_THAT(InstallPathsTestPeer::GetPrefix(paths), Eq(""));

  paths = InstallPaths::MakeExeRelative("foo/bar/baz");
  EXPECT_THAT(paths.error(), Optional(HasSubstr("foo/bar/baz")));
  EXPECT_THAT(InstallPathsTestPeer::GetPrefix(paths), Eq(""));

  // Note that we can't test the runfiles code path from within a test because
  // it succeeds some of the time even with a bogus executable name.
}

}  // namespace
}  // namespace Carbon
