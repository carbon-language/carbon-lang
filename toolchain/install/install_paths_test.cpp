// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/install/install_paths.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

#include "common/check.h"
#include "common/error_test_helpers.h"
#include "common/filesystem.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadic.h"
#include "testing/base/global_exe_path.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace Carbon {

class InstallPathsTestPeer {
 public:
  static auto GetPrefix(const InstallPaths& paths) -> std::filesystem::path {
    return paths.prefix_;
  }
};

namespace {

using ::bazel::tools::cpp::runfiles::Runfiles;
using ::testing::_;
using ::testing::Eq;
using ::testing::HasSubstr;
using Testing::IsSuccess;
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
    std::filesystem::path prefix_path = InstallPathsTestPeer::GetPrefix(paths);

    SCOPED_TRACE(llvm::formatv("Install prefix: '{0}'", prefix_path));

    // Open the prefix directory.
    auto prefix_result = Filesystem::Cwd().OpenDir(prefix_path);
    ASSERT_THAT(prefix_result, IsSuccess(_));
    Filesystem::Dir prefix = *std::move(prefix_result);

    // Now check that all the expected parts of the toolchain's install are in
    // fact found using the API.
    // TODO: Adjust this to work equally well on Windows.
    EXPECT_THAT(
        prefix.Access("bin/carbon", Filesystem::AccessCheckFlags::Execute),
        IsSuccess(Eq(true)))
        << "path: " << (prefix_path / "bin/carbon");

    std::filesystem::path core_package_path = paths.core_package();
    ASSERT_THAT(core_package_path, StartsWith(prefix_path));
    EXPECT_THAT(Filesystem::Cwd().Access(core_package_path / "prelude.carbon"),
                IsSuccess(Eq(true)))
        << "path: " << core_package_path;

    std::filesystem::path llvm_bin_path = paths.llvm_install_bin();
    ASSERT_THAT(llvm_bin_path, StartsWith(prefix_path));
    auto open_result = Filesystem::Cwd().OpenDir(llvm_bin_path);
    ASSERT_THAT(open_result, IsSuccess(_));
    Filesystem::Dir llvm_bin = *std::move(open_result);

    for (std::filesystem::path bin_name : {"ld.lld", "ld64.lld"}) {
      EXPECT_THAT(
          llvm_bin.Access(bin_name, Filesystem::AccessCheckFlags::Execute),
          IsSuccess(Eq(true)))
          << "path: " << (llvm_bin_path / bin_name);
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
  std::filesystem::path test_binary_path =
      test_runfiles_->Rlocation("carbon/toolchain/install/test_binary");
  ASSERT_THAT(Filesystem::Cwd().Access(test_binary_path,
                                       Filesystem::AccessCheckFlags::Execute),
              IsSuccess(Eq(true)))
      << "path: " << test_binary_path;

  auto paths = InstallPaths::MakeForBazelRunfiles(test_binary_path.native());
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
