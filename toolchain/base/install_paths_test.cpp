// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/install_paths.h"

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
  static auto GetRoot(const InstallPaths& paths) -> std::filesystem::path {
    return paths.root_;
  }
};

namespace {

using ::bazel::tools::cpp::runfiles::Runfiles;
using ::testing::_;
using ::testing::EndsWith;
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
  // the detected install root path has the expected location relative to the
  // FHS layout, and then check that the path accessors point to the right kind
  // of file or directory.
  auto TestInstallPaths(const InstallPaths& paths) -> void {
    std::filesystem::path root_path = InstallPathsTestPeer::GetRoot(paths);

    SCOPED_TRACE(llvm::formatv("Install root: '{0}'", root_path));

    // Open the root directory.
    auto root_result = Filesystem::Cwd().OpenDir(root_path);
    ASSERT_THAT(root_result, IsSuccess(_));
    Filesystem::Dir root = *std::move(root_result);

    // Check that the root is located in the expected part of the FHS layout.
    // TODO: Adjust this to work equally well on Windows.
    EXPECT_THAT(root_path.native(), EndsWith("lib/carbon/"));
    EXPECT_THAT(
        root.Access("../../bin/carbon", Filesystem::AccessCheckFlags::Execute),
        IsSuccess(Eq(true)))
        << "path: " << (root_path / "../../bin/carbon");

    std::filesystem::path core_package_path = paths.core_package();
    ASSERT_THAT(core_package_path, StartsWith(root_path));
    EXPECT_THAT(Filesystem::Cwd().Access(core_package_path / "prelude.carbon"),
                IsSuccess(Eq(true)))
        << "path: " << core_package_path;

    std::filesystem::path llvm_bin_path = paths.llvm_install_bin();
    ASSERT_THAT(llvm_bin_path, StartsWith(root_path));
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

TEST_F(InstallPathsTest, RootBusybox) {
  std::string installed_busybox_path = test_runfiles_->Rlocation(
      "carbon/toolchain/install/prefix/lib/carbon/carbon-busybox");

  auto paths = InstallPaths::MakeExeRelative(installed_busybox_path);
  ASSERT_THAT(paths.error(), Eq(std::nullopt)) << *paths.error();
  TestInstallPaths(paths);
}

TEST_F(InstallPathsTest, RootExplicit) {
  std::string marker_path = test_runfiles_->Rlocation(
      "carbon/toolchain/install/prefix/lib/carbon/carbon_install.txt");

  llvm::StringRef root_path = marker_path;
  CARBON_CHECK(root_path.consume_back("carbon_install.txt"),
               "Unexpected suffix of the marker path: {0}", marker_path);

  auto paths = InstallPaths::Make(root_path);
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
      test_runfiles_->Rlocation("carbon/toolchain/base/test_binary");
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
  EXPECT_THAT(InstallPathsTestPeer::GetRoot(paths), Eq(""));

  paths = InstallPaths::MakeExeRelative("foo/bar/baz");
  EXPECT_THAT(paths.error(), Optional(HasSubstr("foo/bar/baz")));
  EXPECT_THAT(InstallPathsTestPeer::GetRoot(paths), Eq(""));

  // Note that we can't test the runfiles code path from within a test because
  // it succeeds some of the time even with a bogus executable name.
}

}  // namespace
}  // namespace Carbon
