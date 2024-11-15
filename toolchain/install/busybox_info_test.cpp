// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/install/busybox_info.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>

#include "common/check.h"

namespace Carbon {
namespace {

using ::testing::Eq;

class BusyboxInfoTest : public ::testing::Test {
 protected:
  // Set up a temp directory for the test case.
  explicit BusyboxInfoTest() {
    const char* tmpdir = std::getenv("TEST_TMPDIR");
    CARBON_CHECK(tmpdir);
    dir_ = MakeDir(std::filesystem::absolute(
        std::filesystem::path(tmpdir) /
        ::testing::UnitTest::GetInstance()->current_test_info()->name()));
  }

  // Delete the test case's temp directory.
  ~BusyboxInfoTest() override {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
    CARBON_CHECK(!ec, "error removing {0}: {1}", dir_, ec.message());
  }

  // Creates a stub file. Returns the input file for easier use.
  auto MakeFile(std::filesystem::path file) -> std::filesystem::path {
    std::ofstream out(file.c_str());
    out << "stub";
    CARBON_CHECK(out, "error creating {0}", file);
    return file;
  }

  // Creates a symlink to the target. Returns the input file for easier use.
  auto MakeSymlink(std::filesystem::path file, auto target)
      -> std::filesystem::path {
    std::error_code ec;
    std::filesystem::create_symlink(target, file, ec);
    CARBON_CHECK(!ec, "error creating {0}: {1}", file, ec.message());
    return file;
  }

  // Creates a directory. Returns the input file for easier use.
  auto MakeDir(std::filesystem::path dir) -> std::filesystem::path {
    std::error_code ec;
    std::filesystem::create_directory(dir, ec);
    CARBON_CHECK(!ec, "error creating {0}: {1}", dir, ec.message());
    return dir;
  }

  // The test's temp directory, deleted on destruction.
  std::filesystem::path dir_;
};

TEST_F(BusyboxInfoTest, Direct) {
  auto busybox = MakeFile(dir_ / "carbon-busybox");

  auto info = GetBusyboxInfo(busybox.string());
  ASSERT_TRUE(info.ok()) << info.error();
  EXPECT_THAT(info->bin_path, Eq(busybox));
  EXPECT_THAT(info->mode, Eq(std::nullopt));
}

TEST_F(BusyboxInfoTest, SymlinkInCurrentDirectory) {
  MakeFile(dir_ / "carbon-busybox");
  auto target = MakeSymlink(dir_ / "carbon", "carbon-busybox");

  auto info = GetBusyboxInfo(target.string());
  ASSERT_TRUE(info.ok()) << info.error();
  EXPECT_THAT(info->bin_path, Eq(dir_ / "carbon-busybox"));
  EXPECT_THAT(info->mode, Eq("carbon"));
}

TEST_F(BusyboxInfoTest, SymlinkInCurrentDirectoryWithDot) {
  MakeFile(dir_ / "carbon-busybox");
  auto target = MakeSymlink(dir_ / "carbon", "./carbon-busybox");

  auto info = GetBusyboxInfo(target.string());
  ASSERT_TRUE(info.ok()) << info.error();
  EXPECT_THAT(info->bin_path, Eq(dir_ / "./carbon-busybox"));
  EXPECT_THAT(info->mode, Eq("carbon"));
}

TEST_F(BusyboxInfoTest, ExtraSymlink) {
  MakeFile(dir_ / "carbon-busybox");
  MakeSymlink(dir_ / "carbon", "carbon-busybox");
  auto target = MakeSymlink(dir_ / "c", "carbon");

  auto info = GetBusyboxInfo(target.string());
  ASSERT_TRUE(info.ok()) << info.error();
  EXPECT_THAT(info->bin_path, Eq(dir_ / "carbon-busybox"));
  EXPECT_THAT(info->mode, Eq("carbon"));
}

TEST_F(BusyboxInfoTest, BusyboxIsSymlink) {
  MakeFile(dir_ / "actual-busybox");
  auto target = MakeSymlink(dir_ / "carbon-busybox", "actual-busybox");

  auto info = GetBusyboxInfo(target.string());
  ASSERT_TRUE(info.ok()) << info.error();
  EXPECT_THAT(info->bin_path, Eq(target));
  EXPECT_THAT(info->mode, Eq(std::nullopt));
}

TEST_F(BusyboxInfoTest, BusyboxIsSymlinkToNowhere) {
  auto target = MakeSymlink(dir_ / "carbon-busybox", "nonexistent");

  auto info = GetBusyboxInfo(target.string());
  ASSERT_TRUE(info.ok()) << info.error();
  EXPECT_THAT(info->bin_path, Eq(dir_ / "carbon-busybox"));
  EXPECT_THAT(info->mode, Eq(std::nullopt));
}

TEST_F(BusyboxInfoTest, RelativeSymlink) {
  MakeDir(dir_ / "lib");
  MakeDir(dir_ / "lib/carbon");
  MakeFile(dir_ / "lib/carbon/carbon-busybox");
  MakeDir(dir_ / "bin");
  auto target =
      MakeSymlink(dir_ / "bin/carbon", "../lib/carbon/carbon-busybox");

  auto info = GetBusyboxInfo(target.string());
  ASSERT_TRUE(info.ok()) << info.error();
  EXPECT_THAT(info->bin_path, Eq(dir_ / "bin/../lib/carbon/carbon-busybox"));
  EXPECT_THAT(info->mode, Eq("carbon"));
}

TEST_F(BusyboxInfoTest, AbsoluteSymlink) {
  MakeDir(dir_ / "lib");
  MakeDir(dir_ / "lib/carbon");
  auto busybox = MakeFile(dir_ / "lib/carbon/carbon-busybox");
  ASSERT_TRUE(busybox.is_absolute());
  MakeDir(dir_ / "bin");
  auto target = MakeSymlink(dir_ / "bin/carbon", busybox);

  auto info = GetBusyboxInfo(target.string());
  ASSERT_TRUE(info.ok()) << info.error();
  EXPECT_THAT(info->bin_path, Eq(busybox));
  EXPECT_THAT(info->mode, Eq("carbon"));
}

TEST_F(BusyboxInfoTest, NotBusyboxFile) {
  auto target = MakeFile(dir_ / "file");

  auto info = GetBusyboxInfo(target.string());
  EXPECT_FALSE(info.ok());
}

TEST_F(BusyboxInfoTest, NotBusyboxSymlink) {
  MakeFile(dir_ / "file");
  auto target = MakeSymlink(dir_ / "carbon", "file");

  auto info = GetBusyboxInfo(target.string());
  EXPECT_FALSE(info.ok());
}

}  // namespace
}  // namespace Carbon
