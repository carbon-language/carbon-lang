// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/filesystem.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <concepts>
#include <string>
#include <utility>

#include "common/error_test_helpers.h"

namespace Carbon::Filesystem {
namespace {

using ::testing::_;
using ::testing::Eq;
using ::testing::HasSubstr;
using Testing::IsError;
using Testing::IsSuccess;

class FilesystemTest : public ::testing::Test {
 public:
  explicit FilesystemTest() {
    auto result = MakeTmpDir();
    CARBON_CHECK(result.ok(), "{0}", result.error());
    dir_ = std::move(*result);
  }

  ~FilesystemTest() override {
    auto result = std::move(dir_).Remove();
    CARBON_CHECK(result.ok(), "{0}", result.error());
  }

  auto path() const -> const std::filesystem::path& { return dir_.abs_path(); }

  // The test's temp directory, deleted on destruction.
  RemovingDir dir_;
};

TEST_F(FilesystemTest, CreateOpenCloseAndUnlink) {
  auto unlink_result = dir_.Unlink("test");
  ASSERT_FALSE(unlink_result.ok());
  EXPECT_TRUE(unlink_result.error().no_entity());
#ifdef _GNU_SOURCE
  EXPECT_THAT(unlink_result, IsError(HasSubstr("ENOENT")));
#elif defined(__APPLE__) || defined(_POSIX_SOURCE)
  EXPECT_THAT(unlink_result, IsError(HasSubstr("No such file")));
#endif

  auto f = dir_.OpenWriteOnly("test", CreationOptions::CreateNew);
  ASSERT_THAT(f, IsSuccess(_));
  auto result = (*std::move(f)).Close();
  EXPECT_THAT(result, IsSuccess(_));

  f = dir_.OpenWriteOnly("test", CreationOptions::CreateNew);
  ASSERT_FALSE(f.ok());
  EXPECT_TRUE(f.error().already_exists());
#ifdef _GNU_SOURCE
  EXPECT_THAT(f, IsError(HasSubstr("EEXIST")));
#elif defined(__APPLE__) || defined(_POSIX_SOURCE)
  EXPECT_THAT(f, IsError(HasSubstr("File exists")));
#endif

  f = dir_.OpenWriteOnly("test");
  ASSERT_THAT(f, IsSuccess(_));
  result = std::move(*f).Close();
  EXPECT_THAT(result, IsSuccess(_));

  f = dir_.OpenWriteOnly("test");
  ASSERT_THAT(f, IsSuccess(_));
  result = std::move(*f).Close();
  EXPECT_THAT(result, IsSuccess(_));

  unlink_result = dir_.Unlink("test");
  EXPECT_THAT(unlink_result, IsSuccess(_));

  f = dir_.OpenWriteOnly("test");
  EXPECT_FALSE(f.ok());
  EXPECT_TRUE(f.error().no_entity());
#ifdef _GNU_SOURCE
  EXPECT_THAT(f, IsError(HasSubstr("ENOENT")));
#elif defined(__APPLE__) || defined(_POSIX_SOURCE)
  EXPECT_THAT(f, IsError(HasSubstr("No such file")));
#endif

  f = dir_.OpenWriteOnly("test", CreationOptions::OpenAlways);
  ASSERT_THAT(f, IsSuccess(_));
  result = std::move(*f).Close();
  EXPECT_THAT(result, IsSuccess(_));

  unlink_result = dir_.Unlink("test");
  EXPECT_THAT(unlink_result, IsSuccess(_));
}

TEST_F(FilesystemTest, BasicWriteAndRead) {
  std::string content_str = "0123456789";
  {
    auto f = dir_.OpenWriteOnly("test", CreationOptions::CreateNew);
    ASSERT_THAT(f, IsSuccess(_));
    auto write_result = f->WriteFromString(content_str);
    EXPECT_THAT(write_result, IsSuccess(_));
    (*std::move(f)).Close().Check();
  }

  {
    auto f = dir_.OpenReadOnly("test");
    ASSERT_THAT(f, IsSuccess(_));
    auto read_result = f->ReadToString();
    EXPECT_THAT(read_result, IsSuccess(Eq(content_str)));
  }

  auto unlink_result = dir_.Unlink("test");
  EXPECT_THAT(unlink_result, IsSuccess(_));
}

TEST_F(FilesystemTest, CreateAndRemoveDirecotries) {
  auto d1 = Cwd().CreateDirectories(path() / "a" / "b" / "c" / "test1");
  ASSERT_THAT(d1, IsSuccess(_));
  auto d2 = Cwd().CreateDirectories(path() / "a" / "b" / "c" / "test2");
  ASSERT_THAT(d2, IsSuccess(_));
  auto d3 = Cwd().CreateDirectories(path() / "a" / "b" / "c" / "test3");
  ASSERT_THAT(d3, IsSuccess(_));
  // Get a directory object to use, this shouldn't cover much new.
  auto d4 = Cwd().CreateDirectories(path());
  EXPECT_THAT(d4, IsSuccess(_));
  // Single, present, relative component.
  auto d5 = d4->CreateDirectories("a");
  EXPECT_THAT(d5, IsSuccess(_));
  // Multiple, present, but relative components.
  auto d6 = d5->CreateDirectories(std::filesystem::path("b") / "c");
  EXPECT_THAT(d6, IsSuccess(_));
  // Single new component.
  auto d7 = d6->CreateDirectories("test4");
  ASSERT_THAT(d7, IsSuccess(_));
  // Two new relative components.
  auto d8 = d6->CreateDirectories(std::filesystem::path("test5") / "d");
  EXPECT_THAT(d8, IsSuccess(_));
  // Mixed relative components.
  auto d9 = d5->CreateDirectories(std::filesystem::path("b") / "test6");
  EXPECT_THAT(d9, IsSuccess(_));

  {
    auto f1 = d1->OpenWriteOnly("file1", CreateNew);
    ASSERT_THAT(f1, IsSuccess(_));
    auto f2 = d2->OpenWriteOnly("file2", CreateNew);
    ASSERT_THAT(f2, IsSuccess(_));
    auto f3 = d3->OpenWriteOnly("file3", CreateNew);
    ASSERT_THAT(f3, IsSuccess(_));
    auto f4 = d7->OpenWriteOnly("file4", CreateNew);
    ASSERT_THAT(f4, IsSuccess(_));
    (*std::move(f1)).Close().Check();
    (*std::move(f2)).Close().Check();
    (*std::move(f3)).Close().Check();
    (*std::move(f4)).Close().Check();
  }

  auto rm_result = Cwd().Rmtree(path() / "a");
  ASSERT_THAT(rm_result, IsSuccess(_));
}

TEST_F(FilesystemTest, StatAndAccess) {
  auto access_result = dir_.Access("test");
  ASSERT_FALSE(access_result.ok());
  EXPECT_TRUE(access_result.error().no_entity());

  // Make sure the flags and bit-or-ing them works in the boring case.
  access_result =
      dir_.Access("test", AccessCheckFlags::Read | AccessCheckFlags::Write |
                              AccessCheckFlags::Execute);
  ASSERT_FALSE(access_result.ok());
  EXPECT_TRUE(access_result.error().no_entity());

  auto stat_result = dir_.Stat("test");
  ASSERT_FALSE(access_result.ok());
  EXPECT_TRUE(access_result.error().no_entity());

  // Create a file for testing, using very unusual and minimal permissions to
  // help us test. Hopefully this isn't modified on the usual `umask` tests run
  // under.
  std::string content_str = "0123456789";
  ModeType permissions = 0450;
  auto f = dir_.OpenWriteOnly("test", CreationOptions::CreateNew, permissions);
  ASSERT_THAT(f, IsSuccess(_));
  auto write_result = f->WriteFromString(content_str);
  EXPECT_THAT(write_result, IsSuccess(_));

  access_result = dir_.Access("test");
  EXPECT_THAT(access_result, IsSuccess(_));
  access_result = dir_.Access("test", AccessCheckFlags::Read);
  EXPECT_THAT(access_result, IsSuccess(_));

  // Neither write nor execute permission should be present though.
  access_result = dir_.Access("test", AccessCheckFlags::Write);
  ASSERT_FALSE(access_result.ok());
  EXPECT_TRUE(access_result.error().access_denied());
  access_result =
      dir_.Access("test", AccessCheckFlags::Read | AccessCheckFlags::Write |
                              AccessCheckFlags::Execute);
  ASSERT_FALSE(access_result.ok());
  EXPECT_TRUE(access_result.error().access_denied());

  stat_result = dir_.Stat("test");
  ASSERT_THAT(stat_result, IsSuccess(_));
  EXPECT_TRUE(stat_result->is_file());
  EXPECT_FALSE(stat_result->is_dir());
  EXPECT_FALSE(stat_result->is_symlink());
  EXPECT_THAT(stat_result->size(), Eq(content_str.size()));
  EXPECT_THAT(stat_result->permissions(), Eq(permissions));

  // Directory instead of file.
  access_result =
      dir_.Access(".", AccessCheckFlags::Read | AccessCheckFlags::Write |
                           AccessCheckFlags::Execute);
  EXPECT_THAT(access_result, IsSuccess(_));

  stat_result = dir_.Stat(".");
  ASSERT_THAT(stat_result, IsSuccess(_));
  EXPECT_FALSE(stat_result->is_file());
  EXPECT_TRUE(stat_result->is_dir());
  EXPECT_FALSE(stat_result->is_symlink());

  // Can remove file but still stat through the file.
  auto unlink_result = dir_.Unlink("test");
  ASSERT_THAT(unlink_result, IsSuccess(_));
  auto file_stat_result = f->Stat();
  ASSERT_THAT(file_stat_result, IsSuccess(_));
  EXPECT_TRUE(file_stat_result->is_file());
  EXPECT_FALSE(file_stat_result->is_dir());
  EXPECT_FALSE(file_stat_result->is_symlink());
  EXPECT_THAT(file_stat_result->size(), Eq(content_str.size()));
  EXPECT_THAT(file_stat_result->permissions(), Eq(permissions));
  (*std::move(f)).Close().Check();
}

TEST_F(FilesystemTest, Symlinks) {
  auto readlink_result = dir_.Readlink("test");
  ASSERT_FALSE(readlink_result.ok());
  EXPECT_TRUE(readlink_result.error().no_entity());

  auto lstat_result = dir_.Lstat("test");
  ASSERT_FALSE(lstat_result.ok());
  EXPECT_TRUE(lstat_result.error().no_entity());

  auto symlink_result = dir_.Symlink("test", "abc");
  EXPECT_THAT(symlink_result, IsSuccess(_));

  readlink_result = dir_.Readlink("test");
  EXPECT_THAT(readlink_result, IsSuccess(Eq("abc")));

  symlink_result = dir_.Symlink("test", "def");
  ASSERT_FALSE(symlink_result.ok());
  EXPECT_TRUE(symlink_result.error().already_exists());

  lstat_result = dir_.Lstat("test");
  ASSERT_THAT(lstat_result, IsSuccess(_));
  EXPECT_FALSE(lstat_result->is_file());
  EXPECT_FALSE(lstat_result->is_dir());
  EXPECT_TRUE(lstat_result->is_symlink());
  EXPECT_THAT(lstat_result->size(), Eq(strlen("abc")));

  auto unlink_result = dir_.Unlink("test");
  EXPECT_THAT(unlink_result, IsSuccess(_));

  readlink_result = dir_.Readlink("test");
  ASSERT_FALSE(readlink_result.ok());
  EXPECT_TRUE(readlink_result.error().no_entity());

  // Try a symlink with null bytes for fun. This demonstrates that the symlink
  // syscall only uses the leading C-string.
  symlink_result = dir_.Symlink("test", std::string("a\0b\0c", 5));
  EXPECT_THAT(symlink_result, IsSuccess(_));
  readlink_result = dir_.Readlink("test");
  EXPECT_THAT(readlink_result, IsSuccess(Eq("a")));
}

TEST_F(FilesystemTest, Chdir) {
  auto current_result = Cwd().OpenDir(".");
  ASSERT_THAT(current_result, IsSuccess(_));

  auto symlink_result = dir_.Symlink("test", "abc");
  EXPECT_THAT(symlink_result, IsSuccess(_));

  auto chdir_result = dir_.Chdir();
  EXPECT_THAT(chdir_result, IsSuccess(_));
  auto readlink_result = Cwd().Readlink("test");
  EXPECT_THAT(readlink_result, IsSuccess(Eq("abc")));

  auto chdir_path_result = dir_.Chdir("missing");
  ASSERT_FALSE(chdir_path_result.ok());
  EXPECT_TRUE(chdir_path_result.error().no_entity());

  // Dangling symlink.
  chdir_path_result = dir_.Chdir("test");
  ASSERT_FALSE(chdir_path_result.ok());
  EXPECT_TRUE(chdir_path_result.error().no_entity());

  // Create a regular file and try to chdir to that.
  auto f = dir_.OpenWriteOnly("test2", CreationOptions::CreateNew);
  ASSERT_THAT(f, IsSuccess(_));
  auto write_result = f->WriteFromString("test2");
  EXPECT_THAT(write_result, IsSuccess(_));
  chdir_path_result = dir_.Chdir("test2");
  ASSERT_FALSE(chdir_path_result.ok());
  EXPECT_TRUE(chdir_path_result.error().not_dir());

  auto d2_result = Cwd().OpenDir("test_d2", CreationOptions::CreateNew);
  ASSERT_THAT(d2_result, IsSuccess(_));
  symlink_result = d2_result->Symlink("test2", "def");
  EXPECT_THAT(symlink_result, IsSuccess(_));

  chdir_path_result = dir_.Chdir("test_d2");
  ASSERT_THAT(chdir_path_result, IsSuccess(_));
  readlink_result = Cwd().Readlink("test2");
  EXPECT_THAT(readlink_result, IsSuccess(Eq("def")));
  readlink_result = Cwd().Readlink("../test");
  EXPECT_THAT(readlink_result, IsSuccess(Eq("abc")));

  chdir_result = current_result->Chdir();
  ASSERT_THAT(chdir_result, IsSuccess(_));
  readlink_result = Cwd().Readlink("test");
  ASSERT_FALSE(readlink_result.ok());
  EXPECT_TRUE(readlink_result.error().no_entity());
  (*std::move(f)).Close().Check();
}

}  // namespace
}  // namespace Carbon::Filesystem
