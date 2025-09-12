// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/filesystem.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <concepts>
#include <string>
#include <thread>
#include <utility>

#include "common/error_test_helpers.h"

namespace Carbon::Filesystem {
namespace {

using ::testing::_;
using ::testing::Eq;
using ::testing::HasSubstr;
using Testing::IsError;
using Testing::IsSuccess;
using ::testing::UnorderedElementsAre;

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
#if defined(_GNU_SOURCE) && \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 32))
  EXPECT_THAT(unlink_result, IsError(HasSubstr("ENOENT")));
#endif
  EXPECT_THAT(unlink_result, IsError(HasSubstr("No such file")));

  auto f = dir_.OpenWriteOnly("test", CreationOptions::CreateNew);
  ASSERT_THAT(f, IsSuccess(_));
  auto result = (*std::move(f)).Close();
  EXPECT_THAT(result, IsSuccess(_));

  f = dir_.OpenWriteOnly("test", CreationOptions::CreateNew);
  ASSERT_FALSE(f.ok());
  EXPECT_TRUE(f.error().already_exists());
#if defined(_GNU_SOURCE) && \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 32))
  EXPECT_THAT(f, IsError(HasSubstr("EEXIST")));
#endif
  EXPECT_THAT(f, IsError(HasSubstr("File exists")));

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
#if defined(_GNU_SOURCE) && \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 32))
  EXPECT_THAT(f, IsError(HasSubstr("ENOENT")));
#endif
  EXPECT_THAT(f, IsError(HasSubstr("No such file")));

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
    auto write_result = f->WriteFileFromString(content_str);
    EXPECT_THAT(write_result, IsSuccess(_));
    (*std::move(f)).Close().Check();
  }

  {
    auto f = dir_.OpenReadOnly("test");
    ASSERT_THAT(f, IsSuccess(_));
    auto read_result = f->ReadFileToString();
    EXPECT_THAT(read_result, IsSuccess(Eq(content_str)));
  }

  auto unlink_result = dir_.Unlink("test");
  EXPECT_THAT(unlink_result, IsSuccess(_));
}

TEST_F(FilesystemTest, SeekReadAndWrite) {
  std::string content_str = "0123456789";
  // First write some initial content.
  {
    auto f = dir_.OpenWriteOnly("test", CreationOptions::CreateNew);
    ASSERT_THAT(f, IsSuccess(_));
    auto write_result = f->WriteFileFromString(content_str);
    EXPECT_THAT(write_result, IsSuccess(_));
    (*std::move(f)).Close().Check();
  }

  // Now seek and read.
  {
    auto f = dir_.OpenReadOnly("test");
    ASSERT_THAT(f, IsSuccess(_));
    auto seek_result = f->Seek(3);
    ASSERT_THAT(seek_result, IsSuccess(Eq(3)));
    std::array<std::byte, 4> buffer;
    auto read_result = f->ReadToBuffer(buffer);
    ASSERT_THAT(read_result, IsSuccess(_));
    EXPECT_THAT(std::string(reinterpret_cast<char*>(read_result->data()),
                            read_result->size()),
                Eq(content_str.substr(3, read_result->size())));

    // Now test that we can seek back to the beginning and read the full file.
    auto read_file_result = f->ReadFileToString();
    EXPECT_THAT(read_file_result, IsSuccess(Eq(content_str)));
  }

  // Now a mixture of reads, writes, an seeking.
  {
    auto f = dir_.OpenReadWrite("test");
    ASSERT_THAT(f, IsSuccess(_));
    auto seek_result = f->SeekFromEnd(-6);
    ASSERT_THAT(seek_result, IsSuccess(Eq(content_str.size() - 6)));
    std::string new_content_str = "abcdefg";
    llvm::ArrayRef<std::byte> new_content_bytes(
        reinterpret_cast<std::byte*>(new_content_str.data()),
        new_content_str.size());
    for (auto write_bytes = new_content_bytes.slice(0, 4);
         !write_bytes.empty();) {
      auto write_result = f->WriteFromBuffer(write_bytes);
      ASSERT_THAT(write_result, IsSuccess(_));
      write_bytes = *write_result;
    }

    std::array<std::byte, 4> buffer;
    auto read_result = f->ReadToBuffer(buffer);
    ASSERT_THAT(read_result, IsSuccess(_));
    EXPECT_THAT(std::string(reinterpret_cast<char*>(read_result->data()),
                            read_result->size()),
                Eq(content_str.substr(8, read_result->size())));

    EXPECT_THAT(*f->ReadFileToString(), "0123abcd89");

    // Now write the entire file, also changing its size, after a fresh seek.
    seek_result = f->Seek(-6);
    ASSERT_THAT(seek_result, IsSuccess(Eq(content_str.size() - 6)));
    auto write_file_result = f->WriteFileFromString(new_content_str);
    EXPECT_THAT(write_file_result, IsSuccess(_));
    EXPECT_THAT(*f->ReadFileToString(), "abcdefg");
    (*std::move(f)).Close().Check();
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
  auto write_result = f->WriteFileFromString(content_str);
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
  auto write_result = f->WriteFileFromString("test2");
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

TEST_F(FilesystemTest, WriteStream) {
  std::string content_str = "0123456789";
  auto write = dir_.OpenWriteOnly("test", CreationOptions::CreateNew);
  ASSERT_THAT(write, IsSuccess(_));
  {
    llvm::raw_fd_ostream os = write->WriteStream();
    os << content_str;
    EXPECT_FALSE(os.has_error()) << os.error();
  }
  (*std::move(write)).Close().Check();

  EXPECT_THAT(dir_.ReadFileToString("test"), IsSuccess(Eq(content_str)));
}

TEST_F(FilesystemTest, Rename) {
  // Rename a file within a directory.
  ASSERT_THAT(dir_.WriteFileFromString("file1", "content1"), IsSuccess(_));
  EXPECT_THAT(dir_.Rename("file1", dir_, "file2"), IsSuccess(_));
  EXPECT_THAT(dir_.ReadFileToString("file2"), IsSuccess(Eq("content1")));
  auto read_missing = dir_.ReadFileToString("file1");
  EXPECT_FALSE(read_missing.ok());
  EXPECT_TRUE(read_missing.error().no_entity());

  // Rename a file between two directories.
  auto d1 = *dir_.CreateDirectories("subdir1");
  EXPECT_THAT(dir_.Rename("file2", d1, "file1"), IsSuccess(_));
  EXPECT_THAT(d1.ReadFileToString("file1"), IsSuccess(Eq("content1")));
  auto d2 = *dir_.CreateDirectories("subdir2");
  EXPECT_THAT(d1.Rename("file1", d2, "file1"), IsSuccess(_));
  EXPECT_THAT(d2.ReadFileToString("file1"), IsSuccess(Eq("content1")));
  // Close the first directory.
  d1 = Filesystem::Dir();
  EXPECT_THAT(dir_.Rmdir("subdir1"), IsSuccess(_))
      << "Directory should have bene empty!";

  // Rename directories.
  ASSERT_THAT(dir_.ReadFileToString(std::filesystem::path("subdir2") / "file1"),
              IsSuccess(Eq("content1")));
  EXPECT_THAT(dir_.Rename("subdir2", dir_, "subdir1"), IsSuccess(_));
  EXPECT_THAT(dir_.ReadFileToString(std::filesystem::path("subdir1") / "file1"),
              IsSuccess(Eq("content1")));

  // The open directory `d2` should survive the rename and point at the same
  // directory.
  EXPECT_THAT(d2.ReadFileToString("file1"), IsSuccess(Eq("content1")));
  EXPECT_THAT(d2.WriteFileFromString("file2", "content2"), IsSuccess(_));
  EXPECT_THAT(dir_.ReadFileToString(std::filesystem::path("subdir1") / "file2"),
              IsSuccess(Eq("content2")));

  // Rename over an existing file.
  EXPECT_THAT(d2.Rename("file2", d2, "file1"), IsSuccess(_));
  EXPECT_THAT(d2.ReadFileToString("file1"), IsSuccess(Eq("content2")));

  // Test error calls as well.
  auto result = dir_.Rename("missing1", dir_, "missing2");
  EXPECT_TRUE(result.error().no_entity()) << result.error();
  result = d2.Rename("file1", dir_,
                     std::filesystem::path("missing_subdir") / "file2");
  EXPECT_TRUE(result.error().no_entity()) << result.error();
  // Note that `d2` was renamed `subdir1` above, which is why this creates
  // infinite subdirectories.
  result = dir_.Rename("subdir1", d2, "infinite_subdirs");
  EXPECT_THAT(result.error().unix_errnum(), EINVAL) << result.error();
}

TEST_F(FilesystemTest, TryLock) {
  auto file = dir_.OpenReadWrite("test_file", CreateNew);
  ASSERT_THAT(file, IsSuccess(_));

  // Acquire an exclusive lock.
  auto lock = file->TryLock(FileLock::Exclusive);
  ASSERT_THAT(lock, IsSuccess(_));
  EXPECT_TRUE(lock->is_locked());

  // Try to acquire a second lock from a different file object.
  auto file2 = dir_.OpenReadOnly("test_file");
  ASSERT_THAT(file2, IsSuccess(_));
  auto lock2 = file2->TryLock(FileLock::Exclusive);
  ASSERT_THAT(lock2, IsError(_));
  EXPECT_TRUE(lock2.error().would_block());

  // A shared lock should also fail.
  auto lock3 = file2->TryLock(FileLock::Shared);
  ASSERT_THAT(lock3, IsError(_));
  EXPECT_TRUE(lock3.error().would_block());

  // Release the first lock.
  *lock = {};
  EXPECT_FALSE(lock->is_locked());

  // Now we can acquire an exclusive lock.
  lock2 = file2->TryLock(FileLock::Exclusive);
  ASSERT_THAT(lock2, IsSuccess(_));
  EXPECT_TRUE(lock2->is_locked());
  *lock2 = {};

  // Test shared locks.
  auto shared_lock1 = file->TryLock(FileLock::Shared);
  ASSERT_THAT(shared_lock1, IsSuccess(_));
  EXPECT_TRUE(shared_lock1->is_locked());

  auto shared_lock2 = file2->TryLock(FileLock::Shared);
  ASSERT_THAT(shared_lock2, IsSuccess(_));
  EXPECT_TRUE(shared_lock2->is_locked());

  // An exclusive lock should fail.
  auto file3 = dir_.OpenReadOnly("test_file");
  ASSERT_THAT(file3, IsSuccess(_));
  auto exclusive_lock = file3->TryLock(FileLock::Exclusive);
  ASSERT_THAT(exclusive_lock, IsError(_));
  EXPECT_TRUE(exclusive_lock.error().would_block());

  // Release locks and close files.
  *shared_lock1 = {};
  *shared_lock2 = {};
  ASSERT_THAT((*std::move(file)).Close(), IsSuccess(_));
  ASSERT_THAT((*std::move(file2)).Close(), IsSuccess(_));
  ASSERT_THAT((*std::move(file3)).Close(), IsSuccess(_));
}

TEST_F(FilesystemTest, ReadAndAppendEntries) {
  // Test with an empty directory.
  {
    auto entries = dir_.ReadEntries();
    ASSERT_THAT(entries, IsSuccess(_));
    EXPECT_TRUE(entries->empty());
  }
  {
    llvm::SmallVector<std::filesystem::path> entries;
    EXPECT_THAT(dir_.AppendEntriesIf(entries), IsSuccess(_));
    EXPECT_TRUE(entries.empty());
  }

  // Create some files and directories.
  ASSERT_THAT(dir_.WriteFileFromString("file1", ""), IsSuccess(_));
  ASSERT_THAT(dir_.WriteFileFromString("file2", ""), IsSuccess(_));
  ASSERT_THAT(dir_.WriteFileFromString(".hidden", ""), IsSuccess(_));
  ASSERT_THAT(dir_.CreateDirectories("subdir1"), IsSuccess(_));
  ASSERT_THAT(dir_.CreateDirectories("subdir2"), IsSuccess(_));

  // Test ReadEntries.
  {
    auto entries = dir_.ReadEntries();
    ASSERT_THAT(entries, IsSuccess(_));
    EXPECT_THAT(*entries, UnorderedElementsAre(".hidden", "file1", "file2",
                                               "subdir1", "subdir2"));
  }

  // Test AppendEntriesIf with no predicate.
  {
    llvm::SmallVector<std::filesystem::path> entries;
    EXPECT_THAT(dir_.AppendEntriesIf(entries), IsSuccess(_));
    EXPECT_THAT(entries, UnorderedElementsAre(".hidden", "file1", "file2",
                                              "subdir1", "subdir2"));
  }

  // Test AppendEntriesIf with a predicate.
  {
    llvm::SmallVector<std::filesystem::path> entries;
    auto result = dir_.AppendEntriesIf(
        entries, [](llvm::StringRef name) { return name.starts_with("file"); });
    EXPECT_THAT(result, IsSuccess(_));
    EXPECT_THAT(entries, UnorderedElementsAre("file1", "file2"));
  }

  // Test AppendEntriesIf with directory splitting and a predicate.
  {
    llvm::SmallVector<std::filesystem::path> dir_entries;
    llvm::SmallVector<std::filesystem::path> non_dir_entries;
    auto result = dir_.AppendEntriesIf(
        dir_entries, non_dir_entries,
        [](llvm::StringRef name) { return !name.starts_with("."); });
    EXPECT_THAT(result, IsSuccess(_));
    EXPECT_THAT(dir_entries, UnorderedElementsAre("subdir1", "subdir2"));
    EXPECT_THAT(non_dir_entries, UnorderedElementsAre("file1", "file2"));
  }
}

TEST_F(FilesystemTest, MtimeAndUpdateTimes) {
  // Test UpdateTimes on a path that doesn't exist.
  auto update_missing = dir_.UpdateTimes("test_file");
  ASSERT_THAT(update_missing, IsError(_));
  EXPECT_TRUE(update_missing.error().no_entity());

  // Create a file and get its initial modification time.
  ASSERT_THAT(dir_.WriteFileFromString("test_file", "content"), IsSuccess(_));
  auto stat = dir_.Stat("test_file");
  ASSERT_THAT(stat, IsSuccess(_));
  auto time1 = stat->mtime();

  // Repeated stats have stable time.
  stat = dir_.Stat("test_file");
  ASSERT_THAT(stat, IsSuccess(_));
  EXPECT_THAT(stat->mtime(), Eq(time1));

  // Update the timestamp to a specific time in the past.
  auto past_time = time1 - std::chrono::seconds(120);
  ASSERT_THAT(dir_.UpdateTimes("test_file", past_time), IsSuccess(_));
  stat = dir_.Stat("test_file");
  ASSERT_THAT(stat, IsSuccess(_));
  EXPECT_THAT(stat->mtime(), Eq(past_time));

  // Now test updating times on an open file. Should still be at `past_time`.
  auto file = *dir_.OpenReadWrite("test_file");
  auto file_stat = file.Stat();
  ASSERT_THAT(file_stat, IsSuccess(_));
  EXPECT_THAT(file_stat->mtime(), Eq(past_time));

  // Update the times through the file and verify those updates arrived.
  ASSERT_THAT(file.UpdateTimes(time1), IsSuccess(_));
  file_stat = file.Stat();
  ASSERT_THAT(file_stat, IsSuccess(_));
  EXPECT_THAT(file_stat->mtime(), Eq(time1));

  ASSERT_THAT(std::move(file).Close(), IsSuccess(_));
}

}  // namespace
}  // namespace Carbon::Filesystem
