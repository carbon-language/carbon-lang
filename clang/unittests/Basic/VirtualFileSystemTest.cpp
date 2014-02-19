//===- unittests/Basic/VirtualFileSystem.cpp ---------------- VFS tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/VirtualFileSystem.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"
#include <map>
using namespace clang;
using namespace llvm;
using llvm::sys::fs::UniqueID;

namespace {
class DummyFileSystem : public vfs::FileSystem {
  int FSID;   // used to produce UniqueIDs
  int FileID; // used to produce UniqueIDs
  std::map<std::string, vfs::Status> FilesAndDirs;

  static int getNextFSID() {
    static int Count = 0;
    return Count++;
  }

public:
  DummyFileSystem() : FSID(getNextFSID()), FileID(0) {}

  ErrorOr<vfs::Status> status(const Twine &Path) {
    std::map<std::string, vfs::Status>::iterator I =
      FilesAndDirs.find(Path.str());
    if (I == FilesAndDirs.end())
      return error_code(errc::no_such_file_or_directory, posix_category());
    return I->second;
  }
  error_code openFileForRead(const Twine &Path, OwningPtr<vfs::File> &Result) {
    llvm_unreachable("unimplemented");
  }
  error_code getBufferForFile(const Twine &Name,
                              OwningPtr<MemoryBuffer> &Result,
                              int64_t FileSize = -1,
                              bool RequiresNullTerminator = true) {
    llvm_unreachable("unimplemented");
  }

  void addEntry(StringRef Path, const vfs::Status &Status) {
    FilesAndDirs[Path] = Status;
  }

  void addRegularFile(StringRef Path, sys::fs::perms Perms=sys::fs::all_all) {
    vfs::Status S(Path, Path, UniqueID(FSID, FileID++), sys::TimeValue::now(),
                  0, 0, 1024, sys::fs::file_type::regular_file, Perms);
    addEntry(Path, S);
  }

  void addDirectory(StringRef Path, sys::fs::perms Perms=sys::fs::all_all) {
    vfs::Status S(Path, Path, UniqueID(FSID, FileID++), sys::TimeValue::now(),
                  0, 0, 0, sys::fs::file_type::directory_file, Perms);
    addEntry(Path, S);
  }

  void addSymlink(StringRef Path) {
    vfs::Status S(Path, Path, UniqueID(FSID, FileID++), sys::TimeValue::now(),
                  0, 0, 0, sys::fs::file_type::symlink_file, sys::fs::all_all);
    addEntry(Path, S);
  }
};
} // end anonymous namespace

TEST(VirtualFileSystemTest, status_queries) {
  IntrusiveRefCntPtr<DummyFileSystem> D(new DummyFileSystem());
  ErrorOr<vfs::Status> Status((error_code()));

  D->addRegularFile("/foo");
  ASSERT_TRUE(Status = D->status("/foo"));
  EXPECT_TRUE(Status->isStatusKnown());
  EXPECT_FALSE(Status->isDirectory());
  EXPECT_TRUE(Status->isRegularFile());
  EXPECT_FALSE(Status->isSymlink());
  EXPECT_FALSE(Status->isOther());
  EXPECT_TRUE(Status->exists());

  D->addDirectory("/bar");
  ASSERT_TRUE(Status = D->status("/bar"));
  EXPECT_TRUE(Status->isStatusKnown());
  EXPECT_TRUE(Status->isDirectory());
  EXPECT_FALSE(Status->isRegularFile());
  EXPECT_FALSE(Status->isSymlink());
  EXPECT_FALSE(Status->isOther());
  EXPECT_TRUE(Status->exists());

  D->addSymlink("/baz");
  ASSERT_TRUE(Status = D->status("/baz"));
  EXPECT_TRUE(Status->isStatusKnown());
  EXPECT_FALSE(Status->isDirectory());
  EXPECT_FALSE(Status->isRegularFile());
  EXPECT_TRUE(Status->isSymlink());
  EXPECT_FALSE(Status->isOther());
  EXPECT_TRUE(Status->exists());

  EXPECT_TRUE(Status->equivalent(*Status));
  ErrorOr<vfs::Status> Status2((error_code()));
  ASSERT_TRUE(Status2 = D->status("/foo"));
  EXPECT_FALSE(Status->equivalent(*Status2));
}

TEST(VirtualFileSystemTest, base_only_overlay) {
  IntrusiveRefCntPtr<DummyFileSystem> D(new DummyFileSystem());
  ErrorOr<vfs::Status> Status((error_code()));
  EXPECT_FALSE(Status = D->status("/foo"));

  IntrusiveRefCntPtr<vfs::OverlayFileSystem> O(new vfs::OverlayFileSystem(D));
  EXPECT_FALSE(Status = O->status("/foo"));

  D->addRegularFile("/foo");
  EXPECT_TRUE(Status = D->status("/foo"));

  ErrorOr<vfs::Status> Status2((error_code()));
  EXPECT_TRUE(Status2 = O->status("/foo"));
  EXPECT_TRUE(Status->equivalent(*Status2));
}

TEST(VirtualFileSystemTest, overlay_files) {
  IntrusiveRefCntPtr<DummyFileSystem> Base(new DummyFileSystem());
  IntrusiveRefCntPtr<DummyFileSystem> Middle(new DummyFileSystem());
  IntrusiveRefCntPtr<DummyFileSystem> Top(new DummyFileSystem());
  IntrusiveRefCntPtr<vfs::OverlayFileSystem> O(new vfs::OverlayFileSystem(Base));
  O->pushOverlay(Middle);
  O->pushOverlay(Top);

  ErrorOr<vfs::Status> Status1((error_code())), Status2((error_code())),
                       Status3((error_code())), StatusB((error_code())),
                       StatusM((error_code())), StatusT((error_code()));

  Base->addRegularFile("/foo");
  ASSERT_TRUE(StatusB = Base->status("/foo"));
  ASSERT_TRUE(Status1 = O->status("/foo"));
  Middle->addRegularFile("/foo");
  ASSERT_TRUE(StatusM = Middle->status("/foo"));
  ASSERT_TRUE(Status2 = O->status("/foo"));
  Top->addRegularFile("/foo");
  ASSERT_TRUE(StatusT = Top->status("/foo"));
  ASSERT_TRUE(Status3 = O->status("/foo"));

  EXPECT_TRUE(Status1->equivalent(*StatusB));
  EXPECT_TRUE(Status2->equivalent(*StatusM));
  EXPECT_TRUE(Status3->equivalent(*StatusT));

  EXPECT_FALSE(Status1->equivalent(*Status2));
  EXPECT_FALSE(Status2->equivalent(*Status3));
  EXPECT_FALSE(Status1->equivalent(*Status3));
}

TEST(VirtualFileSystemTest, overlay_dirs) {
  IntrusiveRefCntPtr<DummyFileSystem> Lower(new DummyFileSystem());
  IntrusiveRefCntPtr<DummyFileSystem> Upper(new DummyFileSystem());
  IntrusiveRefCntPtr<vfs::OverlayFileSystem>
    O(new vfs::OverlayFileSystem(Lower));
  O->pushOverlay(Upper);

  ErrorOr<vfs::Status> Status1((error_code())), Status2((error_code())),
                       Status3((error_code()));

  Lower->addDirectory("/lower-only");
  Lower->addDirectory("/both");
  Upper->addDirectory("/both");
  Upper->addDirectory("/upper-only");

  // non-merged paths should be the same
  ASSERT_TRUE(Status1 = Lower->status("/lower-only"));
  ASSERT_TRUE(Status2 = O->status("/lower-only"));
  EXPECT_TRUE(Status1->equivalent(*Status2));

  ASSERT_TRUE(Status1 = Lower->status("/lower-only"));
  ASSERT_TRUE(Status2 = O->status("/lower-only"));
  EXPECT_TRUE(Status1->equivalent(*Status2));
}

TEST(VirtualFileSystemTest, permissions) {
  // merged directories get the permissions of the upper dir
  IntrusiveRefCntPtr<DummyFileSystem> Lower(new DummyFileSystem());
  IntrusiveRefCntPtr<DummyFileSystem> Upper(new DummyFileSystem());
  IntrusiveRefCntPtr<vfs::OverlayFileSystem>
    O(new vfs::OverlayFileSystem(Lower));
  O->pushOverlay(Upper);

  ErrorOr<vfs::Status> Status((error_code()));
  Lower->addDirectory("/both", sys::fs::owner_read);
  Upper->addDirectory("/both", sys::fs::owner_all | sys::fs::group_read);
  ASSERT_TRUE(Status = O->status("/both"));
  EXPECT_EQ(0740, Status->getPermissions());

  // permissions (as usual) are not recursively applied
  Lower->addRegularFile("/both/foo", sys::fs::owner_read);
  Upper->addRegularFile("/both/bar", sys::fs::owner_write);
  ASSERT_TRUE(Status = O->status("/both/foo"));
  EXPECT_EQ(0400, Status->getPermissions());
  ASSERT_TRUE(Status = O->status("/both/bar"));
  EXPECT_EQ(0200, Status->getPermissions());
}
