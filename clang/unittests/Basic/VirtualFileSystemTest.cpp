//===- unittests/Basic/VirtualFileSystem.cpp ---------------- VFS tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/VirtualFileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
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

  void addRegularFile(StringRef Path, sys::fs::perms Perms = sys::fs::all_all) {
    vfs::Status S(Path, Path, UniqueID(FSID, FileID++), sys::TimeValue::now(),
                  0, 0, 1024, sys::fs::file_type::regular_file, Perms);
    addEntry(Path, S);
  }

  void addDirectory(StringRef Path, sys::fs::perms Perms = sys::fs::all_all) {
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

TEST(VirtualFileSystemTest, StatusQueries) {
  IntrusiveRefCntPtr<DummyFileSystem> D(new DummyFileSystem());
  ErrorOr<vfs::Status> Status((error_code()));

  D->addRegularFile("/foo");
  Status = D->status("/foo");
  ASSERT_EQ(errc::success, Status.getError());
  EXPECT_TRUE(Status->isStatusKnown());
  EXPECT_FALSE(Status->isDirectory());
  EXPECT_TRUE(Status->isRegularFile());
  EXPECT_FALSE(Status->isSymlink());
  EXPECT_FALSE(Status->isOther());
  EXPECT_TRUE(Status->exists());

  D->addDirectory("/bar");
  Status = D->status("/bar");
  ASSERT_EQ(errc::success, Status.getError());
  EXPECT_TRUE(Status->isStatusKnown());
  EXPECT_TRUE(Status->isDirectory());
  EXPECT_FALSE(Status->isRegularFile());
  EXPECT_FALSE(Status->isSymlink());
  EXPECT_FALSE(Status->isOther());
  EXPECT_TRUE(Status->exists());

  D->addSymlink("/baz");
  Status = D->status("/baz");
  ASSERT_EQ(errc::success, Status.getError());
  EXPECT_TRUE(Status->isStatusKnown());
  EXPECT_FALSE(Status->isDirectory());
  EXPECT_FALSE(Status->isRegularFile());
  EXPECT_TRUE(Status->isSymlink());
  EXPECT_FALSE(Status->isOther());
  EXPECT_TRUE(Status->exists());

  EXPECT_TRUE(Status->equivalent(*Status));
  ErrorOr<vfs::Status> Status2 = D->status("/foo");
  ASSERT_EQ(errc::success, Status2.getError());
  EXPECT_FALSE(Status->equivalent(*Status2));
}

TEST(VirtualFileSystemTest, BaseOnlyOverlay) {
  IntrusiveRefCntPtr<DummyFileSystem> D(new DummyFileSystem());
  ErrorOr<vfs::Status> Status((error_code()));
  EXPECT_FALSE(Status = D->status("/foo"));

  IntrusiveRefCntPtr<vfs::OverlayFileSystem> O(new vfs::OverlayFileSystem(D));
  EXPECT_FALSE(Status = O->status("/foo"));

  D->addRegularFile("/foo");
  Status = D->status("/foo");
  EXPECT_EQ(errc::success, Status.getError());

  ErrorOr<vfs::Status> Status2((error_code()));
  Status2 = O->status("/foo");
  EXPECT_EQ(errc::success, Status2.getError());
  EXPECT_TRUE(Status->equivalent(*Status2));
}

TEST(VirtualFileSystemTest, OverlayFiles) {
  IntrusiveRefCntPtr<DummyFileSystem> Base(new DummyFileSystem());
  IntrusiveRefCntPtr<DummyFileSystem> Middle(new DummyFileSystem());
  IntrusiveRefCntPtr<DummyFileSystem> Top(new DummyFileSystem());
  IntrusiveRefCntPtr<vfs::OverlayFileSystem> O(
      new vfs::OverlayFileSystem(Base));
  O->pushOverlay(Middle);
  O->pushOverlay(Top);

  ErrorOr<vfs::Status> Status1((error_code())), Status2((error_code())),
      Status3((error_code())), StatusB((error_code())), StatusM((error_code())),
      StatusT((error_code()));

  Base->addRegularFile("/foo");
  StatusB = Base->status("/foo");
  ASSERT_EQ(errc::success, StatusB.getError());
  Status1 = O->status("/foo");
  ASSERT_EQ(errc::success, Status1.getError());
  Middle->addRegularFile("/foo");
  StatusM = Middle->status("/foo");
  ASSERT_EQ(errc::success, StatusM.getError());
  Status2 = O->status("/foo");
  ASSERT_EQ(errc::success, Status2.getError());
  Top->addRegularFile("/foo");
  StatusT = Top->status("/foo");
  ASSERT_EQ(errc::success, StatusT.getError());
  Status3 = O->status("/foo");
  ASSERT_EQ(errc::success, Status3.getError());

  EXPECT_TRUE(Status1->equivalent(*StatusB));
  EXPECT_TRUE(Status2->equivalent(*StatusM));
  EXPECT_TRUE(Status3->equivalent(*StatusT));

  EXPECT_FALSE(Status1->equivalent(*Status2));
  EXPECT_FALSE(Status2->equivalent(*Status3));
  EXPECT_FALSE(Status1->equivalent(*Status3));
}

TEST(VirtualFileSystemTest, OverlayDirsNonMerged) {
  IntrusiveRefCntPtr<DummyFileSystem> Lower(new DummyFileSystem());
  IntrusiveRefCntPtr<DummyFileSystem> Upper(new DummyFileSystem());
  IntrusiveRefCntPtr<vfs::OverlayFileSystem> O(
      new vfs::OverlayFileSystem(Lower));
  O->pushOverlay(Upper);

  Lower->addDirectory("/lower-only");
  Upper->addDirectory("/upper-only");

  // non-merged paths should be the same
  ErrorOr<vfs::Status> Status1 = Lower->status("/lower-only");
  ASSERT_EQ(errc::success, Status1.getError());
  ErrorOr<vfs::Status> Status2 = O->status("/lower-only");
  ASSERT_EQ(errc::success, Status2.getError());
  EXPECT_TRUE(Status1->equivalent(*Status2));

  Status1 = Upper->status("/upper-only");
  ASSERT_EQ(errc::success, Status1.getError());
  Status2 = O->status("/upper-only");
  ASSERT_EQ(errc::success, Status2.getError());
  EXPECT_TRUE(Status1->equivalent(*Status2));
}

TEST(VirtualFileSystemTest, MergedDirPermissions) {
  // merged directories get the permissions of the upper dir
  IntrusiveRefCntPtr<DummyFileSystem> Lower(new DummyFileSystem());
  IntrusiveRefCntPtr<DummyFileSystem> Upper(new DummyFileSystem());
  IntrusiveRefCntPtr<vfs::OverlayFileSystem> O(
      new vfs::OverlayFileSystem(Lower));
  O->pushOverlay(Upper);

  ErrorOr<vfs::Status> Status((error_code()));
  Lower->addDirectory("/both", sys::fs::owner_read);
  Upper->addDirectory("/both", sys::fs::owner_all | sys::fs::group_read);
  Status = O->status("/both");
  ASSERT_EQ(errc::success, Status.getError());
  EXPECT_EQ(0740, Status->getPermissions());

  // permissions (as usual) are not recursively applied
  Lower->addRegularFile("/both/foo", sys::fs::owner_read);
  Upper->addRegularFile("/both/bar", sys::fs::owner_write);
  Status = O->status("/both/foo");
  ASSERT_EQ(errc::success, Status.getError());
  EXPECT_EQ(0400, Status->getPermissions());
  Status = O->status("/both/bar");
  ASSERT_EQ(errc::success, Status.getError());
  EXPECT_EQ(0200, Status->getPermissions());
}

class VFSFromYAMLTest : public ::testing::Test {
public:
  int NumDiagnostics;
  void SetUp() {
    NumDiagnostics = 0;
  }

  static void CountingDiagHandler(const SMDiagnostic &, void *Context) {
    VFSFromYAMLTest *Test = static_cast<VFSFromYAMLTest *>(Context);
    ++Test->NumDiagnostics;
  }

  IntrusiveRefCntPtr<vfs::FileSystem>
  getFromYAMLRawString(StringRef Content,
                       IntrusiveRefCntPtr<vfs::FileSystem> ExternalFS) {
    MemoryBuffer *Buffer = MemoryBuffer::getMemBuffer(Content);
    return getVFSFromYAML(Buffer, CountingDiagHandler, this, ExternalFS);
  }

  IntrusiveRefCntPtr<vfs::FileSystem> getFromYAMLString(
      StringRef Content,
      IntrusiveRefCntPtr<vfs::FileSystem> ExternalFS = new DummyFileSystem()) {
    std::string VersionPlusContent("{\n  'version':0,\n");
    VersionPlusContent += Content.slice(Content.find('{') + 1, StringRef::npos);
    return getFromYAMLRawString(VersionPlusContent, ExternalFS);
  }

};

TEST_F(VFSFromYAMLTest, BasicVFSFromYAML) {
  IntrusiveRefCntPtr<vfs::FileSystem> FS;
  FS = getFromYAMLString("");
  EXPECT_EQ(NULL, FS.getPtr());
  FS = getFromYAMLString("[]");
  EXPECT_EQ(NULL, FS.getPtr());
  FS = getFromYAMLString("'string'");
  EXPECT_EQ(NULL, FS.getPtr());
  EXPECT_EQ(3, NumDiagnostics);
}

TEST_F(VFSFromYAMLTest, MappedFiles) {
  IntrusiveRefCntPtr<DummyFileSystem> Lower(new DummyFileSystem());
  Lower->addRegularFile("/foo/bar/a");
  IntrusiveRefCntPtr<vfs::FileSystem> FS =
      getFromYAMLString("{ 'roots': [\n"
                        "{\n"
                        "  'type': 'directory',\n"
                        "  'name': '/',\n"
                        "  'contents': [ {\n"
                        "                  'type': 'file',\n"
                        "                  'name': 'file1',\n"
                        "                  'external-contents': '/foo/bar/a'\n"
                        "                },\n"
                        "                {\n"
                        "                  'type': 'file',\n"
                        "                  'name': 'file2',\n"
                        "                  'external-contents': '/foo/b'\n"
                        "                }\n"
                        "              ]\n"
                        "}\n"
                        "]\n"
                        "}",
                        Lower);
  ASSERT_TRUE(FS.getPtr());

  IntrusiveRefCntPtr<vfs::OverlayFileSystem> O(
      new vfs::OverlayFileSystem(Lower));
  O->pushOverlay(FS);

  // file
  ErrorOr<vfs::Status> S = O->status("/file1");
  ASSERT_EQ(errc::success, S.getError());
  EXPECT_EQ("/file1", S->getName());
  EXPECT_EQ("/foo/bar/a", S->getExternalName());

  ErrorOr<vfs::Status> SLower = O->status("/foo/bar/a");
  EXPECT_EQ("/foo/bar/a", SLower->getName());
  EXPECT_TRUE(S->equivalent(*SLower));

  // directory
  S = O->status("/");
  ASSERT_EQ(errc::success, S.getError());
  EXPECT_TRUE(S->isDirectory());
  EXPECT_TRUE(S->equivalent(*O->status("/"))); // non-volatile UniqueID

  // broken mapping
  EXPECT_EQ(errc::no_such_file_or_directory, O->status("/file2").getError());
  EXPECT_EQ(0, NumDiagnostics);
}

TEST_F(VFSFromYAMLTest, CaseInsensitive) {
  IntrusiveRefCntPtr<DummyFileSystem> Lower(new DummyFileSystem());
  Lower->addRegularFile("/foo/bar/a");
  IntrusiveRefCntPtr<vfs::FileSystem> FS =
      getFromYAMLString("{ 'case-sensitive': 'false',\n"
                        "  'roots': [\n"
                        "{\n"
                        "  'type': 'directory',\n"
                        "  'name': '/',\n"
                        "  'contents': [ {\n"
                        "                  'type': 'file',\n"
                        "                  'name': 'XX',\n"
                        "                  'external-contents': '/foo/bar/a'\n"
                        "                }\n"
                        "              ]\n"
                        "}]}",
                        Lower);
  ASSERT_TRUE(FS.getPtr());

  IntrusiveRefCntPtr<vfs::OverlayFileSystem> O(
      new vfs::OverlayFileSystem(Lower));
  O->pushOverlay(FS);

  ErrorOr<vfs::Status> S = O->status("/XX");
  ASSERT_EQ(errc::success, S.getError());

  ErrorOr<vfs::Status> SS = O->status("/xx");
  ASSERT_EQ(errc::success, SS.getError());
  EXPECT_TRUE(S->equivalent(*SS));
  SS = O->status("/xX");
  EXPECT_TRUE(S->equivalent(*SS));
  SS = O->status("/Xx");
  EXPECT_TRUE(S->equivalent(*SS));
  EXPECT_EQ(0, NumDiagnostics);
}

TEST_F(VFSFromYAMLTest, CaseSensitive) {
  IntrusiveRefCntPtr<DummyFileSystem> Lower(new DummyFileSystem());
  Lower->addRegularFile("/foo/bar/a");
  IntrusiveRefCntPtr<vfs::FileSystem> FS =
      getFromYAMLString("{ 'case-sensitive': 'true',\n"
                        "  'roots': [\n"
                        "{\n"
                        "  'type': 'directory',\n"
                        "  'name': '/',\n"
                        "  'contents': [ {\n"
                        "                  'type': 'file',\n"
                        "                  'name': 'XX',\n"
                        "                  'external-contents': '/foo/bar/a'\n"
                        "                }\n"
                        "              ]\n"
                        "}]}",
                        Lower);
  ASSERT_TRUE(FS.getPtr());

  IntrusiveRefCntPtr<vfs::OverlayFileSystem> O(
      new vfs::OverlayFileSystem(Lower));
  O->pushOverlay(FS);

  ErrorOr<vfs::Status> SS = O->status("/xx");
  EXPECT_EQ(errc::no_such_file_or_directory, SS.getError());
  SS = O->status("/xX");
  EXPECT_EQ(errc::no_such_file_or_directory, SS.getError());
  SS = O->status("/Xx");
  EXPECT_EQ(errc::no_such_file_or_directory, SS.getError());
  EXPECT_EQ(0, NumDiagnostics);
}

TEST_F(VFSFromYAMLTest, IllegalVFSFile) {
  IntrusiveRefCntPtr<DummyFileSystem> Lower(new DummyFileSystem());

  // invalid YAML at top-level
  IntrusiveRefCntPtr<vfs::FileSystem> FS = getFromYAMLString("{]", Lower);
  EXPECT_FALSE(FS.getPtr());
  // invalid YAML in roots
  FS = getFromYAMLString("{ 'roots':[}", Lower);
  // invalid YAML in directory
  FS = getFromYAMLString(
      "{ 'roots':[ { 'name': 'foo', 'type': 'directory', 'contents': [}",
      Lower);
  EXPECT_FALSE(FS.getPtr());

  // invalid configuration
  FS = getFromYAMLString("{ 'knobular': 'true', 'roots':[] }", Lower);
  EXPECT_FALSE(FS.getPtr());
  FS = getFromYAMLString("{ 'case-sensitive': 'maybe', 'roots':[] }", Lower);
  EXPECT_FALSE(FS.getPtr());

  // invalid roots
  FS = getFromYAMLString("{ 'roots':'' }", Lower);
  EXPECT_FALSE(FS.getPtr());
  FS = getFromYAMLString("{ 'roots':{} }", Lower);
  EXPECT_FALSE(FS.getPtr());

  // invalid entries
  FS = getFromYAMLString(
      "{ 'roots':[ { 'type': 'other', 'name': 'me', 'contents': '' }", Lower);
  EXPECT_FALSE(FS.getPtr());
  FS = getFromYAMLString("{ 'roots':[ { 'type': 'file', 'name': [], "
                         "'external-contents': 'other' }",
                         Lower);
  EXPECT_FALSE(FS.getPtr());
  FS = getFromYAMLString(
      "{ 'roots':[ { 'type': 'file', 'name': 'me', 'external-contents': [] }",
      Lower);
  EXPECT_FALSE(FS.getPtr());
  FS = getFromYAMLString(
      "{ 'roots':[ { 'type': 'file', 'name': 'me', 'external-contents': {} }",
      Lower);
  EXPECT_FALSE(FS.getPtr());
  FS = getFromYAMLString(
      "{ 'roots':[ { 'type': 'directory', 'name': 'me', 'contents': {} }",
      Lower);
  EXPECT_FALSE(FS.getPtr());
  FS = getFromYAMLString(
      "{ 'roots':[ { 'type': 'directory', 'name': 'me', 'contents': '' }",
      Lower);
  EXPECT_FALSE(FS.getPtr());
  FS = getFromYAMLString(
      "{ 'roots':[ { 'thingy': 'directory', 'name': 'me', 'contents': [] }",
      Lower);
  EXPECT_FALSE(FS.getPtr());

  // missing mandatory fields
  FS = getFromYAMLString("{ 'roots':[ { 'type': 'file', 'name': 'me' }", Lower);
  EXPECT_FALSE(FS.getPtr());
  FS = getFromYAMLString(
      "{ 'roots':[ { 'type': 'file', 'external-contents': 'other' }", Lower);
  EXPECT_FALSE(FS.getPtr());
  FS = getFromYAMLString("{ 'roots':[ { 'name': 'me', 'contents': [] }", Lower);
  EXPECT_FALSE(FS.getPtr());

  // duplicate keys
  FS = getFromYAMLString("{ 'roots':[], 'roots':[] }", Lower);
  EXPECT_FALSE(FS.getPtr());
  FS = getFromYAMLString(
      "{ 'case-sensitive':'true', 'case-sensitive':'true', 'roots':[] }",
      Lower);
  EXPECT_FALSE(FS.getPtr());
  FS =
      getFromYAMLString("{ 'roots':[{'name':'me', 'name':'you', 'type':'file', "
                        "'external-contents':'blah' } ] }",
                        Lower);
  EXPECT_FALSE(FS.getPtr());

  // missing version
  FS = getFromYAMLRawString("{ 'roots':[] }", Lower);
  EXPECT_FALSE(FS.getPtr());

  // bad version number
  FS = getFromYAMLRawString("{ 'version':'foo', 'roots':[] }", Lower);
  EXPECT_FALSE(FS.getPtr());
  FS = getFromYAMLRawString("{ 'version':-1, 'roots':[] }", Lower);
  EXPECT_FALSE(FS.getPtr());
  FS = getFromYAMLRawString("{ 'version':100000, 'roots':[] }", Lower);
  EXPECT_FALSE(FS.getPtr());
  EXPECT_EQ(24, NumDiagnostics);
}
