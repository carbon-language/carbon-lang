//===- llvm/unittest/Support/Path.cpp - Path tests ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/PathV2.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::sys;

#define ASSERT_NO_ERROR(x) \
  if (error_code ASSERT_NO_ERROR_ec = x) { \
    SmallString<128> MessageStorage; \
    raw_svector_ostream Message(MessageStorage); \
    Message << #x ": did not return errc::success.\n" \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n" \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n"; \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str()); \
  } else {}

namespace {

TEST(Support, Path) {
  SmallVector<StringRef, 40> paths;
  paths.push_back("");
  paths.push_back(".");
  paths.push_back("..");
  paths.push_back("foo");
  paths.push_back("/");
  paths.push_back("/foo");
  paths.push_back("foo/");
  paths.push_back("/foo/");
  paths.push_back("foo/bar");
  paths.push_back("/foo/bar");
  paths.push_back("//net");
  paths.push_back("//net/foo");
  paths.push_back("///foo///");
  paths.push_back("///foo///bar");
  paths.push_back("/.");
  paths.push_back("./");
  paths.push_back("/..");
  paths.push_back("../");
  paths.push_back("foo/.");
  paths.push_back("foo/..");
  paths.push_back("foo/./");
  paths.push_back("foo/./bar");
  paths.push_back("foo/..");
  paths.push_back("foo/../");
  paths.push_back("foo/../bar");
  paths.push_back("c:");
  paths.push_back("c:/");
  paths.push_back("c:foo");
  paths.push_back("c:/foo");
  paths.push_back("c:foo/");
  paths.push_back("c:/foo/");
  paths.push_back("c:/foo/bar");
  paths.push_back("prn:");
  paths.push_back("c:\\");
  paths.push_back("c:foo");
  paths.push_back("c:\\foo");
  paths.push_back("c:foo\\");
  paths.push_back("c:\\foo\\");
  paths.push_back("c:\\foo/");
  paths.push_back("c:/foo\\bar");

  for (SmallVector<StringRef, 40>::const_iterator i = paths.begin(),
                                                  e = paths.end();
                                                  i != e;
                                                  ++i) {
    for (sys::path::const_iterator ci = sys::path::begin(*i),
                                   ce = sys::path::end(*i);
                                   ci != ce;
                                   ++ci) {
      ASSERT_FALSE(ci->empty());
    }

#if 0 // Valgrind is whining about this.
    outs() << "    Reverse Iteration: [";
    for (sys::path::reverse_iterator ci = sys::path::rbegin(*i),
                                     ce = sys::path::rend(*i);
                                     ci != ce;
                                     ++ci) {
      outs() << *ci << ',';
    }
    outs() << "]\n";
#endif

    path::has_root_path(*i);
    path::root_path(*i);
    path::has_root_name(*i);
    path::root_name(*i);
    path::has_root_directory(*i);
    path::root_directory(*i);
    path::has_parent_path(*i);
    path::parent_path(*i);
    path::has_filename(*i);
    path::filename(*i);
    path::has_stem(*i);
    path::stem(*i);
    path::has_extension(*i);
    path::extension(*i);
    path::is_absolute(*i);
    path::is_relative(*i);

    SmallString<128> temp_store;
    temp_store = *i;
    ASSERT_NO_ERROR(fs::make_absolute(temp_store));
    temp_store = *i;
    path::remove_filename(temp_store);

    temp_store = *i;
    path::replace_extension(temp_store, "ext");
    StringRef filename(temp_store.begin(), temp_store.size()), stem, ext;
    stem = path::stem(filename);
    ext  = path::extension(filename);
    EXPECT_EQ(*(--sys::path::end(filename)), (stem + ext).str());

    path::native(*i, temp_store);
  }
}

class FileSystemTest : public testing::Test {
protected:
  /// Unique temporary directory in which all created filesystem entities must
  /// be placed. It is recursively removed at the end of each test.
  SmallString<128> TestDirectory;

  virtual void SetUp() {
    int fd;
    ASSERT_NO_ERROR(
      fs::unique_file("file-system-test-%%-%%-%%-%%/test-directory.anchor", fd,
                      TestDirectory));
    // We don't care about this specific file.
    ::close(fd);
    TestDirectory = path::parent_path(TestDirectory);
    errs() << "Test Directory: " << TestDirectory << '\n';
    errs().flush();
  }

  virtual void TearDown() {
    uint32_t removed;
    ASSERT_NO_ERROR(fs::remove_all(TestDirectory.str(), removed));
  }
};

TEST_F(FileSystemTest, TempFiles) {
  // Create a temp file.
  int FileDescriptor;
  SmallString<64> TempPath;
  ASSERT_NO_ERROR(
    fs::unique_file("%%-%%-%%-%%.temp", FileDescriptor, TempPath));

  // Make sure it exists.
  bool TempFileExists;
  ASSERT_NO_ERROR(sys::fs::exists(Twine(TempPath), TempFileExists));
  EXPECT_TRUE(TempFileExists);

  // Create another temp tile.
  int FD2;
  SmallString<64> TempPath2;
  ASSERT_NO_ERROR(fs::unique_file("%%-%%-%%-%%.temp", FD2, TempPath2));
  ASSERT_NE(TempPath.str(), TempPath2.str());

  // Try to copy the first to the second.
  EXPECT_EQ(
    fs::copy_file(Twine(TempPath), Twine(TempPath2)), errc::file_exists);

  ::close(FD2);
  // Try again with the proper options.
  ASSERT_NO_ERROR(fs::copy_file(Twine(TempPath), Twine(TempPath2),
                                fs::copy_option::overwrite_if_exists));
  // Remove Temp2.
  ASSERT_NO_ERROR(fs::remove(Twine(TempPath2), TempFileExists));
  EXPECT_TRUE(TempFileExists);

  // Make sure Temp2 doesn't exist.
  ASSERT_NO_ERROR(fs::exists(Twine(TempPath2), TempFileExists));
  EXPECT_FALSE(TempFileExists);

  // Create a hard link to Temp1.
  ASSERT_NO_ERROR(fs::create_hard_link(Twine(TempPath), Twine(TempPath2)));
  bool equal;
  ASSERT_NO_ERROR(fs::equivalent(Twine(TempPath), Twine(TempPath2), equal));
  EXPECT_TRUE(equal);

  // Remove Temp1.
  ::close(FileDescriptor);
  ASSERT_NO_ERROR(fs::remove(Twine(TempPath), TempFileExists));
  EXPECT_TRUE(TempFileExists);

  // Remove the hard link.
  ASSERT_NO_ERROR(fs::remove(Twine(TempPath2), TempFileExists));
  EXPECT_TRUE(TempFileExists);

  // Make sure Temp1 doesn't exist.
  ASSERT_NO_ERROR(fs::exists(Twine(TempPath), TempFileExists));
  EXPECT_FALSE(TempFileExists);
}

TEST_F(FileSystemTest, DirectoryIteration) {
  error_code ec;
  for (fs::directory_iterator i(".", ec), e; i != e; i.increment(ec))
    ASSERT_NO_ERROR(ec);
}

TEST_F(FileSystemTest, Magic) {
  struct type {
    const char *filename;
    const char *magic_str;
    size_t      magic_str_len;
  } types [] = {{"magic.archive", "!<arch>\x0A", 8}};

  // Create some files filled with magic.
  for (type *i = types, *e = types + (sizeof(types) / sizeof(type)); i != e;
                                                                     ++i) {
    SmallString<128> file_pathname(TestDirectory);
    path::append(file_pathname, i->filename);
    std::string ErrMsg;
    raw_fd_ostream file(file_pathname.c_str(), ErrMsg,
                        raw_fd_ostream::F_Binary);
    ASSERT_FALSE(file.has_error());
    StringRef magic(i->magic_str, i->magic_str_len);
    file << magic;
    file.flush();
    bool res = false;
    ASSERT_NO_ERROR(fs::has_magic(file_pathname.c_str(), magic, res));
    EXPECT_TRUE(res);
  }
}

} // anonymous namespace
