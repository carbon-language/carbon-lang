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

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::sys;

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

    bool      bres;
    StringRef sfres;
    ASSERT_FALSE(path::has_root_path(*i, bres));
    ASSERT_FALSE(path::root_path(*i, sfres));
    ASSERT_FALSE(path::has_root_name(*i, bres));
    ASSERT_FALSE(path::root_name(*i, sfres));
    ASSERT_FALSE(path::has_root_directory(*i, bres));
    ASSERT_FALSE(path::root_directory(*i, sfres));
    ASSERT_FALSE(path::has_parent_path(*i, bres));
    ASSERT_FALSE(path::parent_path(*i, sfres));
    ASSERT_FALSE(path::has_filename(*i, bres));
    ASSERT_FALSE(path::filename(*i, sfres));
    ASSERT_FALSE(path::has_stem(*i, bres));
    ASSERT_FALSE(path::stem(*i, sfres));
    ASSERT_FALSE(path::has_extension(*i, bres));
    ASSERT_FALSE(path::extension(*i, sfres));
    ASSERT_FALSE(path::is_absolute(*i, bres));
    ASSERT_FALSE(path::is_relative(*i, bres));

    SmallString<16> temp_store;
    temp_store = *i;
    ASSERT_FALSE(path::make_absolute(temp_store));
    temp_store = *i;
    ASSERT_FALSE(path::remove_filename(temp_store));

    temp_store = *i;
    ASSERT_FALSE(path::replace_extension(temp_store, "ext"));
    StringRef filename(temp_store.begin(), temp_store.size()), stem, ext;
    ASSERT_FALSE(path::stem(filename, stem));
    ASSERT_FALSE(path::extension(filename, ext));
    EXPECT_EQ(*(--sys::path::end(filename)), (stem + ext).str());

    ASSERT_FALSE(path::native(*i, temp_store));

    outs().flush();
  }

  // Create a temp file.
  int FileDescriptor;
  SmallString<64> TempPath;
  ASSERT_FALSE(fs::unique_file("%%-%%-%%-%%.temp", FileDescriptor, TempPath));

  // Make sure it exists.
  bool TempFileExists;
  ASSERT_FALSE(sys::fs::exists(Twine(TempPath), TempFileExists));
  EXPECT_TRUE(TempFileExists);

  // Create another temp tile.
  int FD2;
  SmallString<64> TempPath2;
  ASSERT_FALSE(fs::unique_file("%%-%%-%%-%%.temp", FD2, TempPath2));
  ASSERT_NE(TempPath.str(), TempPath2.str());

  // Try to copy the first to the second.
  EXPECT_EQ(fs::copy_file(Twine(TempPath), Twine(TempPath2)), errc::file_exists);

  ::close(FD2);
  // Try again with the proper options.
  ASSERT_FALSE(fs::copy_file(Twine(TempPath), Twine(TempPath2),
                             fs::copy_option::overwrite_if_exists));
  // Remove Temp2.
  ASSERT_FALSE(fs::remove(Twine(TempPath2), TempFileExists));
  EXPECT_TRUE(TempFileExists);

  // Make sure Temp2 doesn't exist.
  ASSERT_FALSE(fs::exists(Twine(TempPath2), TempFileExists));
  EXPECT_FALSE(TempFileExists);

  // Create a hard link to Temp1.
  ASSERT_FALSE(fs::create_hard_link(Twine(TempPath), Twine(TempPath2)));
  bool equal;
  ASSERT_FALSE(fs::equivalent(Twine(TempPath), Twine(TempPath2), equal));
  EXPECT_TRUE(equal);

  // Remove Temp1.
  ::close(FileDescriptor);
  ASSERT_FALSE(fs::remove(Twine(TempPath), TempFileExists));
  EXPECT_TRUE(TempFileExists);

  // Remove the hard link.
  ASSERT_FALSE(fs::remove(Twine(TempPath2), TempFileExists));
  EXPECT_TRUE(TempFileExists);

  // Make sure Temp1 doesn't exist.
  ASSERT_FALSE(fs::exists(Twine(TempPath), TempFileExists));
  EXPECT_FALSE(TempFileExists);

  // I've yet to do directory iteration on Unix.
#ifdef LLVM_ON_WIN32
  error_code ec;
  for (fs::directory_iterator i(".", ec), e; i != e; i.increment(ec)) {
    if (ec) {
      errs() << ec.message() << '\n';
      errs().flush();
      report_fatal_error("Directory iteration failed!");
    }
  }
#endif
}

} // anonymous namespace
