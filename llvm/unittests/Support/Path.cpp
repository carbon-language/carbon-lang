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

  int FileDescriptor;
  SmallString<64> TempPath;
  ASSERT_FALSE(fs::unique_file("%%-%%-%%-%%.temp", FileDescriptor, TempPath));

  bool TempFileExists;
  ASSERT_FALSE(sys::fs::exists(Twine(TempPath), TempFileExists));
  EXPECT_TRUE(TempFileExists);

  ::close(FileDescriptor);
  ::remove(TempPath.begin());

  ASSERT_FALSE(fs::exists(Twine(TempPath), TempFileExists));
  // FIXME: This is returning true on some systems...
  // EXPECT_FALSE(TempFileExists);
}

} // anonymous namespace
