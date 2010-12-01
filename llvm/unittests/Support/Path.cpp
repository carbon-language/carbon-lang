//===- llvm/unittest/Support/Path.cpp - Path tests ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PathV2.h"

#include "gtest/gtest.h"

using namespace llvm;

#define TEST_OUT(func, result) outs() << "    " #func ": " << result << '\n';

#define TEST_PATH_(header, func, funcname, output) \
  header; \
  if (error_code ec = sys::path::func) \
    ASSERT_FALSE(ec.message().c_str()); \
  TEST_OUT(funcname, output)

#define TEST_PATH(func, ipath, res) TEST_PATH_(;, func(ipath, res), func, res);

#define TEST_PATH_SMALLVEC(func, ipath, inout) \
  TEST_PATH_(inout = ipath, func(inout), func, inout)

#define TEST_PATH_SMALLVEC_P(func, ipath, inout, param) \
  TEST_PATH_(inout = ipath, func(inout, param), func, inout)

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
    outs() << *i << " =>\n    Iteration: [";
    for (sys::path::const_iterator ci = sys::path::begin(*i),
                                   ce = sys::path::end(*i);
                                   ci != ce;
                                   ++ci) {
      outs() << *ci << ',';
    }
    outs() << "]\n";

    outs() << "    Reverse Iteration: [";
    for (sys::path::reverse_iterator ci = sys::path::rbegin(*i),
                                     ce = sys::path::rend(*i);
                                     ci != ce;
                                     ++ci) {
      outs() << *ci << ',';
    }
    outs() << "]\n";

    bool      bres;
    StringRef sfres;
    TEST_PATH(has_root_path, *i, bres);
    TEST_PATH(root_path, *i, sfres);
    TEST_PATH(has_root_name, *i, bres);
    TEST_PATH(root_name, *i, sfres);
    TEST_PATH(has_root_directory, *i, bres);
    TEST_PATH(root_directory, *i, sfres);
    TEST_PATH(has_parent_path, *i, bres);
    TEST_PATH(parent_path, *i, sfres);
    TEST_PATH(has_filename, *i, bres);
    TEST_PATH(filename, *i, sfres);
    TEST_PATH(has_stem, *i, bres);
    TEST_PATH(stem, *i, sfres);
    TEST_PATH(has_extension, *i, bres);
    TEST_PATH(extension, *i, sfres);
    TEST_PATH(is_absolute, *i, bres);
    TEST_PATH(is_relative, *i, bres);

    SmallString<16> temp_store;
    TEST_PATH_SMALLVEC(make_absolute, *i, temp_store);
    TEST_PATH_SMALLVEC(remove_filename, *i, temp_store);

    TEST_PATH_SMALLVEC_P(replace_extension, *i, temp_store, "ext");
    StringRef filename(temp_store.begin(), temp_store.size()), stem, ext;
    TEST_PATH(stem, filename, stem);
    TEST_PATH(extension, filename, ext);
    EXPECT_EQ(*(--sys::path::end(filename)), (stem + ext).str());

    TEST_PATH_(;, native(*i, temp_store), native, temp_store);

    outs().flush();
  }
}

} // anonymous namespace
