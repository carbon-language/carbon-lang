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

    StringRef res;
    SmallString<16> temp_store;
    if (error_code ec = sys::path::root_path(*i, res))
      ASSERT_FALSE(ec.message().c_str());
    outs() << "    root_path: " << res << '\n';
    if (error_code ec = sys::path::root_name(*i, res))
      ASSERT_FALSE(ec.message().c_str());
    outs() << "    root_name: " << res << '\n';
    if (error_code ec = sys::path::root_directory(*i, res))
      ASSERT_FALSE(ec.message().c_str());
    outs() << "    root_directory: " << res << '\n';
    if (error_code ec = sys::path::parent_path(*i, res))
      ASSERT_FALSE(ec.message().c_str());
    outs() << "    parent_path: " << res << '\n';

    temp_store = *i;
    if (error_code ec = sys::path::make_absolute(temp_store))
      ASSERT_FALSE(ec.message().c_str());
    outs() << "    make_absolute: " << temp_store << '\n';
    temp_store = *i;
    if (error_code ec = sys::path::remove_filename(temp_store))
      ASSERT_FALSE(ec.message().c_str());
    outs() << "    remove_filename: " << temp_store << '\n';

    outs().flush();
  }
}

} // anonymous namespace
