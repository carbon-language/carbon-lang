//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class path

// path lexically_normal() const;

#include "filesystem_include.h"
#include <cstdio>
#include <string>

#include "test_macros.h"
#include "count_new.h"
#include "filesystem_test_helper.h"


int main(int, char**) {
  // clang-format off
  struct {
    std::string input;
    std::string expect;
  } TestCases[] = {
      {"", ""},
      {"/a/b/c", "/a/b/c"},
      {"/a/b//c", "/a/b/c"},
      {"foo/./bar/..", "foo/"},
      {"foo/.///bar/../", "foo/"},
      {"/a/b/", "/a/b/"},
      {"a/b", "a/b"},
      {"a/b/.", "a/b/"},
      {"a/b/./", "a/b/"},
      {"a/..", "."},
      {".", "."},
      {"./", "."},
      {"./.", "."},
      {"./..", ".."},
      {"..", ".."},
      {"../..", "../.."},
      {"/../", "/"},
      {"/../..", "/"},
      {"/../../", "/"},
      {"..", ".."},
      {"../", ".."},
      {"/a/b/c/../", "/a/b/"},
      {"/a/b/./", "/a/b/"},
      {"/a/b/c/../d", "/a/b/d"},
      {"/a/b/c/../d/", "/a/b/d/"},
#ifdef _WIN32
      {"//a/", "//a/"},
      {"//a/b/", "//a/b/"},
      {"//a/b/.", "//a/b/"},
      {"//a/..", "//a/"},
#else
      {"//a/", "/a/"},
      {"//a/b/", "/a/b/"},
      {"//a/b/.", "/a/b/"},
      {"//a/..", "/"},
#endif
      ///===---------------------------------------------------------------===//
      /// Tests specifically for the clauses under [fs.path.generic]p6
      ///===---------------------------------------------------------------===//
      // p1: If the path is empty, stop.
      {"", ""},
      // p2: Replace each slash character in the root-name with a preferred
      // separator.
      {"NO_ROOT_NAME_ON_LINUX", "NO_ROOT_NAME_ON_LINUX"},
      // p3: Replace each directory-separator with a preferred-separator.
      // [ Note: The generic pathname grammar ([fs.path.generic]) defines
      //   directory-separator as one or more slashes and preferred-separators.
      //   — end note ]
      {"/", "/"},
      {"//", "/"},
      {"///", "/"},
      {"a/b", "a/b"},
      {"a//b", "a/b"},
      {"a///b", "a/b"},
      {"a/b/", "a/b/"},
      {"a/b//", "a/b/"},
      {"a/b///", "a/b/"},
      {"///a////b//////", "/a/b/"},
      // p4: Remove each dot filename and any immediately following directory
      // separators
      {"foo/.", "foo/"},
      {"foo/./bar/.", "foo/bar/"},
      {"./foo/././bar/./", "foo/bar/"},
      {".///foo//.////./bar/.///", "foo/bar/"},
      // p5: As long as any appear, remove a non-dot-dot filename immediately
      // followed by a directory-separator and a dot-dot filename, along with
      // any immediately following directory separator.
      {"foo/..", "."},
      {"foo/../", "."},
      {"foo/bar/..", "foo/"},
      {"foo/bar/../", "foo/"},
      {"foo/bar/../..", "."},
      {"foo/bar/../../", "."},
      {"foo/bar/baz/../..", "foo/"},
      {"foo/bar/baz/../../", "foo/"},
      {"foo/bar/./..", "foo/"},
      {"foo/bar/./../", "foo/"},
      // p6: If there is a root-directory, remove all dot-dot filenames and any
      // directory-separators immediately following them. [ Note: These dot-dot
      // filenames attempt to refer to nonexistent parent directories. — end note ]
      {"/..", "/"},
      {"/../", "/"},
      {"/foo/../..", "/"},
      {"/../foo", "/foo"},
      {"/../foo/../..", "/"},
      // p7: If the last filename is dot-dot, remove any trailing
      // directory-separator.
      {"../", ".."},
      {"../../", "../.."},
      {"foo/../bar/../..///", ".."},
      {"foo/../bar/..//..///../", "../.."},
      // p8: If the path is empty, add a dot
      {".", "."},
      {"./", "."},
      {"foo/..", "."}
  };
  // clang-format on
  int ID = 0;
  bool Failed = false;
  for (auto& TC : TestCases) {
    ++ID;
    fs::path p(TC.input);
    const fs::path output = p.lexically_normal();
    fs::path expect(TC.expect);
    expect.make_preferred();
    if (!PathEq(output, expect)) {
      Failed = true;
      std::fprintf(stderr, "TEST CASE #%d FAILED:\n"
                  "  Input: '%s'\n"
                  "  Expected: '%s'\n"
                  "  Output: '%s'\n",
        ID, TC.input.c_str(), expect.string().c_str(), output.string().c_str());
    }
  }
  return Failed;
}
