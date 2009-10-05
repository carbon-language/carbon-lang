//===- Version.cpp - Clang Version Number -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines several version-related utility functions for Clang.
//
//===----------------------------------------------------------------------===//
#include <cstring>
#include <cstdlib>
using namespace std;

namespace clang {
  
const char *getClangSubversionPath() {
  static const char *Path = 0;
  if (Path)
    return Path;
  
  static char URL[] = "$URL$";
  char *End = strstr(URL, "/lib/Basic");
  if (End)
    *End = 0;
  
  char *Begin = strstr(URL, "cfe/");
  if (Begin) {
    Path = Begin + 4;
    return Path;
  }
  
  Path = URL;
  return Path;
}


unsigned getClangSubversionRevision() {
#ifndef SVN_REVISION
  return 0;
#else
  // What follows is an evil trick. We can end up getting three different
  // kinds of results when asking for the Subversion information:
  //   - if SVN_REVISION is a number, we return that number (adding 0 to it is
  //     harmless).
  //   - if SVN_REVISION is "exported" (for an tree without any Subversion 
  //     info), we end up referencing the local variable "exported" and adding
  //     zero to it, and we return 0.
  //   - if SVN_REVISION is empty (because "svn info" returned no results),
  //     the "+" turns into a unary "+" operator and we return 0.
  //
  // Making this less ugly requires doing more computation in the CMake- and
  // Makefile-based build systems, with all sorts of checking to make sure we
  // don't end up breaking this build. It's better this way. Really.
  const unsigned exported = 0;
  (void)exported;
  return SVN_REVISION + 0;
#endif
}

} // end namespace clang
