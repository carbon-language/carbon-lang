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
  // Subversion was not available at build time?
  return 0;
#else
  return strtol(SVN_REVISION, 0, 10);
#endif
}

} // end namespace clang
