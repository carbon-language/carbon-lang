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

#include "clang/Basic/Version.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <cstdlib>

using namespace std;

namespace clang {
  
llvm::StringRef getClangRepositoryPath() {
  static const char *Path = 0;
  if (Path)
    return Path;
  
  static char URL[] = "$URL$";
  char *End = strstr(URL, "/lib/Basic");
  if (End)
    *End = 0;
  
  End = strstr(URL, "/clang/tools/clang");
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


llvm::StringRef getClangRevision() {
#ifndef SVN_REVISION
  // Subversion was not available at build time?
  return llvm::StringRef();
#else
  static std::string revision;
  if (revision.empty()) {
    llvm::raw_string_ostream OS(revision);
    OS << strtol(SVN_REVISION, 0, 10);
  }
  return revision;
#endif
}

llvm::StringRef getClangFullRepositoryVersion() {
  static std::string buf;
  if (buf.empty()) {
    llvm::raw_string_ostream OS(buf);
    OS << getClangRepositoryPath();
    llvm::StringRef Revision = getClangRevision();
    if (!Revision.empty())
      OS << ' ' << Revision;
  }
  return buf;
}
  
const char *getClangFullVersion() {
  static std::string buf;
  if (buf.empty()) {
    llvm::raw_string_ostream OS(buf);
#ifdef CLANG_VENDOR
    OS << CLANG_VENDOR;
#endif
    OS << "clang version " CLANG_VERSION_STRING " ("
       << getClangFullRepositoryVersion() << ')';
  }
  return buf.c_str();
}
  
} // end namespace clang
