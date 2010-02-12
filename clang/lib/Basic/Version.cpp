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
  static const char URL[] = "$URL$";
  const char *URLEnd = URL + strlen(URL);

  const char *End = strstr(URL, "/lib/Basic");
  if (End)
    URLEnd = End;

  End = strstr(URL, "/clang/tools/clang");
  if (End)
    URLEnd = End;

  const char *Begin = strstr(URL, "cfe/");
  if (Begin)
    return llvm::StringRef(Begin + 4, URLEnd - Begin - 4);

  return llvm::StringRef(URL, URLEnd - URL);
}

std::string getClangRevision() {
#ifndef SVN_REVISION
  // Subversion was not available at build time?
  return "";
#else
  std::string revision;
  llvm::raw_string_ostream OS(revision);
  OS << strtol(SVN_REVISION, 0, 10);
  return revision;
#endif
}

std::string getClangFullRepositoryVersion() {
  std::string buf;
  llvm::raw_string_ostream OS(buf);
  OS << getClangRepositoryPath();
  llvm::StringRef Revision = getClangRevision();
  if (!Revision.empty())
    OS << ' ' << Revision;
  return buf;
}
  
std::string getClangFullVersion() {
  std::string buf;
  llvm::raw_string_ostream OS(buf);
#ifdef CLANG_VENDOR
  OS << CLANG_VENDOR;
#endif
  OS << "clang version " CLANG_VERSION_STRING " ("
     << getClangFullRepositoryVersion() << ')';
  return buf;
}

} // end namespace clang
