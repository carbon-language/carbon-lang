//===- Version.cpp - Flang Version Number -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines several version-related utility functions for Flang.
//
//===----------------------------------------------------------------------===//

#include "flang/Basic/Version.h"
#include "flang/Basic/LLVM.h"
#include "flang/Config/config.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <cstring>

#ifdef HAVE_SVN_VERSION_INC
#  include "SVNVersion.inc"
#endif

namespace flang {

std::string getFlangRepositoryPath() {
#if defined(FLANG_REPOSITORY_STRING)
  return FLANG_REPOSITORY_STRING;
#else
#ifdef SVN_REPOSITORY
  StringRef URL(SVN_REPOSITORY);
#else
  StringRef URL("");
#endif

  // If the SVN_REPOSITORY is empty, try to use the SVN keyword. This helps us
  // pick up a tag in an SVN export, for example.
  StringRef SVNRepository("$URL$");
  if (URL.empty()) {
    URL = SVNRepository.slice(SVNRepository.find(':'),
                              SVNRepository.find("/lib/Basic"));
  }

  // Strip off version from a build from an integration branch.
  URL = URL.slice(0, URL.find("/src/tools/flang"));

  // Trim path prefix off, assuming path came from standard cfe path.
  size_t Start = URL.find("cfe/");
  if (Start != StringRef::npos)
    URL = URL.substr(Start + 4);

  return URL;
#endif
}

std::string getLLVMRepositoryPath() {
#ifdef LLVM_REPOSITORY
  StringRef URL(LLVM_REPOSITORY);
#else
  StringRef URL("");
#endif

  // Trim path prefix off, assuming path came from standard llvm path.
  // Leave "llvm/" prefix to distinguish the following llvm revision from the
  // flang revision.
  size_t Start = URL.find("llvm/");
  if (Start != StringRef::npos)
    URL = URL.substr(Start);

  return URL;
}

std::string getFlangRevision() {
#ifdef SVN_REVISION
  return SVN_REVISION;
#else
  return "";
#endif
}

std::string getLLVMRevision() {
#ifdef LLVM_REVISION
  return LLVM_REVISION;
#else
  return "";
#endif
}

std::string getFlangFullRepositoryVersion() {
  std::string buf;
  llvm::raw_string_ostream OS(buf);
  std::string Path = getFlangRepositoryPath();
  std::string Revision = getFlangRevision();
  if (!Path.empty() || !Revision.empty()) {
    OS << '(';
    if (!Path.empty())
      OS << Path;
    if (!Revision.empty()) {
      if (!Path.empty())
        OS << ' ';
      OS << Revision;
    }
    OS << ')';
  }
  // Support LLVM in a separate repository.
  std::string LLVMRev = getLLVMRevision();
  if (!LLVMRev.empty() && LLVMRev != Revision) {
    OS << " (";
    std::string LLVMRepo = getLLVMRepositoryPath();
    if (!LLVMRepo.empty())
      OS << LLVMRepo << ' ';
    OS << LLVMRev << ')';
  }
  return OS.str();
}

std::string getFlangFullVersion() {
  return getFlangToolFullVersion("flang");
}

std::string getFlangToolFullVersion(StringRef ToolName) {
  std::string buf;
  llvm::raw_string_ostream OS(buf);
#ifdef FLANG_VENDOR
  OS << FLANG_VENDOR;
#endif
  OS << ToolName << " version " FLANG_VERSION_STRING " "
     << getFlangFullRepositoryVersion();

  // If vendor supplied, include the base LLVM version as well.
#ifdef FLANG_VENDOR
  OS << " (based on " << BACKEND_PACKAGE_STRING << ")";
#endif

  return OS.str();
}

std::string getFlangFullCPPVersion() {
  // The version string we report in __VERSION__ is just a compacted version of
  // the one we report on the command line.
  std::string buf;
  llvm::raw_string_ostream OS(buf);
#ifdef FLANG_VENDOR
  OS << FLANG_VENDOR;
#endif
  OS << "Flang " FLANG_VERSION_STRING " " << getFlangFullRepositoryVersion();
  return OS.str();
}

} // end namespace flang
