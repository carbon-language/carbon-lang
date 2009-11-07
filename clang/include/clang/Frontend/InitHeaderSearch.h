//===--- InitHeaderSearch.h - Initialize header search paths ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the InitHeaderSearch class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_INIT_HEADER_SEARCH_H_
#define LLVM_CLANG_FRONTEND_INIT_HEADER_SEARCH_H_

#include "clang/Lex/DirectoryLookup.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include <string>
#include <vector>

namespace clang {

class HeaderSearch;
class LangOptions;

/// InitHeaderSearch - This class makes it easier to set the search paths of
///  a HeaderSearch object. InitHeaderSearch stores several search path lists
///  internally, which can be sent to a HeaderSearch object in one swoop.
class InitHeaderSearch {
  std::vector<DirectoryLookup> IncludeGroup[4];
  HeaderSearch& Headers;
  bool Verbose;
  std::string isysroot;

public:
  /// IncludeDirGroup - Identifies the several search path
  /// lists stored internally.
  enum IncludeDirGroup {
    Quoted = 0,     ///< `#include ""` paths. Thing `gcc -iquote`.
    Angled,         ///< Paths for both `#include ""` and `#include <>`. (`-I`)
    System,         ///< Like Angled, but marks system directories.
    After           ///< Like System, but searched after the system directories.
  };

  InitHeaderSearch(HeaderSearch &HS,
      bool verbose = false, const std::string &iSysroot = "")
    : Headers(HS), Verbose(verbose), isysroot(iSysroot) {}

  /// AddPath - Add the specified path to the specified group list.
  void AddPath(const llvm::StringRef &Path, IncludeDirGroup Group,
               bool isCXXAware, bool isUserSupplied,
               bool isFramework, bool IgnoreSysRoot = false);

  /// AddGnuCPlusPlusIncludePaths - Add the necessary paths to suport a gnu
  ///  libstdc++.
  void AddGnuCPlusPlusIncludePaths(const std::string &Base, const char *Dir32,
                                   const char *Dir64,
                                   const llvm::Triple &triple);

  /// AddMinGWCPlusPlusIncludePaths - Add the necessary paths to suport a MinGW
  ///  libstdc++.
  void AddMinGWCPlusPlusIncludePaths(const std::string &Base,
                                     const char *Arch,
                                     const char *Version);

  /// AddDelimitedPaths - Add a list of paths delimited by the system PATH
  /// separator. The processing follows that of the CPATH variable for gcc.
  void AddDelimitedPaths(const char *String);

  // AddDefaultCIncludePaths - Add paths that should always be searched.
  void AddDefaultCIncludePaths(const llvm::Triple &triple);

  // AddDefaultCPlusPlusIncludePaths -  Add paths that should be searched when
  //  compiling c++.
  void AddDefaultCPlusPlusIncludePaths(const llvm::Triple &triple);

  /// AddDefaultSystemIncludePaths - Adds the default system include paths so
  ///  that e.g. stdio.h is found.
  void AddDefaultSystemIncludePaths(const LangOptions &Lang,
                                    const llvm::Triple &triple);

  /// Realize - Merges all search path lists into one list and send it to
  /// HeaderSearch.
  void Realize();
};

} // end namespace clang

#endif
