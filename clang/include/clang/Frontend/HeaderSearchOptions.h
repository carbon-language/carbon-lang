//===--- HeaderSearchOptions.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_HEADERSEARCHOPTIONS_H
#define LLVM_CLANG_FRONTEND_HEADERSEARCHOPTIONS_H

// FIXME: Drop this dependency.
#include "clang/Frontend/InitHeaderSearch.h"
#include "llvm/ADT/StringRef.h"

namespace clang {

/// HeaderSearchOptions - Helper class for storing options related to the
/// initialization of the HeaderSearch object.
class HeaderSearchOptions {
public:
  struct Entry {
    std::string Path;
    InitHeaderSearch::IncludeDirGroup Group;
    unsigned IsCXXAware : 1;
    unsigned IsUserSupplied : 1;
    unsigned IsFramework : 1;
    unsigned IgnoreSysRoot : 1;

    Entry(llvm::StringRef _Path, InitHeaderSearch::IncludeDirGroup _Group,
          bool _IsCXXAware, bool _IsUserSupplied, bool _IsFramework,
          bool _IgnoreSysRoot)
      : Path(_Path), Group(_Group), IsCXXAware(_IsCXXAware),
        IsUserSupplied(_IsUserSupplied), IsFramework(_IsFramework),
        IgnoreSysRoot(_IgnoreSysRoot) {}
  };

  /// If non-empty, the directory to use as a "virtual system root" for include
  /// paths.
  std::string Sysroot;

  /// User specified include entries.
  std::vector<Entry> UserEntries;

  /// A (system-path) delimited list of include paths to be added from the
  /// environment following the user specified includes (but prior to builtin
  /// and standard includes). This is parsed in the same manner as the CPATH
  /// environment variable for gcc.
  std::string EnvIncPath;

  /// A (system-path) delimited list of include paths to be added from the
  /// environment following the user specified includes and the \see EnvIncPath
  /// includes (but prior to builtin and standard includes). This is parsed in
  /// the same manner as the CPATH environment variable for gcc.
  std::string LangEnvIncPath;

  /// If non-empty, the path to the compiler builtin include directory, which
  /// will be searched following the user and environment includes.
  std::string BuiltinIncludePath;

  /// Include the system standard include search directories.
  unsigned UseStandardIncludes : 1;

  /// Whether header search information should be output as for -v.
  unsigned Verbose : 1;

public:
  HeaderSearchOptions(llvm::StringRef _Sysroot = "")
    : Sysroot(_Sysroot), UseStandardIncludes(true) {}

  /// AddPath - Add the \arg Path path to the specified \arg Group list.
  void AddPath(llvm::StringRef Path, InitHeaderSearch::IncludeDirGroup Group,
               bool IsCXXAware, bool IsUserSupplied,
               bool IsFramework, bool IgnoreSysRoot = false) {
    UserEntries.push_back(Entry(Path, Group, IsCXXAware, IsUserSupplied,
                                IsFramework, IgnoreSysRoot));
  }
};

} // end namespace clang

#endif
