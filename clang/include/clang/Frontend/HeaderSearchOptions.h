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

#include "llvm/ADT/StringRef.h"
#include <vector>

namespace clang {

namespace frontend {
  /// IncludeDirGroup - Identifiers the group a include entry belongs to, which
  /// represents its relative positive in the search list.
  enum IncludeDirGroup {
    Quoted = 0,     ///< `#include ""` paths. Thing `gcc -iquote`.
    Angled,         ///< Paths for both `#include ""` and `#include <>`. (`-I`)
    System,         ///< Like Angled, but marks system directories.
    After           ///< Like System, but searched after the system directories.
  };
}

/// HeaderSearchOptions - Helper class for storing options related to the
/// initialization of the HeaderSearch object.
class HeaderSearchOptions {
public:
  struct Entry {
    std::string Path;
    frontend::IncludeDirGroup Group;
    unsigned IsUserSupplied : 1;
    unsigned IsFramework : 1;
    
    /// IsSysRootRelative - This is true if an absolute path should be treated
    /// relative to the sysroot, or false if it should always be the absolute
    /// path.
    unsigned IsSysRootRelative : 1;

    Entry(llvm::StringRef path, frontend::IncludeDirGroup group,
          bool isUserSupplied, bool isFramework, bool isSysRootRelative)
      : Path(path), Group(group), IsUserSupplied(isUserSupplied),
        IsFramework(isFramework), IsSysRootRelative(isSysRootRelative) {}
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

  /// Per-language environmental include paths, see \see EnvIncPath.
  std::string CEnvIncPath;
  std::string ObjCEnvIncPath;
  std::string CXXEnvIncPath;
  std::string ObjCXXEnvIncPath;

  /// The directory which holds the compiler resource files (builtin includes,
  /// etc.).
  std::string ResourceDir;

  /// Include the compiler builtin includes.
  unsigned UseBuiltinIncludes : 1;

  /// Include the system standard include search directories.
  unsigned UseStandardIncludes : 1;

  /// Include the system standard C++ library include search directories.
  unsigned UseStandardCXXIncludes : 1;

  /// Whether header search information should be output as for -v.
  unsigned Verbose : 1;

public:
  HeaderSearchOptions(llvm::StringRef _Sysroot = "/")
    : Sysroot(_Sysroot), UseBuiltinIncludes(true),
      UseStandardIncludes(true), UseStandardCXXIncludes(true),
      Verbose(false) {}

  /// AddPath - Add the \arg Path path to the specified \arg Group list.
  void AddPath(llvm::StringRef Path, frontend::IncludeDirGroup Group,
               bool IsUserSupplied, bool IsFramework, bool IsSysRootRelative) {
    UserEntries.push_back(Entry(Path, Group, IsUserSupplied, IsFramework,
                                IsSysRootRelative));
  }
};

} // end namespace clang

#endif
