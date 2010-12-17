//===- Version.h - Clang Version Number -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines version macros and version-related utility functions 
// for Clang.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_VERSION_H
#define LLVM_CLANG_BASIC_VERSION_H

#include "llvm/ADT/StringRef.h"

#include "clang/Basic/Version.inc"

/// \brief Helper macro for CLANG_VERSION_STRING.
#define CLANG_MAKE_VERSION_STRING2(X) #X

#ifdef CLANG_VERSION_PATCHLEVEL
/// \brief Helper macro for CLANG_VERSION_STRING.
#define CLANG_MAKE_VERSION_STRING(X,Y,Z) CLANG_MAKE_VERSION_STRING2(X.Y.Z)

/// \brief A string that describes the Clang version number, e.g.,
/// "1.0".
#define CLANG_VERSION_STRING \
  CLANG_MAKE_VERSION_STRING(CLANG_VERSION_MAJOR,CLANG_VERSION_MINOR,CLANG_VERSION_PATCHLEVEL)
#else
/// \brief Helper macro for CLANG_VERSION_STRING.
#define CLANG_MAKE_VERSION_STRING(X,Y) CLANG_MAKE_VERSION_STRING2(X.Y)

/// \brief A string that describes the Clang version number, e.g.,
/// "1.0".
#define CLANG_VERSION_STRING \
  CLANG_MAKE_VERSION_STRING(CLANG_VERSION_MAJOR,CLANG_VERSION_MINOR)
#endif

/// \brief A horrible hack to work around projects that depend upon gcc
/// compatible compilers having versions that resemble gcc.
#define GCC_COMPAT_VERSION_STRING CLANG_MAKE_VERSION_STRING(4,2)

namespace clang {
  /// \brief Retrieves the repository path (e.g., Subversion path) that 
  /// identifies the particular Clang branch, tag, or trunk from which this
  /// Clang was built.
  std::string getClangRepositoryPath();
  
  /// \brief Retrieves the repository revision number (or identifer) from which
  ///  this Clang was built.
  std::string getClangRevision();
  
  /// \brief Retrieves the full repository version that is an amalgamation of
  ///  the information in getClangRepositoryPath() and getClangRevision().
  std::string getClangFullRepositoryVersion();
  
  /// \brief Retrieves a string representing the complete clang version,
  ///   which includes the clang version number, the repository version, 
  ///   and the vendor tag.
  std::string getClangFullVersion();
}

#endif // LLVM_CLANG_BASIC_VERSION_H
