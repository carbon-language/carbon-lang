//===- lld/Config/Version.h - LLD Version Number ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines version macros and version-related utility functions
/// for lld.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_VERSION_H
#define LLD_VERSION_H

#include "lld/Config/Version.inc"
#include "llvm/ADT/StringRef.h"
#include <string>

/// \brief Helper macro for LLD_VERSION_STRING.
#define LLD_MAKE_VERSION_STRING2(X) #X

/// \brief Helper macro for LLD_VERSION_STRING.
#define LLD_MAKE_VERSION_STRING(X, Y) LLD_MAKE_VERSION_STRING2(X.Y)

/// \brief A string that describes the lld version number, e.g., "1.0".
#define LLD_VERSION_STRING                                                     \
  LLD_MAKE_VERSION_STRING(LLD_VERSION_MAJOR, LLD_VERSION_MINOR)

namespace lld {
/// \brief Retrieves the repository path (e.g., Subversion path) that
/// identifies the particular lld branch, tag, or trunk from which this
/// lld was built.
llvm::StringRef getLLDRepositoryPath();

/// \brief Retrieves the repository revision number (or identifer) from which
/// this lld was built.
llvm::StringRef getLLDRevision();

/// \brief Retrieves the full repository version that is an amalgamation of
/// the information in getLLDRepositoryPath() and getLLDRevision().
std::string getLLDRepositoryVersion();

/// \brief Retrieves a string representing the complete lld version.
llvm::StringRef getLLDVersion();
}

#endif // LLD_VERSION_H
