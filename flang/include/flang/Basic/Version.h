//===- Version.h - Flang Version Number -------------------------*- C++ -*-===//
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
/// for Flang.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_BASIC_VERSION_H
#define LLVM_FLANG_BASIC_VERSION_H

#include "flang/Basic/Version.inc"
#include "llvm/ADT/StringRef.h"

namespace flang {
/// \brief Retrieves the repository path (e.g., Subversion path) that
/// identifies the particular Flang branch, tag, or trunk from which this
/// Flang was built.
std::string getFlangRepositoryPath();

/// \brief Retrieves the repository path from which LLVM was built.
///
/// This supports LLVM residing in a separate repository from flang.
std::string getLLVMRepositoryPath();

/// \brief Retrieves the repository revision number (or identifer) from which
/// this Flang was built.
std::string getFlangRevision();

/// \brief Retrieves the repository revision number (or identifer) from which
/// LLVM was built.
///
/// If Flang and LLVM are in the same repository, this returns the same
/// string as getFlangRevision.
std::string getLLVMRevision();

/// \brief Retrieves the full repository version that is an amalgamation of
/// the information in getFlangRepositoryPath() and getFlangRevision().
std::string getFlangFullRepositoryVersion();

/// \brief Retrieves a string representing the complete flang version,
/// which includes the flang version number, the repository version,
/// and the vendor tag.
std::string getFlangFullVersion();

/// \brief Like getFlangFullVersion(), but with a custom tool name.
std::string getFlangToolFullVersion(llvm::StringRef ToolName);

/// \brief Retrieves a string representing the complete flang version suitable
/// for use in the CPP __VERSION__ macro, which includes the flang version
/// number, the repository version, and the vendor tag.
std::string getFlangFullCPPVersion();
}  // namespace flang

#endif  // LLVM_FLANG_BASIC_VERSION_H
