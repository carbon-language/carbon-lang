//===--- CustomCompilationDatabase.h --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains a hook to supply a custom \c CompilationDatabase
//  implementation.
//
//  The mechanism can be used by IDEs or non-public code bases to integrate with
//  their build system. Currently we support statically linking in an
//  implementation of \c findCompilationDatabaseForDirectory and enabling it
//  with -DUSE_CUSTOM_COMPILATION_DATABASE when compiling the Tooling library.
//
//  FIXME: The strategy forward is to provide a plugin system that can load
//  custom compilation databases and make enabling that a build option.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLING_CUSTOM_COMPILATION_DATABASE_H
#define LLVM_CLANG_TOOLING_CUSTOM_COMPILATION_DATABASE_H

#include "llvm/ADT/StringRef.h"

namespace clang {
namespace tooling {
class CompilationDatabase;

/// \brief Returns a CompilationDatabase for the given \c Directory.
///
/// \c Directory can be any directory within a project. This methods will
/// then try to find compilation database files in \c Directory or any of its
/// parents. If a compilation database cannot be found or loaded, returns NULL.
clang::tooling::CompilationDatabase *findCompilationDatabaseForDirectory(
  llvm::StringRef Directory);

} // namespace tooling
} // namespace clang

#endif // LLVM_CLANG_TOOLING_CUSTOM_COMPILATION_DATABASE_H
