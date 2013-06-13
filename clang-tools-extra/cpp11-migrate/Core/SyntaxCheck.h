//===-- Core/SyntaxCheck.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file exposes functionaliy for doing a syntax-only check on
/// files with overridden contents.
///
//===----------------------------------------------------------------------===//
#ifndef CPP11_MIGRATE_SYNTAX_CHECK_H
#define CPP11_MIGRATE_SYNTAX_CHECK_H

#include <vector>
#include "Core/FileOverrides.h"

// Forward Declarations
namespace clang {
namespace tooling {
class CompilationDatabase;
} // namespace tooling
} // namespace clang

/// \brief Perform a syntax-only check over all files in \c SourcePaths using
/// options provided by \c Database using file contents from \c Overrides if
/// available.
extern bool doSyntaxCheck(const clang::tooling::CompilationDatabase &Database,
                          const std::vector<std::string> &SourcePaths,
                          const FileOverrides &Overrides);

#endif // CPP11_MIGRATE_SYNTAX_CHECK_H
