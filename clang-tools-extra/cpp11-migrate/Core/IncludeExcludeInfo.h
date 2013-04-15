//===-- Core/IncludeExcludeInfo.h - IncludeExclude class def'n --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the definition for the IncludeExcludeInfo class
/// to handle the include and exclude command line options.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_INCLUDEEXCLUDEINFO_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_INCLUDEEXCLUDEINFO_H

#include "llvm/ADT/StringRef.h"
#include <vector>

/// \brief Class encapsulating the handling of include and exclude paths
/// provided by the user through command line options.
class IncludeExcludeInfo {
public:
  /// \brief Determine if the given file is safe to transform.
  ///
  /// \a Include and \a Exclude must be formatted as a comma-seperated list.
  IncludeExcludeInfo(llvm::StringRef Include, llvm::StringRef Exclude);

  /// \brief Determine if the given filepath is in the list of include paths but
  /// not in the list of exclude paths.
  bool isFileIncluded(llvm::StringRef FilePath);

private:
  std::vector<std::string> IncludeList;
  std::vector<std::string> ExcludeList;
};

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_INCLUDEEXCLUDEINFO_H
