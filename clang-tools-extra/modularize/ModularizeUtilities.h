//=====-- ModularizeUtilities.h - Utilities for modularize -*- C++ -*-======//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------===//
///
/// \file
/// \brief ModularizeUtilities class definition.
///
//===--------------------------------------------------------------------===//

#ifndef MODULARIZEUTILITIES_H
#define MODULARIZEUTILITIES_H

#include "Modularize.h"
#include <string>
#include <vector>

namespace Modularize {

/// Modularize utilities class.
/// Support functions and data for modularize.
class ModularizeUtilities {
public:
  // Input arguments.

  /// The input file paths.
  std::vector<std::string> InputFilePaths;
  /// The header prefix.
  llvm::StringRef HeaderPrefix;

  // Output data.

  // List of top-level header files.
  llvm::SmallVector<std::string, 32> HeaderFileNames;
  // Map of top-level header file dependencies.
  DependencyMap Dependencies;

  // Functions.

  /// Constructor.
  /// You can use the static createModularizeUtilities to create an instance
  /// of this object.
  /// \param InputPaths The input file paths.
  /// \param Prefix The headear path prefix.
  ModularizeUtilities(std::vector<std::string> &InputPaths,
                      llvm::StringRef Prefix);

  /// Create instance of ModularizeUtilities.
  /// \param InputPaths The input file paths.
  /// \param Prefix The headear path prefix.
  /// \returns Initialized ModularizeUtilities object.
  static ModularizeUtilities *createModularizeUtilities(
      std::vector<std::string> &InputPaths,
                  llvm::StringRef Prefix);

  /// Load header list and dependencies.
  /// \returns std::error_code.
  std::error_code loadAllHeaderListsAndDependencies();

protected:
  /// Load single header list and dependencies.
  /// \param InputPath The input file path.
  /// \returns std::error_code.
  std::error_code loadSingleHeaderListsAndDependencies(
      llvm::StringRef InputPath);
};

} // end namespace Modularize

#endif // MODULARIZEUTILITIES_H
