//===-- Core/FileOverrides.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides types and functionality for dealing with source
/// and header file content overrides.
///
//===----------------------------------------------------------------------===//

#ifndef CPP11_MIGRATE_FILE_OVERRIDES_H
#define CPP11_MIGRATE_FILE_OVERRIDES_H

#include "llvm/ADT/StringRef.h"

#include <map>
#include <string>

// Forward Declarations
namespace clang {
class SourceManager;
class FileManager;
} // namespace clang

/// \brief Container for storing override information for a single headers.
struct HeaderOverride {
  HeaderOverride() {}
  HeaderOverride(llvm::StringRef FileName) : FileName(FileName) {}

  std::string FileName;
  std::string FileOverride;
};

/// \brief Container mapping header file names to override information.
typedef std::map<std::string, HeaderOverride> HeaderOverrides;

/// \brief Container storing the file content overrides for a source file.
struct SourceOverrides {
  SourceOverrides(const std::string &MainFileName)
      : MainFileName(MainFileName) {}

  /// \brief Convenience function for applying this source's overrides to
  /// the given SourceManager.
  void applyOverrides(clang::SourceManager &SM) const;

  /// \brief Indicates if the source file has been overridden.
  ///
  /// It's possible for a source to remain unchanged while only headers are
  /// changed.
  bool isSourceOverriden() const { return !MainFileOverride.empty(); }

  std::string MainFileName;
  std::string MainFileOverride;
  HeaderOverrides Headers;
};

/// \brief Maps source file names to content override information.
typedef std::map<std::string, SourceOverrides> FileOverrides;

#endif // CPP11_MIGRATE_FILE_OVERRIDES_H
