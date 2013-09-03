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

#include "Core/Refactoring.h"
#include "clang-apply-replacements/Tooling/ApplyReplacements.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "llvm/ADT/StringMap.h"

// Forward Declarations
namespace llvm {
template <typename T>
class SmallVectorImpl;
} // namespace llvm
namespace clang {
class SourceManager;
class Rewriter;
} // namespace clang

/// \brief Class encapsulating a list of \c tooling::Range with some
/// convenience methods.
///
/// The ranges stored are used to keep track of the overriden parts of a file.
class ChangedRanges {
  typedef std::vector<clang::tooling::Range> RangeVec;

public:
  typedef RangeVec::const_iterator const_iterator;

  /// \brief Create new ranges from the replacements and adjust existing one
  /// to remove replaced parts.
  ///
  /// Note that all replacements should come from the same file.
  void adjustChangedRanges(const clang::tooling::ReplacementsVec &Replaces);

  /// \brief Iterators.
  /// \{
  const_iterator begin() const { return Ranges.begin(); }
  const_iterator end() const { return Ranges.end(); }
  /// \}

private:
  void coalesceRanges();

  RangeVec Ranges;
};

/// \brief Maintains current state of transformed files and tracks source ranges
/// where changes have been made.
class FileOverrides {
public:
  /// \brief Maps file names to file contents.
  typedef llvm::StringMap<std::string> FileStateMap;

  /// \brief Maps file names to change tracking info for a file.
  typedef llvm::StringMap<ChangedRanges> ChangeMap;


  /// \brief Override file contents seen by \c SM for all files stored by this
  /// object.
  void applyOverrides(clang::SourceManager &SM) const;

  /// \brief Update change tracking information based on replacements stored in
  /// \c Replaces.
  void
  adjustChangedRanges(const clang::replace::FileToReplacementsMap &Replaces);

  /// \brief Accessor for change tracking information.
  const ChangeMap &getChangedRanges() const {
    return ChangeTracking;
  }

  /// \brief Coalesce changes stored in \c Rewrites and replace file contents 
  /// overrides stored in this object.
  ///
  /// \param Rewrites Rewriter containing changes to files.
  void updateState(const clang::Rewriter &Rewrites);

  /// \brief Accessor for current file state.
  const FileStateMap &getState() const { return FileStates; }

  /// \brief Write all file contents overrides to disk.
  ///
  /// \param Diagnostics DiagnosticsEngine for error output.
  ///
  /// \returns \li true if all files with overridden file contents were written
  ///              to disk successfully.
  ///          \li false if any failure occurred.
  bool writeToDisk(clang::DiagnosticsEngine &Diagnostics) const;

private:
  FileStateMap FileStates;
  ChangeMap ChangeTracking;
};

/// \brief Generate a unique filename to store the replacements.
///
/// Generates a unique filename in the same directory as \c MainSourceFile. The
/// filename is generated following this pattern:
///
/// MainSourceFile_%%_%%_%%_%%_%%_%%.yaml
///
/// where all '%' will be replaced by a randomly chosen hex number.
///
/// \param[in] MainSourceFile Full path to the source file.
/// \param[out] Result The resulting unique filename in the same directory as
///             the \c MainSourceFile.
/// \param[out] Error If an error occurs a description of that error is
///             placed in this string.
/// \returns true on success, false if a unique file name could not be created.
bool generateReplacementsFileName(const llvm::StringRef MainSourceFile,
                                  llvm::SmallVectorImpl<char> &Result,
                                  llvm::SmallVectorImpl<char> &Error);

#endif // CPP11_MIGRATE_FILE_OVERRIDES_H
