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

#include "clang/Tooling/Refactoring.h"
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
  void adjustChangedRanges(const clang::tooling::Replacements &Replaces);

  /// \brief Iterators.
  /// @{
  const_iterator begin() const { return Ranges.begin(); }
  const_iterator end() const { return Ranges.end(); }
  /// @}

private:
  void coalesceRanges();

  RangeVec Ranges;
};

/// \brief Container for storing override information for a single headers.
struct HeaderOverride {
  HeaderOverride() {}
  HeaderOverride(llvm::StringRef FileName) : FileName(FileName) {}

  std::string FileName;
  std::string FileOverride;
  ChangedRanges Changes;
};

/// \brief Container mapping header file names to override information.
typedef llvm::StringMap<HeaderOverride> HeaderOverrides;

/// \brief Container storing the file content overrides for a source file and
/// any headers included by the source file either directly or indirectly to
/// which changes have been made.
class SourceOverrides {
public:
  SourceOverrides(llvm::StringRef MainFileName, bool TrackChanges);

  /// \brief Accessors.
  /// @{
  llvm::StringRef getMainFileName() const { return MainFileName; }
  llvm::StringRef getMainFileContent() const { return MainFileOverride; }
  const ChangedRanges &getChangedRanges() const { return MainFileChanges; }

  /// \brief Is file change tracking enabled?
  ///
  /// Tracking file changes can be useful to reformat the code for example.
  bool isTrackingFileChanges() const { return TrackChanges; }
  /// @}

  /// \brief Indicates if the source file has been overridden.
  ///
  /// It's possible for a source to remain unchanged while only headers are
  /// changed.
  bool isSourceOverriden() const { return !MainFileOverride.empty(); }

  /// \brief Override the file contents by applying all the replacements.
  ///
  /// \param Replaces The replacements to apply.
  /// \param SM A user provided SourceManager to be used when applying rewrites.
  void applyReplacements(clang::tooling::Replacements &Replaces,
                         clang::SourceManager &SM);
  void applyReplacements(clang::tooling::Replacements &Replaces);

  /// \brief Convenience function for applying this source's overrides to
  /// the given SourceManager.
  void applyOverrides(clang::SourceManager &SM) const;

  /// \brief Iterators.
  /// @{
  HeaderOverrides::iterator headers_begin() { return Headers.begin(); }
  HeaderOverrides::iterator headers_end() { return Headers.end(); }
  HeaderOverrides::const_iterator headers_begin() const {
    return Headers.begin();
  }
  HeaderOverrides::const_iterator headers_end() const { return Headers.end(); }
  /// @}

private:
  /// \brief Flatten the Rewriter buffers of \p Rewrite and store results as
  /// file content overrides.
  void applyRewrites(clang::Rewriter &Rewrite);

  /// \brief Adjust the changed ranges to reflect the parts of the files that
  /// have been replaced.
  void adjustChangedRanges(const clang::tooling::Replacements &Replaces);

  const std::string MainFileName;
  std::string MainFileOverride;
  const bool TrackChanges;
  ChangedRanges MainFileChanges;
  HeaderOverrides Headers;
};

/// \brief Maps source file names to content override information.
class FileOverrides {
public:
  typedef llvm::StringMap<SourceOverrides *> SourceOverridesMap;
  typedef SourceOverridesMap::const_iterator const_iterator;

  /// \brief Construct the SourceOverrides manager.
  ///
  /// \param TrackChanges Wether or not the \c SourceOverrides should keep track
  /// of changes. See \c SourceOverrides::isTrackingFileChanges().
  FileOverrides(bool TrackChanges) : TrackChanges(TrackChanges) {}
  ~FileOverrides();

  const_iterator find(llvm::StringRef Filename) const {
    return Overrides.find(Filename);
  }

  /// \brief Get the \c SourceOverrides for \p Filename, creating it if
  /// necessary.
  SourceOverrides &getOrCreate(llvm::StringRef Filename);

  /// \brief Iterators.
  /// @{
  const_iterator begin() const { return Overrides.begin(); }
  const_iterator end() const { return Overrides.end(); }
  /// @}

private:
  FileOverrides(const FileOverrides &) LLVM_DELETED_FUNCTION;
  FileOverrides &operator=(const FileOverrides &) LLVM_DELETED_FUNCTION;

  SourceOverridesMap Overrides;
  const bool TrackChanges;
};

/// \brief Generate a unique filename to store the replacements.
///
/// Generates a unique filename in the same directory as the header file. The
/// filename is based on the following model:
///
/// source.cpp_header.h_%%_%%_%%_%%_%%_%%.yaml
///
/// where all '%' will be replaced by a randomly chosen hex number.
///
/// @param SourceFile Full path to the source file.
/// @param HeaderFile Full path to the header file.
/// @param Result The resulting unique filename in the same directory as the
///        header file.
/// @param Error Description of the error if there is any.
/// @returns true if succeeded, false otherwise.
bool generateReplacementsFileName(llvm::StringRef SourceFile,
                                    llvm::StringRef HeaderFile,
                                    llvm::SmallVectorImpl<char> &Result,
                                    llvm::SmallVectorImpl<char> &Error);

#endif // CPP11_MIGRATE_FILE_OVERRIDES_H
