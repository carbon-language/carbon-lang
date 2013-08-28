//===-- Core/ApplyChangeDescriptions.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the interface for finding and applying change
/// description files.
///
//===----------------------------------------------------------------------===//

#ifndef CPP11_MIGRATE_APPLYCHANGEDESCRIPTIONS_H
#define CPP11_MIGRATE_APPLYCHANGEDESCRIPTIONS_H

#include "clang/Tooling/Refactoring.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/system_error.h"
#include <vector>
#include <string>

namespace clang {

class DiagnosticsEngine;
class Rewriter;

namespace replace {

/// \brief Collection of TranslationUnitReplacements.
typedef std::vector<clang::tooling::TranslationUnitReplacements>
TUReplacements;

/// \brief Collection of TranslationUnitReplacement files.
typedef std::vector<std::string> TUReplacementFiles;

/// \brief Map mapping file name to Replacements targeting that file.
typedef llvm::StringMap<std::vector<clang::tooling::Replacement> >
FileToReplacementsMap;

/// \brief Recursively descends through a directory structure rooted at \p
/// Directory and attempts to deserialize *.yaml files as
/// TranslationUnitReplacements. All docs that successfully deserialize are
/// added to \p TUs.
///
/// Directories starting with '.' are ignored during traversal.
///
/// \param[in] Directory Directory to begin search for serialized
/// TranslationUnitReplacements.
/// \param[out] TUs Collection of all found and deserialized
/// TranslationUnitReplacements.
/// \param[out] TURFiles Collection of all TranslationUnitReplacement files
/// found in \c Directory.
/// \param[in] Diagnostics DiagnosticsEngine used for error output.
///
/// \returns An error_code indicating success or failure in navigating the
/// directory structure.
llvm::error_code
collectReplacementsFromDirectory(const llvm::StringRef Directory,
                                 TUReplacements &TUs,
                                 TUReplacementFiles &TURFiles,
                                 clang::DiagnosticsEngine &Diagnostics);

/// \brief Deduplicate, check for conflicts, and apply all Replacements stored
/// in \c TUs. If conflicts occur, no Replacements are applied.
///
/// \param[in] TUs Collection of TranslationUnitReplacements to merge,
/// deduplicate, and test for conflicts.
/// \param[out] GroupedReplacements Container grouping all Replacements by the
/// file they target.
/// \param[in] SM SourceManager required for conflict reporting.
///
/// \returns \li true If all changes were applied successfully.
///          \li false If there were conflicts.
bool mergeAndDeduplicate(const TUReplacements &TUs,
                         FileToReplacementsMap &GroupedReplacements,
                         clang::SourceManager &SM);

/// \brief Apply all replacements in \c GroupedReplacements.
///
/// \param[in] GroupedReplacements Deduplicated and conflict free Replacements
/// to apply.
/// \param[out] Rewrites The results of applying replacements will be applied
/// to this Rewriter.
///
/// \returns \li true If all changes were applied successfully.
///          \li false If a replacement failed to apply.
bool applyReplacements(const FileToReplacementsMap &GroupedReplacements,
                       clang::Rewriter &Rewrites);

/// \brief Write the contents of \c FileContents to disk. Keys of the map are
/// filenames and values are the new contents for those files.
///
/// \param[in] Rewrites Rewriter containing written files to write to disk.
bool writeFiles(const clang::Rewriter &Rewrites);

/// \brief Delete the replacement files.
///
/// \param[in] Files Replacement files to delete.
/// \param[in] Diagnostics DiagnosticsEngine used for error output.
///
/// \returns \li true If all files have been deleted successfully.
///          \li false If at least one or more failures occur when deleting
/// files.
bool deleteReplacementFiles(const TUReplacementFiles &Files,
                            clang::DiagnosticsEngine &Diagnostics);

} // end namespace replace
} // end namespace clang

#endif // CPP11_MIGRATE_APPLYCHANGEDESCRIPTIONS_H
