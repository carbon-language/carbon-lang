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

#include "clang/Tooling/ReplacementsYaml.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/system_error.h"
#include <vector>

namespace clang {

class DiagnosticsEngine;

namespace replace {

/// \brief Collection of TranslationUnitReplacements.
typedef std::vector<clang::tooling::TranslationUnitReplacements>
TUReplacements;

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
/// \param[in] Diagnostics DiagnosticsEngine used for error output.
///
/// \returns An error_code indicating success or failure in navigating the
/// directory structure.
llvm::error_code
collectReplacementsFromDirectory(const llvm::StringRef Directory,
                                 TUReplacements &TUs,
                                 clang::DiagnosticsEngine &Diagnostics);

/// \brief Deduplicate, check for conflicts, and apply all Replacements stored
/// in \c TUs. If conflicts occur, no Replacements are applied.
///
/// \param[in] TUs Collection of TranslationUnitReplacements to merge,
/// deduplicate, and test for conflicts.
/// \param[out] GroupedReplacements Container grouping all Replacements by the
/// file they target.
/// \param[in] Diagnostics DiagnosticsEngine used for error/warning output.
///
/// \returns \li true If all changes were applied successfully.
///          \li false If there were conflicts.
bool mergeAndDeduplicate(const TUReplacements &TUs,
                         FileToReplacementsMap &GroupedReplacements,
                         clang::DiagnosticsEngine &Diagnostics);

} // end namespace replace
} // end namespace clang

#endif // CPP11_MIGRATE_APPLYCHANGEDESCRIPTIONS_H
