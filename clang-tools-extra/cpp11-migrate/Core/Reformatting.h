//===-- Core/Reformatting.h - LibFormat integration -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the LibFormat integration used to reformat
/// migrated code.
///
//===----------------------------------------------------------------------===//

#ifndef CPP11_MIGRATE_REFORMATTING_H
#define CPP11_MIGRATE_REFORMATTING_H

#include "Core/Refactoring.h"
#include "clang/Format/Format.h"

class FileOverrides;
class ChangedRanges;

class Reformatter {
public:
  Reformatter(const clang::format::FormatStyle &Style) : Style(Style) {}

  /// \brief Reformat the changes made to the file overrides.
  ///
  /// This function will apply the state of files stored in \c FileState to \c
  /// SM.
  ///
  /// \param[in] FileState Files to reformat.
  /// \param[in] SM SourceManager for access to source files.
  /// \param[out] Replaces Container to store all reformatting replacements.
  void reformatChanges(const FileOverrides &FileState, clang::SourceManager &SM,
                       clang::tooling::ReplacementsVec &Replaces);

  /// \brief Produce a list of replacements to apply on \p FileName, only the
  /// ranges in \p Changes are replaced.
  ///
  /// Since this routine use \c clang::format::reformat() the rules that
  /// function applies to ranges also apply here.
  ///
  /// \param[in] FileName Name of file to reformat.
  /// \param[in] Changes Description of where changes were made to the file.
  /// \param[in] SM SourceManager required to create replacements.
  /// \param[out] FormatReplacements New reformatting replacements are appended
  /// to this container.
  void reformatSingleFile(const llvm::StringRef FileName,
                          const ChangedRanges &Changes,
                          clang::SourceManager &SM,
                          clang::tooling::ReplacementsVec &FormatReplacements);

private:
  clang::format::FormatStyle Style;
};

#endif // CPP11_MIGRATE_REFORMATTING_H
