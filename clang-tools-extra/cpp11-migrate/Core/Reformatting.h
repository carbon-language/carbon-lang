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

#include "clang/Format/Format.h"

class SourceOverrides;
class ChangedRanges;

class Reformatter {
public:
  Reformatter(const clang::format::FormatStyle &Style) : Style(Style) {}

  /// \brief Reformat the changes made to the file overrides.
  ///
  /// \param Overrides Overriden source files to reformat. Note that since only
  /// the changes are reformatted, file change tracking has to be enabled.
  /// \param SM A SourceManager where the overridens files can be found.
  ///
  /// \sa \c SourceOverrides::isTrackingFileChanges()
  void reformatChanges(SourceOverrides &Overrides, clang::SourceManager &SM);

  /// \brief Overload of \c reformatChanges() providing it's own
  /// \c SourceManager.
  void reformatChanges(SourceOverrides &Overrides);

  /// \brief Produce a list of replacements to apply on \p FileName, only the
  /// ranges in \p Changes are replaced.
  ///
  /// Since this routine use \c clang::format::reformat() the rules that applies
  /// on the ranges are identical:
  ///
  /// \par
  /// Each range is extended on either end to its next bigger logic
  /// unit, i.e. everything that might influence its formatting or might be
  /// influenced by its formatting.
  /// -- \c clang::format::reformat()
  clang::tooling::Replacements reformatSingleFile(llvm::StringRef FileName,
                                                  const ChangedRanges &Changes,
                                                  clang::SourceManager &SM);

private:
  clang::format::FormatStyle Style;
};

#endif // CPP11_MIGRATE_REFORMATTING_H
