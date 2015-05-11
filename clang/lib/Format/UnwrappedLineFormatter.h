//===--- UnwrappedLineFormatter.h - Format C++ code -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Implements a combinartorial exploration of all the different
/// linebreaks unwrapped lines can be formatted in.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_UNWRAPPEDLINEFORMATTER_H
#define LLVM_CLANG_LIB_FORMAT_UNWRAPPEDLINEFORMATTER_H

#include "ContinuationIndenter.h"
#include "clang/Format/Format.h"
#include <map>
#include <queue>
#include <string>

namespace clang {
namespace format {

class ContinuationIndenter;
class WhitespaceManager;

class UnwrappedLineFormatter {
public:
  UnwrappedLineFormatter(ContinuationIndenter *Indenter,
                         WhitespaceManager *Whitespaces,
                         const FormatStyle &Style,
                         const AdditionalKeywords &Keywords,
                         bool *IncompleteFormat)
      : Indenter(Indenter), Whitespaces(Whitespaces), Style(Style),
        Keywords(Keywords), IncompleteFormat(IncompleteFormat) {}

  /// \brief Format the current block and return the penalty.
  unsigned format(const SmallVectorImpl<AnnotatedLine *> &Lines,
                  bool DryRun = false, int AdditionalIndent = 0,
                  bool FixBadIndentation = false);

private:
  /// \brief Get the offset of the line relatively to the level.
  ///
  /// For example, 'public:' labels in classes are offset by 1 or 2
  /// characters to the left from their level.
  int getIndentOffset(const FormatToken &RootToken) {
    if (Style.Language == FormatStyle::LK_Java ||
        Style.Language == FormatStyle::LK_JavaScript)
      return 0;
    if (RootToken.isAccessSpecifier(false) ||
        RootToken.isObjCAccessSpecifier() ||
        (RootToken.is(Keywords.kw_signals) && RootToken.Next &&
         RootToken.Next->is(tok::colon)))
      return Style.AccessModifierOffset;
    return 0;
  }

  /// \brief Add a new line and the required indent before the first Token
  /// of the \c UnwrappedLine if there was no structural parsing error.
  void formatFirstToken(FormatToken &RootToken,
                        const AnnotatedLine *PreviousLine, unsigned IndentLevel,
                        unsigned Indent, bool InPPDirective);

  /// \brief Get the indent of \p Level from \p IndentForLevel.
  ///
  /// \p IndentForLevel must contain the indent for the level \c l
  /// at \p IndentForLevel[l], or a value < 0 if the indent for
  /// that level is unknown.
  unsigned getIndent(ArrayRef<int> IndentForLevel, unsigned Level);

  void join(AnnotatedLine &A, const AnnotatedLine &B);

  unsigned getColumnLimit(bool InPPDirective) const {
    // In preprocessor directives reserve two chars for trailing " \"
    return Style.ColumnLimit - (InPPDirective ? 2 : 0);
  }

  // Cache to store the penalty of formatting a vector of AnnotatedLines
  // starting from a specific additional offset. Improves performance if there
  // are many nested blocks.
  std::map<std::pair<const SmallVectorImpl<AnnotatedLine *> *, unsigned>,
           unsigned> PenaltyCache;

  ContinuationIndenter *Indenter;
  WhitespaceManager *Whitespaces;
  const FormatStyle &Style;
  const AdditionalKeywords &Keywords;
  bool *IncompleteFormat;
};
} // end namespace format
} // end namespace clang

#endif // LLVM_CLANG_LIB_FORMAT_UNWRAPPEDLINEFORMATTER_H
