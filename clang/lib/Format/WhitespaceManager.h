//===--- WhitespaceManager.h - Format C++ code ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WhitespaceManager class manages whitespace around tokens and their
/// replacements.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FORMAT_WHITESPACEMANAGER_H
#define LLVM_CLANG_FORMAT_WHITESPACEMANAGER_H

#include "TokenAnnotator.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include <string>

namespace clang {
namespace format {

/// \brief Manages the whitespaces around tokens and their replacements.
///
/// This includes special handling for certain constructs, e.g. the alignment of
/// trailing line comments.
class WhitespaceManager {
public:
  WhitespaceManager(SourceManager &SourceMgr, const FormatStyle &Style)
      : SourceMgr(SourceMgr), Style(Style) {}

  /// \brief Replaces the whitespace in front of \p Tok. Only call once for
  /// each \c AnnotatedToken.
  void replaceWhitespace(const AnnotatedToken &Tok, unsigned NewLines,
                         unsigned Spaces, unsigned WhitespaceStartColumn);

  /// \brief Like \c replaceWhitespace, but additionally adds right-aligned
  /// backslashes to escape newlines inside a preprocessor directive.
  ///
  /// This function and \c replaceWhitespace have the same behavior if
  /// \c Newlines == 0.
  void replacePPWhitespace(const AnnotatedToken &Tok, unsigned NewLines,
                           unsigned Spaces, unsigned WhitespaceStartColumn);

  /// \brief Inserts a line break into the middle of a token.
  ///
  /// Will break at \p Offset inside \p Tok, putting \p Prefix before the line
  /// break and \p Postfix before the rest of the token starts in the next line.
  ///
  /// \p InPPDirective, \p Spaces, \p WhitespaceStartColumn and \p Style are
  /// used to generate the correct line break.
  void breakToken(const FormatToken &Tok, unsigned Offset,
                  unsigned ReplaceChars, StringRef Prefix, StringRef Postfix,
                  bool InPPDirective, unsigned Spaces,
                  unsigned WhitespaceStartColumn);

  /// \brief Returns all the \c Replacements created during formatting.
  const tooling::Replacements &generateReplacements();

  void addReplacement(const SourceLocation &SourceLoc, unsigned ReplaceChars,
                      StringRef Text);

  void addUntouchableComment(unsigned Column);

private:
  std::string getNewLineText(unsigned NewLines, unsigned Spaces);

  std::string getNewLineText(unsigned NewLines, unsigned Spaces,
                             unsigned WhitespaceStartColumn);

  /// \brief Structure to store a comment for later layout and alignment.
  struct StoredComment {
    FormatToken Tok;
    unsigned MinColumn;
    unsigned MaxColumn;
    unsigned NewLines;
    unsigned Spaces;
    bool Untouchable;
  };
  SmallVector<StoredComment, 16> Comments;
  typedef SmallVector<StoredComment, 16>::iterator comment_iterator;

  /// \brief Try to align all stashed comments.
  void alignComments();

  /// \brief Put all the comments between \p I and \p E into \p Column.
  void alignComments(comment_iterator I, comment_iterator E, unsigned Column);

  /// \brief Stores \p Text as the replacement for the whitespace in front of
  /// \p Tok.
  void storeReplacement(const FormatToken &Tok, const std::string Text);

  SourceManager &SourceMgr;
  tooling::Replacements Replaces;
  const FormatStyle &Style;
};

} // namespace format
} // namespace clang

#endif // LLVM_CLANG_FORMAT_WHITESPACEMANAGER_H
