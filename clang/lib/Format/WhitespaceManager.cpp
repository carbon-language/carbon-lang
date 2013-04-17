//===--- WhitespaceManager.cpp - Format C++ code --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements WhitespaceManager class.
///
//===----------------------------------------------------------------------===//

#include "WhitespaceManager.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {
namespace format {

void WhitespaceManager::replaceWhitespace(const AnnotatedToken &Tok,
                                          unsigned NewLines, unsigned Spaces,
                                          unsigned WhitespaceStartColumn) {
  // 2+ newlines mean an empty line separating logic scopes.
  if (NewLines >= 2)
    alignComments();

  // Align line comments if they are trailing or if they continue other
  // trailing comments.
  if (Tok.isTrailingComment()) {
    SourceLocation TokenEndLoc = Tok.FormatTok.getStartOfNonWhitespace()
        .getLocWithOffset(Tok.FormatTok.TokenLength);
    // Remove the comment's trailing whitespace.
    if (Tok.FormatTok.TrailingWhiteSpaceLength != 0)
      Replaces.insert(tooling::Replacement(
          SourceMgr, TokenEndLoc, Tok.FormatTok.TrailingWhiteSpaceLength, ""));

    bool LineExceedsColumnLimit =
        Spaces + WhitespaceStartColumn + Tok.FormatTok.TokenLength >
        Style.ColumnLimit;
    // Align comment with other comments.
    if ((Tok.Parent != NULL || !Comments.empty()) && !LineExceedsColumnLimit) {
      StoredComment Comment;
      Comment.Tok = Tok.FormatTok;
      Comment.Spaces = Spaces;
      Comment.NewLines = NewLines;
      Comment.MinColumn =
          NewLines > 0 ? Spaces : WhitespaceStartColumn + Spaces;
      Comment.MaxColumn = Style.ColumnLimit - Tok.FormatTok.TokenLength;
      Comment.Untouchable = false;
      Comments.push_back(Comment);
      return;
    }
  }

  // If this line does not have a trailing comment, align the stored comments.
  if (Tok.Children.empty() && !Tok.isTrailingComment())
    alignComments();

  storeReplacement(Tok.FormatTok, getNewLineText(NewLines, Spaces));
}

void WhitespaceManager::replacePPWhitespace(const AnnotatedToken &Tok,
                                            unsigned NewLines, unsigned Spaces,
                                            unsigned WhitespaceStartColumn) {
  storeReplacement(Tok.FormatTok,
                   getNewLineText(NewLines, Spaces, WhitespaceStartColumn));
}

void WhitespaceManager::breakToken(const FormatToken &Tok, unsigned Offset,
                                   unsigned ReplaceChars, StringRef Prefix,
                                   StringRef Postfix, bool InPPDirective,
                                   unsigned Spaces,
                                   unsigned WhitespaceStartColumn) {
  std::string NewLineText;
  if (!InPPDirective)
    NewLineText = getNewLineText(1, Spaces);
  else
    NewLineText = getNewLineText(1, Spaces, WhitespaceStartColumn);
  std::string ReplacementText = (Prefix + NewLineText + Postfix).str();
  SourceLocation Location =
      Tok.getStartOfNonWhitespace().getLocWithOffset(Offset);
  Replaces.insert(
      tooling::Replacement(SourceMgr, Location, ReplaceChars, ReplacementText));
}

const tooling::Replacements &WhitespaceManager::generateReplacements() {
  alignComments();
  return Replaces;
}

void WhitespaceManager::addReplacement(const SourceLocation &SourceLoc,
                                       unsigned ReplaceChars, StringRef Text) {
  Replaces.insert(
      tooling::Replacement(SourceMgr, SourceLoc, ReplaceChars, Text));
}

void WhitespaceManager::addUntouchableComment(unsigned Column) {
  StoredComment Comment;
  Comment.MinColumn = Column;
  Comment.MaxColumn = Column;
  Comment.Untouchable = true;
  Comments.push_back(Comment);
}

std::string WhitespaceManager::getNewLineText(unsigned NewLines,
                                              unsigned Spaces) {
  return std::string(NewLines, '\n') + std::string(Spaces, ' ');
}

std::string WhitespaceManager::getNewLineText(unsigned NewLines,
                                              unsigned Spaces,
                                              unsigned WhitespaceStartColumn) {
  std::string NewLineText;
  if (NewLines > 0) {
    unsigned Offset =
        std::min<int>(Style.ColumnLimit - 1, WhitespaceStartColumn);
    for (unsigned i = 0; i < NewLines; ++i) {
      NewLineText += std::string(Style.ColumnLimit - Offset - 1, ' ');
      NewLineText += "\\\n";
      Offset = 0;
    }
  }
  return NewLineText + std::string(Spaces, ' ');
}

void WhitespaceManager::alignComments() {
  unsigned MinColumn = 0;
  unsigned MaxColumn = UINT_MAX;
  comment_iterator Start = Comments.begin();
  for (comment_iterator I = Start, E = Comments.end(); I != E; ++I) {
    if (I->MinColumn > MaxColumn || I->MaxColumn < MinColumn) {
      alignComments(Start, I, MinColumn);
      MinColumn = I->MinColumn;
      MaxColumn = I->MaxColumn;
      Start = I;
    } else {
      MinColumn = std::max(MinColumn, I->MinColumn);
      MaxColumn = std::min(MaxColumn, I->MaxColumn);
    }
  }
  alignComments(Start, Comments.end(), MinColumn);
  Comments.clear();
}

void WhitespaceManager::alignComments(comment_iterator I, comment_iterator E,
                                      unsigned Column) {
  while (I != E) {
    if (!I->Untouchable) {
      unsigned Spaces = I->Spaces + Column - I->MinColumn;
      storeReplacement(I->Tok, getNewLineText(I->NewLines, Spaces));
    }
    ++I;
  }
}

void WhitespaceManager::storeReplacement(const FormatToken &Tok,
                                         const std::string Text) {
  // Don't create a replacement, if it does not change anything.
  if (StringRef(SourceMgr.getCharacterData(Tok.WhiteSpaceStart),
                Tok.WhiteSpaceLength) == Text)
    return;

  Replaces.insert(tooling::Replacement(SourceMgr, Tok.WhiteSpaceStart,
                                       Tok.WhiteSpaceLength, Text));
}

} // namespace format
} // namespace clang
