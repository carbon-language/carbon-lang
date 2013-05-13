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
  if (NewLines > 0)
    alignEscapedNewlines();

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
    if ((Tok.Parent != NULL || !Comments.empty()) &&
        !LineExceedsColumnLimit) {
      unsigned MinColumn =
          NewLines > 0 ? Spaces : WhitespaceStartColumn + Spaces;
      unsigned MaxColumn = Style.ColumnLimit - Tok.FormatTok.TokenLength;
      Comments.push_back(StoredToken(
          Tok.FormatTok.WhiteSpaceStart, Tok.FormatTok.WhiteSpaceLength,
          MinColumn, MaxColumn, NewLines, Spaces));
      return;
    }
  }

  // If this line does not have a trailing comment, align the stored comments.
  if (Tok.Children.empty() && !Tok.isTrailingComment())
    alignComments();

  storeReplacement(Tok.FormatTok.WhiteSpaceStart,
                   Tok.FormatTok.WhiteSpaceLength,
                   getNewLineText(NewLines, Spaces));
}

void WhitespaceManager::replacePPWhitespace(const AnnotatedToken &Tok,
                                            unsigned NewLines, unsigned Spaces,
                                            unsigned WhitespaceStartColumn) {
  if (NewLines == 0) {
    replaceWhitespace(Tok, NewLines, Spaces, WhitespaceStartColumn);
  } else {
    // The earliest position for "\" is 2 after the last token.
    unsigned MinColumn = WhitespaceStartColumn + 2;
    unsigned MaxColumn = Style.ColumnLimit;
    EscapedNewlines.push_back(StoredToken(
        Tok.FormatTok.WhiteSpaceStart, Tok.FormatTok.WhiteSpaceLength,
        MinColumn, MaxColumn, NewLines, Spaces));
  }
}

void WhitespaceManager::breakToken(const FormatToken &Tok, unsigned Offset,
                                   unsigned ReplaceChars, StringRef Prefix,
                                   StringRef Postfix, bool InPPDirective,
                                   unsigned Spaces,
                                   unsigned WhitespaceStartColumn) {
  SourceLocation Location =
      Tok.getStartOfNonWhitespace().getLocWithOffset(Offset);
  if (InPPDirective) {
    // The earliest position for "\" is 2 after the last token.
    unsigned MinColumn = WhitespaceStartColumn + 2;
    unsigned MaxColumn = Style.ColumnLimit;
    StoredToken StoredTok = StoredToken(Location, ReplaceChars, MinColumn,
                                        MaxColumn, /*NewLines=*/ 1, Spaces);
    StoredTok.Prefix = Prefix;
    StoredTok.Postfix = Postfix;
    EscapedNewlines.push_back(StoredTok);
  } else {
    std::string ReplacementText =
        (Prefix + getNewLineText(1, Spaces) + Postfix).str();
    Replaces.insert(tooling::Replacement(SourceMgr, Location, ReplaceChars,
                                         ReplacementText));
  }
}

const tooling::Replacements &WhitespaceManager::generateReplacements() {
  alignComments();
  alignEscapedNewlines();
  return Replaces;
}

void WhitespaceManager::addReplacement(const SourceLocation &SourceLoc,
                                       unsigned ReplaceChars, StringRef Text) {
  Replaces.insert(
      tooling::Replacement(SourceMgr, SourceLoc, ReplaceChars, Text));
}

void WhitespaceManager::addUntouchableComment(unsigned Column) {
  StoredToken Tok = StoredToken(SourceLocation(), 0, Column, Column, 0, 0);
  Tok.Untouchable = true;
  Comments.push_back(Tok);
}

std::string WhitespaceManager::getNewLineText(unsigned NewLines,
                                              unsigned Spaces) {
  return std::string(NewLines, '\n') + getIndentText(Spaces);
}

std::string WhitespaceManager::getNewLineText(unsigned NewLines,
                                              unsigned Spaces,
                                              unsigned WhitespaceStartColumn,
                                              unsigned EscapedNewlineColumn) {
  std::string NewLineText;
  if (NewLines > 0) {
    unsigned Offset =
        std::min<int>(EscapedNewlineColumn - 1, WhitespaceStartColumn);
    for (unsigned i = 0; i < NewLines; ++i) {
      NewLineText += std::string(EscapedNewlineColumn - Offset - 1, ' ');
      NewLineText += "\\\n";
      Offset = 0;
    }
  }
  return NewLineText + getIndentText(Spaces);
}

std::string WhitespaceManager::getIndentText(unsigned Spaces) {
  if (!Style.UseTab)
    return std::string(Spaces, ' ');

  return std::string(Spaces / Style.IndentWidth, '\t') +
         std::string(Spaces % Style.IndentWidth, ' ');
}

void WhitespaceManager::alignComments() {
  unsigned MinColumn = 0;
  unsigned MaxColumn = UINT_MAX;
  token_iterator Start = Comments.begin();
  for (token_iterator I = Start, E = Comments.end(); I != E; ++I) {
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

void WhitespaceManager::alignComments(token_iterator I, token_iterator E,
                                      unsigned Column) {
  while (I != E) {
    if (!I->Untouchable) {
      unsigned Spaces = I->Spaces + Column - I->MinColumn;
      storeReplacement(I->ReplacementLoc, I->ReplacementLength,
                       getNewLineText(I->NewLines, Spaces));
    }
    ++I;
  }
}

void WhitespaceManager::alignEscapedNewlines() {
  unsigned MinColumn;
  if (Style.AlignEscapedNewlinesLeft) {
    MinColumn = 0;
    for (token_iterator I = EscapedNewlines.begin(), E = EscapedNewlines.end();
         I != E; ++I) {
      if (I->MinColumn > MinColumn)
        MinColumn = I->MinColumn;
    }
  } else {
    MinColumn = Style.ColumnLimit;
  }

  for (token_iterator I = EscapedNewlines.begin(), E = EscapedNewlines.end();
       I != E; ++I) {
    // I->MinColumn - 2 is the end of the previous token (i.e. the
    // WhitespaceStartColumn).
    storeReplacement(
        I->ReplacementLoc, I->ReplacementLength,
        I->Prefix + getNewLineText(I->NewLines, I->Spaces, I->MinColumn - 2,
                                   MinColumn) + I->Postfix);

  }
  EscapedNewlines.clear();
}

void WhitespaceManager::storeReplacement(SourceLocation Loc, unsigned Length,
                                         const std::string Text) {
  // Don't create a replacement, if it does not change anything.
  if (StringRef(SourceMgr.getCharacterData(Loc), Length) == Text)
    return;
  Replaces.insert(tooling::Replacement(SourceMgr, Loc, Length, Text));
}

} // namespace format
} // namespace clang
