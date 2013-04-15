//===--- BreakableToken.cpp - Format C++ code -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Contains implementation of BreakableToken class and classes derived
/// from it.
///
//===----------------------------------------------------------------------===//

#include "BreakableToken.h"
#include <algorithm>

namespace clang {
namespace format {

BreakableBlockComment::BreakableBlockComment(const SourceManager &SourceMgr,
                                             const AnnotatedToken &Token,
                                             unsigned StartColumn)
    : Tok(Token.FormatTok), StartColumn(StartColumn) {

  SourceLocation TokenLoc = Tok.Tok.getLocation();
  TokenText = StringRef(SourceMgr.getCharacterData(TokenLoc), Tok.TokenLength);
  assert(TokenText.startswith("/*") && TokenText.endswith("*/"));

  OriginalStartColumn = SourceMgr.getSpellingColumnNumber(TokenLoc) - 1;

  TokenText.substr(2, TokenText.size() - 4).split(Lines, "\n");

  NeedsStar = true;
  CommonPrefixLength = UINT_MAX;
  if (Lines.size() == 1) {
    if (Token.Parent == 0) {
      // Standalone block comments will be aligned and prefixed with *s.
      CommonPrefixLength = OriginalStartColumn + 1;
    } else {
      // Trailing comments can start on arbitrary column, and available
      // horizontal space can be too small to align consecutive lines with
      // the first one. We could, probably, align them to current
      // indentation level, but now we just wrap them without indentation
      // and stars.
      CommonPrefixLength = 0;
      NeedsStar = false;
    }
  } else {
    for (size_t i = 1; i < Lines.size(); ++i) {
      size_t FirstNonWhitespace = Lines[i].find_first_not_of(" ");
      if (FirstNonWhitespace != StringRef::npos) {
        NeedsStar = NeedsStar && (Lines[i][FirstNonWhitespace] == '*');
        CommonPrefixLength =
            std::min<unsigned>(CommonPrefixLength, FirstNonWhitespace);
      }
    }
  }
  if (CommonPrefixLength == UINT_MAX)
    CommonPrefixLength = 0;

  IndentAtLineBreak =
      std::max<int>(StartColumn - OriginalStartColumn + CommonPrefixLength, 0);
}

void BreakableBlockComment::alignLines(WhitespaceManager &Whitespaces) {
  SourceLocation TokenLoc = Tok.Tok.getLocation();
  int IndentDelta = StartColumn - OriginalStartColumn;
  if (IndentDelta > 0) {
    std::string WhiteSpace(IndentDelta, ' ');
    for (size_t i = 1; i < Lines.size(); ++i) {
      Whitespaces.addReplacement(
          TokenLoc.getLocWithOffset(Lines[i].data() - TokenText.data()), 0,
          WhiteSpace);
    }
  } else if (IndentDelta < 0) {
    std::string WhiteSpace(-IndentDelta, ' ');
    // Check that the line is indented enough.
    for (size_t i = 1; i < Lines.size(); ++i) {
      if (!Lines[i].startswith(WhiteSpace))
        return;
    }
    for (size_t i = 1; i < Lines.size(); ++i) {
      Whitespaces.addReplacement(
          TokenLoc.getLocWithOffset(Lines[i].data() - TokenText.data()),
          -IndentDelta, "");
    }
  }

  for (unsigned i = 1; i < Lines.size(); ++i)
    Lines[i] = Lines[i].substr(CommonPrefixLength + (NeedsStar ? 2 : 0));
}

BreakableToken::Split BreakableBlockComment::getSplit(unsigned LineIndex,
                                                      unsigned TailOffset,
                                                      unsigned ColumnLimit) {
  StringRef Text = getLine(LineIndex).substr(TailOffset);
  unsigned DecorationLength =
      (TailOffset == 0 && LineIndex == 0) ? StartColumn + 2 : getPrefixLength();
  if (ColumnLimit <= DecorationLength + 1)
    return Split(StringRef::npos, 0);

  unsigned MaxSplit = ColumnLimit - DecorationLength + 1;
  StringRef::size_type SpaceOffset = Text.rfind(' ', MaxSplit);
  if (SpaceOffset == StringRef::npos ||
      Text.find_last_not_of(' ', SpaceOffset) == StringRef::npos) {
    SpaceOffset = Text.find(' ', MaxSplit);
  }
  if (SpaceOffset != StringRef::npos && SpaceOffset != 0) {
    StringRef BeforeCut = Text.substr(0, SpaceOffset).rtrim();
    StringRef AfterCut = Text.substr(SpaceOffset).ltrim();
    return BreakableToken::Split(BeforeCut.size(),
                                 AfterCut.begin() - BeforeCut.end());
  }
  return BreakableToken::Split(StringRef::npos, 0);
}

void BreakableBlockComment::insertBreak(unsigned LineIndex, unsigned TailOffset,
                                        Split Split, bool InPPDirective,
                                        WhitespaceManager &Whitespaces) {
  StringRef Text = getLine(LineIndex).substr(TailOffset);
  StringRef AdditionalPrefix = NeedsStar ? "* " : "";
  if (Text.size() == Split.first + Split.second) {
    // For all but the last line handle trailing space separately.
    if (LineIndex < Lines.size() - 1)
      return;
    // For the last line we need to break before "*/", but not to add "* ".
    AdditionalPrefix = "";
  }

  unsigned WhitespaceStartColumn =
      Split.first +
      (LineIndex == 0 && TailOffset == 0 ? StartColumn + 2 : getPrefixLength());
  unsigned BreakOffset = Text.data() - TokenText.data() + Split.first;
  unsigned CharsToRemove = Split.second;
  Whitespaces.breakToken(Tok, BreakOffset, CharsToRemove, "", AdditionalPrefix,
                         InPPDirective, IndentAtLineBreak,
                         WhitespaceStartColumn);
}

void BreakableBlockComment::trimLine(unsigned LineIndex, unsigned TailOffset,
                                     unsigned InPPDirective,
                                     WhitespaceManager &Whitespaces) {
  if (LineIndex == Lines.size() - 1)
    return;
  StringRef Text = Lines[LineIndex].substr(TailOffset);
  if (!Text.endswith(" ") && !InPPDirective)
    return;

  StringRef TrimmedLine = Text.rtrim();
  unsigned WhitespaceStartColumn =
      getLineLengthAfterSplit(LineIndex, TailOffset);
  unsigned BreakOffset = TrimmedLine.end() - TokenText.data();
  unsigned CharsToRemove = Text.size() - TrimmedLine.size() + 1;
  Whitespaces.breakToken(Tok, BreakOffset, CharsToRemove, "", "", InPPDirective,
                         0, WhitespaceStartColumn);
}

} // namespace format
} // namespace clang
