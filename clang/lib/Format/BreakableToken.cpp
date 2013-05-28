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

#define DEBUG_TYPE "format-token-breaker"

#include "BreakableToken.h"
#include "clang/Format/Format.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

namespace clang {
namespace format {
namespace {

// FIXME: Move helper string functions to where it makes sense.

unsigned getOctalLength(StringRef Text) {
  unsigned I = 1;
  while (I < Text.size() && I < 4 && (Text[I] >= '0' && Text[I] <= '7')) {
    ++I;
  }
  return I;
}

unsigned getHexLength(StringRef Text) {
  unsigned I = 2; // Point after '\x'.
  while (I < Text.size() && ((Text[I] >= '0' && Text[I] <= '9') ||
                             (Text[I] >= 'a' && Text[I] <= 'f') ||
                             (Text[I] >= 'A' && Text[I] <= 'F'))) {
    ++I;
  }
  return I;
}

unsigned getEscapeSequenceLength(StringRef Text) {
  assert(Text[0] == '\\');
  if (Text.size() < 2)
    return 1;

  switch (Text[1]) {
  case 'u':
    return 6;
  case 'U':
    return 10;
  case 'x':
    return getHexLength(Text);
  default:
    if (Text[1] >= '0' && Text[1] <= '7')
      return getOctalLength(Text);
    return 2;
  }
}

StringRef::size_type getStartOfCharacter(StringRef Text,
                                         StringRef::size_type Offset) {
  StringRef::size_type NextEscape = Text.find('\\');
  while (NextEscape != StringRef::npos && NextEscape < Offset) {
    StringRef::size_type SequenceLength =
        getEscapeSequenceLength(Text.substr(NextEscape));
    if (Offset < NextEscape + SequenceLength)
      return NextEscape;
    NextEscape = Text.find('\\', NextEscape + SequenceLength);
  }
  return Offset;
}

BreakableToken::Split getCommentSplit(StringRef Text,
                                      unsigned ContentStartColumn,
                                      unsigned ColumnLimit) {
  if (ColumnLimit <= ContentStartColumn + 1)
    return BreakableToken::Split(StringRef::npos, 0);

  unsigned MaxSplit = ColumnLimit - ContentStartColumn + 1;
  StringRef::size_type SpaceOffset = Text.rfind(' ', MaxSplit);
  if (SpaceOffset == StringRef::npos ||
      // Don't break at leading whitespace.
      Text.find_last_not_of(' ', SpaceOffset) == StringRef::npos)
    SpaceOffset = Text.find(' ', MaxSplit);
  if (SpaceOffset != StringRef::npos && SpaceOffset != 0) {
    StringRef BeforeCut = Text.substr(0, SpaceOffset).rtrim();
    StringRef AfterCut = Text.substr(SpaceOffset).ltrim();
    return BreakableToken::Split(BeforeCut.size(),
                                 AfterCut.begin() - BeforeCut.end());
  }
  return BreakableToken::Split(StringRef::npos, 0);
}

BreakableToken::Split getStringSplit(StringRef Text,
                                     unsigned ContentStartColumn,
                                     unsigned ColumnLimit) {

  if (ColumnLimit <= ContentStartColumn)
    return BreakableToken::Split(StringRef::npos, 0);
  unsigned MaxSplit = ColumnLimit - ContentStartColumn;
  // FIXME: Reduce unit test case.
  if (Text.empty())
    return BreakableToken::Split(StringRef::npos, 0);
  MaxSplit = std::min<unsigned>(MaxSplit, Text.size() - 1);
  StringRef::size_type SpaceOffset = Text.rfind(' ', MaxSplit);
  if (SpaceOffset != StringRef::npos && SpaceOffset != 0)
    return BreakableToken::Split(SpaceOffset + 1, 0);
  StringRef::size_type SlashOffset = Text.rfind('/', MaxSplit);
  if (SlashOffset != StringRef::npos && SlashOffset != 0)
    return BreakableToken::Split(SlashOffset + 1, 0);
  StringRef::size_type SplitPoint = getStartOfCharacter(Text, MaxSplit);
  if (SplitPoint == StringRef::npos || SplitPoint == 0)
    return BreakableToken::Split(StringRef::npos, 0);
  return BreakableToken::Split(SplitPoint, 0);
}

} // namespace

unsigned BreakableSingleLineToken::getLineCount() const { return 1; }

unsigned
BreakableSingleLineToken::getLineLengthAfterSplit(unsigned LineIndex,
                                                  unsigned TailOffset) const {
  return StartColumn + Prefix.size() + Postfix.size() + Line.size() -
         TailOffset;
}

void BreakableSingleLineToken::insertBreak(unsigned LineIndex,
                                           unsigned TailOffset, Split Split,
                                           bool InPPDirective,
                                           WhitespaceManager &Whitespaces) {
  Whitespaces.breakToken(Tok, Prefix.size() + TailOffset + Split.first,
                         Split.second, Postfix, Prefix, InPPDirective,
                         StartColumn);
}

BreakableSingleLineToken::BreakableSingleLineToken(const FormatToken &Tok,
                                                   unsigned StartColumn,
                                                   StringRef Prefix,
                                                   StringRef Postfix)
    : BreakableToken(Tok), StartColumn(StartColumn), Prefix(Prefix),
      Postfix(Postfix) {
  assert(Tok.TokenText.startswith(Prefix) && Tok.TokenText.endswith(Postfix));
  Line = Tok.TokenText.substr(
      Prefix.size(), Tok.TokenText.size() - Prefix.size() - Postfix.size());
}

BreakableStringLiteral::BreakableStringLiteral(const FormatToken &Tok,
                                               unsigned StartColumn)
    : BreakableSingleLineToken(Tok, StartColumn, "\"", "\"") {}

BreakableToken::Split
BreakableStringLiteral::getSplit(unsigned LineIndex, unsigned TailOffset,
                                 unsigned ColumnLimit) const {
  return getStringSplit(Line.substr(TailOffset), StartColumn + 2, ColumnLimit);
}

static StringRef getLineCommentPrefix(StringRef Comment) {
  const char *KnownPrefixes[] = { "/// ", "///", "// ", "//" };
  for (size_t i = 0, e = llvm::array_lengthof(KnownPrefixes); i != e; ++i)
    if (Comment.startswith(KnownPrefixes[i]))
      return KnownPrefixes[i];
  return "";
}

BreakableLineComment::BreakableLineComment(const FormatToken &Token,
                                           unsigned StartColumn)
    : BreakableSingleLineToken(Token, StartColumn,
                               getLineCommentPrefix(Token.TokenText), "") {}

BreakableToken::Split
BreakableLineComment::getSplit(unsigned LineIndex, unsigned TailOffset,
                               unsigned ColumnLimit) const {
  return getCommentSplit(Line.substr(TailOffset), StartColumn + Prefix.size(),
                         ColumnLimit);
}

BreakableBlockComment::BreakableBlockComment(const FormatStyle &Style,
                                             const FormatToken &Token,
                                             unsigned StartColumn,
                                             unsigned OriginalStartColumn,
                                             bool FirstInLine)
    : BreakableToken(Token) {
  StringRef TokenText(Token.TokenText);
  assert(TokenText.startswith("/*") && TokenText.endswith("*/"));
  TokenText.substr(2, TokenText.size() - 4).split(Lines, "\n");

  int IndentDelta = StartColumn - OriginalStartColumn;
  bool NeedsStar = true;
  LeadingWhitespace.resize(Lines.size());
  StartOfLineColumn.resize(Lines.size());
  if (Lines.size() == 1 && !FirstInLine) {
    // Comments for which FirstInLine is false can start on arbitrary column,
    // and available horizontal space can be too small to align consecutive
    // lines with the first one.
    // FIXME: We could, probably, align them to current indentation level, but
    // now we just wrap them without stars.
    NeedsStar = false;
  }
  StartOfLineColumn[0] = StartColumn + 2;
  for (size_t i = 1; i < Lines.size(); ++i) {
    adjustWhitespace(Style, i, IndentDelta);
    if (Lines[i].empty())
      // If the last line is empty, the closing "*/" will have a star.
      NeedsStar = NeedsStar && i + 1 == Lines.size();
    else
      NeedsStar = NeedsStar && Lines[i][0] == '*';
  }
  Decoration = NeedsStar ? "* " : "";
  IndentAtLineBreak = StartOfLineColumn[0] + 1;
  for (size_t i = 1; i < Lines.size(); ++i) {
    if (Lines[i].empty()) {
      if (!NeedsStar && i + 1 != Lines.size())
        // For all but the last line (which always ends in */), set the
        // start column to 0 if they're empty, so we do not insert
        // trailing whitespace anywhere.
        StartOfLineColumn[i] = 0;
      continue;
    }
    if (NeedsStar) {
      // The first line already excludes the star.
      // For all other lines, adjust the line to exclude the star and
      // (optionally) the first whitespace.
      int Offset = Lines[i].startswith("* ") ? 2 : 1;
      StartOfLineColumn[i] += Offset;
      Lines[i] = Lines[i].substr(Offset);
      LeadingWhitespace[i] += Offset;
    }
    IndentAtLineBreak = std::min<int>(IndentAtLineBreak, StartOfLineColumn[i]);
  }
  DEBUG({
    for (size_t i = 0; i < Lines.size(); ++i) {
      llvm::dbgs() << i << " |" << Lines[i] << "| " << LeadingWhitespace[i]
                   << "\n";
    }
  });
}

void BreakableBlockComment::adjustWhitespace(const FormatStyle &Style,
                                             unsigned LineIndex,
                                             int IndentDelta) {
  // Calculate the end of the non-whitespace text in the previous line.
  size_t EndOfPreviousLine = Lines[LineIndex - 1].find_last_not_of(" \\\t");
  if (EndOfPreviousLine == StringRef::npos)
    EndOfPreviousLine = 0;
  else
    ++EndOfPreviousLine;
  // Calculate the start of the non-whitespace text in the current line.
  size_t StartOfLine = Lines[LineIndex].find_first_not_of(" \t");
  if (StartOfLine == StringRef::npos)
    StartOfLine = Lines[LineIndex].size();
  // FIXME: Tabs are not always 8 characters. Make configurable in the style.
  unsigned Column = 0;
  StringRef OriginalIndentText = Lines[LineIndex].substr(0, StartOfLine);
  for (int i = 0, e = OriginalIndentText.size(); i != e; ++i) {
    if (Lines[LineIndex][i] == '\t')
      Column += 8 - (Column % 8);
    else
      ++Column;
  }

  // Adjust Lines to only contain relevant text.
  Lines[LineIndex - 1] = Lines[LineIndex - 1].substr(0, EndOfPreviousLine);
  Lines[LineIndex] = Lines[LineIndex].substr(StartOfLine);
  // Adjust LeadingWhitespace to account all whitespace between the lines
  // to the current line.
  LeadingWhitespace[LineIndex] =
      Lines[LineIndex].begin() - Lines[LineIndex - 1].end();
  // Adjust the start column uniformly accross all lines.
  StartOfLineColumn[LineIndex] = std::max<int>(0, Column + IndentDelta);
}

unsigned BreakableBlockComment::getLineCount() const { return Lines.size(); }

unsigned
BreakableBlockComment::getLineLengthAfterSplit(unsigned LineIndex,
                                               unsigned TailOffset) const {
  return getContentStartColumn(LineIndex, TailOffset) +
         (Lines[LineIndex].size() - TailOffset) +
         // The last line gets a "*/" postfix.
         (LineIndex + 1 == Lines.size() ? 2 : 0);
}

BreakableToken::Split
BreakableBlockComment::getSplit(unsigned LineIndex, unsigned TailOffset,
                                unsigned ColumnLimit) const {
  return getCommentSplit(Lines[LineIndex].substr(TailOffset),
                         getContentStartColumn(LineIndex, TailOffset),
                         ColumnLimit);
}

void BreakableBlockComment::insertBreak(unsigned LineIndex, unsigned TailOffset,
                                        Split Split, bool InPPDirective,
                                        WhitespaceManager &Whitespaces) {
  StringRef Text = Lines[LineIndex].substr(TailOffset);
  StringRef Prefix = Decoration;
  if (LineIndex + 1 == Lines.size() &&
      Text.size() == Split.first + Split.second) {
    // For the last line we need to break before "*/", but not to add "* ".
    Prefix = "";
  }

  unsigned BreakOffsetInToken =
      Text.data() - Tok.TokenText.data() + Split.first;
  unsigned CharsToRemove = Split.second;
  Whitespaces.breakToken(Tok, BreakOffsetInToken, CharsToRemove, "", Prefix,
                         InPPDirective, IndentAtLineBreak - Decoration.size());
}

void
BreakableBlockComment::replaceWhitespaceBefore(unsigned LineIndex,
                                               unsigned InPPDirective,
                                               WhitespaceManager &Whitespaces) {
  if (LineIndex == 0)
    return;
  StringRef Prefix = Decoration;
  if (Lines[LineIndex].empty()) {
    if (LineIndex + 1 == Lines.size()) {
      // If the last line is empty, we don't need a prefix, as the */ will line
      // up with the decoration (if it exists).
      Prefix = "";
    } else if (!Decoration.empty()) {
      // For other empty lines, if we do have a decoration, adapt it to not
      // contain a trailing whitespace.
      Prefix = Prefix.substr(0, 1);
    }
  }

  unsigned WhitespaceOffsetInToken =
      Lines[LineIndex].data() - Tok.TokenText.data() -
      LeadingWhitespace[LineIndex];
  Whitespaces.breakToken(
      Tok, WhitespaceOffsetInToken, LeadingWhitespace[LineIndex], "", Prefix,
      InPPDirective, StartOfLineColumn[LineIndex] - Prefix.size());
}

unsigned
BreakableBlockComment::getContentStartColumn(unsigned LineIndex,
                                             unsigned TailOffset) const {
  // If we break, we always break at the predefined indent.
  if (TailOffset != 0)
    return IndentAtLineBreak;
  return StartOfLineColumn[LineIndex];
}

} // namespace format
} // namespace clang
