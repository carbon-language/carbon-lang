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
#include "clang/Basic/CharInfo.h"
#include "clang/Format/Format.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

namespace clang {
namespace format {

static const char *const Blanks = " \t\v\f";
static bool IsBlank(char C) {
  switch (C) {
  case ' ':
  case '\t':
  case '\v':
  case '\f':
    return true;
  default:
    return false;
  }
}

static BreakableToken::Split getCommentSplit(StringRef Text,
                                             unsigned ContentStartColumn,
                                             unsigned ColumnLimit,
                                             encoding::Encoding Encoding) {
  if (ColumnLimit <= ContentStartColumn + 1)
    return BreakableToken::Split(StringRef::npos, 0);

  unsigned MaxSplit = ColumnLimit - ContentStartColumn + 1;
  unsigned MaxSplitBytes = 0;

  for (unsigned NumChars = 0;
       NumChars < MaxSplit && MaxSplitBytes < Text.size(); ++NumChars)
    MaxSplitBytes +=
        encoding::getCodePointNumBytes(Text[MaxSplitBytes], Encoding);

  StringRef::size_type SpaceOffset = Text.find_last_of(Blanks, MaxSplitBytes);
  if (SpaceOffset == StringRef::npos ||
      // Don't break at leading whitespace.
      Text.find_last_not_of(Blanks, SpaceOffset) == StringRef::npos) {
    // Make sure that we don't break at leading whitespace that
    // reaches past MaxSplit.
    StringRef::size_type FirstNonWhitespace = Text.find_first_not_of(Blanks);
    if (FirstNonWhitespace == StringRef::npos)
      // If the comment is only whitespace, we cannot split.
      return BreakableToken::Split(StringRef::npos, 0);
    SpaceOffset = Text.find_first_of(
        Blanks, std::max<unsigned>(MaxSplitBytes, FirstNonWhitespace));
  }
  if (SpaceOffset != StringRef::npos && SpaceOffset != 0) {
    StringRef BeforeCut = Text.substr(0, SpaceOffset).rtrim(Blanks);
    StringRef AfterCut = Text.substr(SpaceOffset).ltrim(Blanks);
    return BreakableToken::Split(BeforeCut.size(),
                                 AfterCut.begin() - BeforeCut.end());
  }
  return BreakableToken::Split(StringRef::npos, 0);
}

static BreakableToken::Split getStringSplit(StringRef Text,
                                            unsigned ContentStartColumn,
                                            unsigned ColumnLimit,
                                            encoding::Encoding Encoding) {
  // FIXME: Reduce unit test case.
  if (Text.empty())
    return BreakableToken::Split(StringRef::npos, 0);
  if (ColumnLimit <= ContentStartColumn)
    return BreakableToken::Split(StringRef::npos, 0);
  unsigned MaxSplit =
      std::min<unsigned>(ColumnLimit - ContentStartColumn,
                         encoding::getCodePointCount(Text, Encoding) - 1);
  StringRef::size_type SpaceOffset = 0;
  StringRef::size_type SlashOffset = 0;
  StringRef::size_type WordStartOffset = 0;
  StringRef::size_type SplitPoint = 0;
  for (unsigned Chars = 0;;) {
    unsigned Advance;
    if (Text[0] == '\\') {
      Advance = encoding::getEscapeSequenceLength(Text);
      Chars += Advance;
    } else {
      Advance = encoding::getCodePointNumBytes(Text[0], Encoding);
      Chars += 1;
    }

    if (Chars > MaxSplit)
      break;

    if (IsBlank(Text[0]))
      SpaceOffset = SplitPoint;
    if (Text[0] == '/')
      SlashOffset = SplitPoint;
    if (Advance == 1 && !isAlphanumeric(Text[0]))
      WordStartOffset = SplitPoint;

    SplitPoint += Advance;
    Text = Text.substr(Advance);
  }

  if (SpaceOffset != 0)
    return BreakableToken::Split(SpaceOffset + 1, 0);
  if (SlashOffset != 0)
    return BreakableToken::Split(SlashOffset + 1, 0);
  if (WordStartOffset != 0)
    return BreakableToken::Split(WordStartOffset + 1, 0);
  if (SplitPoint != 0)
    return BreakableToken::Split(SplitPoint, 0);
  return BreakableToken::Split(StringRef::npos, 0);
}

unsigned BreakableSingleLineToken::getLineCount() const { return 1; }

unsigned BreakableSingleLineToken::getLineLengthAfterSplit(
    unsigned LineIndex, unsigned Offset, StringRef::size_type Length) const {
  return StartColumn + Prefix.size() + Postfix.size() +
         encoding::getCodePointCount(Line.substr(Offset, Length), Encoding);
}

BreakableSingleLineToken::BreakableSingleLineToken(
    const FormatToken &Tok, unsigned StartColumn, StringRef Prefix,
    StringRef Postfix, bool InPPDirective, encoding::Encoding Encoding)
    : BreakableToken(Tok, InPPDirective, Encoding), StartColumn(StartColumn),
      Prefix(Prefix), Postfix(Postfix) {
  assert(Tok.TokenText.startswith(Prefix) && Tok.TokenText.endswith(Postfix));
  Line = Tok.TokenText.substr(
      Prefix.size(), Tok.TokenText.size() - Prefix.size() - Postfix.size());
}

BreakableStringLiteral::BreakableStringLiteral(const FormatToken &Tok,
                                               unsigned StartColumn,
                                               bool InPPDirective,
                                               encoding::Encoding Encoding)
    : BreakableSingleLineToken(Tok, StartColumn, "\"", "\"", InPPDirective,
                               Encoding) {}

BreakableToken::Split
BreakableStringLiteral::getSplit(unsigned LineIndex, unsigned TailOffset,
                                 unsigned ColumnLimit) const {
  return getStringSplit(Line.substr(TailOffset), StartColumn + 2, ColumnLimit,
                        Encoding);
}

void BreakableStringLiteral::insertBreak(unsigned LineIndex,
                                         unsigned TailOffset, Split Split,
                                         WhitespaceManager &Whitespaces) {
  Whitespaces.replaceWhitespaceInToken(
      Tok, Prefix.size() + TailOffset + Split.first, Split.second, Postfix,
      Prefix, InPPDirective, 1, StartColumn);
}

static StringRef getLineCommentPrefix(StringRef Comment) {
  const char *KnownPrefixes[] = { "/// ", "///", "// ", "//" };
  for (size_t i = 0, e = llvm::array_lengthof(KnownPrefixes); i != e; ++i)
    if (Comment.startswith(KnownPrefixes[i]))
      return KnownPrefixes[i];
  return "";
}

BreakableLineComment::BreakableLineComment(const FormatToken &Token,
                                           unsigned StartColumn,
                                           bool InPPDirective,
                                           encoding::Encoding Encoding)
    : BreakableSingleLineToken(Token, StartColumn,
                               getLineCommentPrefix(Token.TokenText), "",
                               InPPDirective, Encoding) {
  OriginalPrefix = Prefix;
  if (Token.TokenText.size() > Prefix.size() &&
      isAlphanumeric(Token.TokenText[Prefix.size()])) {
    if (Prefix == "//")
      Prefix = "// ";
    else if (Prefix == "///")
      Prefix = "/// ";
  }
}

BreakableToken::Split
BreakableLineComment::getSplit(unsigned LineIndex, unsigned TailOffset,
                               unsigned ColumnLimit) const {
  return getCommentSplit(Line.substr(TailOffset), StartColumn + Prefix.size(),
                         ColumnLimit, Encoding);
}

void BreakableLineComment::insertBreak(unsigned LineIndex, unsigned TailOffset,
                                       Split Split,
                                       WhitespaceManager &Whitespaces) {
  Whitespaces.replaceWhitespaceInToken(
      Tok, OriginalPrefix.size() + TailOffset + Split.first, Split.second,
      Postfix, Prefix, InPPDirective, 1, StartColumn);
}

void
BreakableLineComment::replaceWhitespaceBefore(unsigned LineIndex,
                                              WhitespaceManager &Whitespaces) {
  if (OriginalPrefix != Prefix) {
    Whitespaces.replaceWhitespaceInToken(Tok, OriginalPrefix.size(), 0, "", "",
                                         false, 0, 1);
  }
}

BreakableBlockComment::BreakableBlockComment(
    const FormatStyle &Style, const FormatToken &Token, unsigned StartColumn,
    unsigned OriginalStartColumn, bool FirstInLine, bool InPPDirective,
    encoding::Encoding Encoding)
    : BreakableToken(Token, InPPDirective, Encoding) {
  StringRef TokenText(Token.TokenText);
  assert(TokenText.startswith("/*") && TokenText.endswith("*/"));
  TokenText.substr(2, TokenText.size() - 4).split(Lines, "\n");

  int IndentDelta = StartColumn - OriginalStartColumn;
  LeadingWhitespace.resize(Lines.size());
  StartOfLineColumn.resize(Lines.size());
  StartOfLineColumn[0] = StartColumn + 2;
  for (size_t i = 1; i < Lines.size(); ++i)
    adjustWhitespace(Style, i, IndentDelta);

  Decoration = "* ";
  if (Lines.size() == 1 && !FirstInLine) {
    // Comments for which FirstInLine is false can start on arbitrary column,
    // and available horizontal space can be too small to align consecutive
    // lines with the first one.
    // FIXME: We could, probably, align them to current indentation level, but
    // now we just wrap them without stars.
    Decoration = "";
  }
  for (size_t i = 1, e = Lines.size(); i < e && !Decoration.empty(); ++i) {
    // If the last line is empty, the closing "*/" will have a star.
    if (i + 1 == e && Lines[i].empty())
      break;
    while (!Lines[i].startswith(Decoration))
      Decoration = Decoration.substr(0, Decoration.size() - 1);
  }

  LastLineNeedsDecoration = true;
  IndentAtLineBreak = StartOfLineColumn[0] + 1;
  for (size_t i = 1; i < Lines.size(); ++i) {
    if (Lines[i].empty()) {
      if (i + 1 == Lines.size()) {
        // Empty last line means that we already have a star as a part of the
        // trailing */. We also need to preserve whitespace, so that */ is
        // correctly indented.
        LastLineNeedsDecoration = false;
      } else if (Decoration.empty()) {
        // For all other lines, set the start column to 0 if they're empty, so
        // we do not insert trailing whitespace anywhere.
        StartOfLineColumn[i] = 0;
      }
      continue;
    }
    // The first line already excludes the star.
    // For all other lines, adjust the line to exclude the star and
    // (optionally) the first whitespace.
    StartOfLineColumn[i] += Decoration.size();
    Lines[i] = Lines[i].substr(Decoration.size());
    LeadingWhitespace[i] += Decoration.size();
    IndentAtLineBreak = std::min<int>(IndentAtLineBreak, StartOfLineColumn[i]);
  }
  IndentAtLineBreak = std::max<unsigned>(IndentAtLineBreak, Decoration.size());
  DEBUG({
    llvm::dbgs() << "IndentAtLineBreak " << IndentAtLineBreak << "\n";
    for (size_t i = 0; i < Lines.size(); ++i) {
      llvm::dbgs() << i << " |" << Lines[i] << "| " << LeadingWhitespace[i]
                   << "\n";
    }
  });
}

void BreakableBlockComment::adjustWhitespace(const FormatStyle &Style,
                                             unsigned LineIndex,
                                             int IndentDelta) {
  // When in a preprocessor directive, the trailing backslash in a block comment
  // is not needed, but can serve a purpose of uniformity with necessary escaped
  // newlines outside the comment. In this case we remove it here before
  // trimming the trailing whitespace. The backslash will be re-added later when
  // inserting a line break.
  size_t EndOfPreviousLine = Lines[LineIndex - 1].size();
  if (InPPDirective && Lines[LineIndex - 1].endswith("\\"))
    --EndOfPreviousLine;

  // Calculate the end of the non-whitespace text in the previous line.
  EndOfPreviousLine =
      Lines[LineIndex - 1].find_last_not_of(Blanks, EndOfPreviousLine);
  if (EndOfPreviousLine == StringRef::npos)
    EndOfPreviousLine = 0;
  else
    ++EndOfPreviousLine;
  // Calculate the start of the non-whitespace text in the current line.
  size_t StartOfLine = Lines[LineIndex].find_first_not_of(Blanks);
  if (StartOfLine == StringRef::npos)
    StartOfLine = Lines[LineIndex].size();

  // Adjust Lines to only contain relevant text.
  Lines[LineIndex - 1] = Lines[LineIndex - 1].substr(0, EndOfPreviousLine);
  Lines[LineIndex] = Lines[LineIndex].substr(StartOfLine);
  // Adjust LeadingWhitespace to account all whitespace between the lines
  // to the current line.
  LeadingWhitespace[LineIndex] =
      Lines[LineIndex].begin() - Lines[LineIndex - 1].end();

  // FIXME: We currently count tabs as 1 character. To solve this, we need to
  // get the correct indentation width of the start of the comment, which
  // requires correct counting of the tab expansions before the comment, and
  // a configurable tab width. Since the current implementation only breaks
  // if leading tabs are intermixed with spaces, that is not a high priority.

  // Adjust the start column uniformly accross all lines.
  StartOfLineColumn[LineIndex] = std::max<int>(0, StartOfLine + IndentDelta);
}

unsigned BreakableBlockComment::getLineCount() const { return Lines.size(); }

unsigned BreakableBlockComment::getLineLengthAfterSplit(
    unsigned LineIndex, unsigned Offset, StringRef::size_type Length) const {
  return getContentStartColumn(LineIndex, Offset) +
         encoding::getCodePointCount(Lines[LineIndex].substr(Offset, Length),
                                     Encoding) +
         // The last line gets a "*/" postfix.
         (LineIndex + 1 == Lines.size() ? 2 : 0);
}

BreakableToken::Split
BreakableBlockComment::getSplit(unsigned LineIndex, unsigned TailOffset,
                                unsigned ColumnLimit) const {
  return getCommentSplit(Lines[LineIndex].substr(TailOffset),
                         getContentStartColumn(LineIndex, TailOffset),
                         ColumnLimit, Encoding);
}

void BreakableBlockComment::insertBreak(unsigned LineIndex, unsigned TailOffset,
                                        Split Split,
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
  assert(IndentAtLineBreak >= Decoration.size());
  Whitespaces.replaceWhitespaceInToken(Tok, BreakOffsetInToken, CharsToRemove,
                                       "", Prefix, InPPDirective, 1,
                                       IndentAtLineBreak - Decoration.size());
}

void
BreakableBlockComment::replaceWhitespaceBefore(unsigned LineIndex,
                                               WhitespaceManager &Whitespaces) {
  if (LineIndex == 0)
    return;
  StringRef Prefix = Decoration;
  if (Lines[LineIndex].empty()) {
    if (LineIndex + 1 == Lines.size()) {
      if (!LastLineNeedsDecoration) {
        // If the last line was empty, we don't need a prefix, as the */ will
        // line up with the decoration (if it exists).
        Prefix = "";
      }
    } else if (!Decoration.empty()) {
      // For other empty lines, if we do have a decoration, adapt it to not
      // contain a trailing whitespace.
      Prefix = Prefix.substr(0, 1);
    }
  } else {
    if (StartOfLineColumn[LineIndex] == 1) {
      // This line starts immediately after the decorating *.
      Prefix = Prefix.substr(0, 1);
    }
  }

  unsigned WhitespaceOffsetInToken = Lines[LineIndex].data() -
                                     Tok.TokenText.data() -
                                     LeadingWhitespace[LineIndex];
  assert(StartOfLineColumn[LineIndex] >= Prefix.size());
  Whitespaces.replaceWhitespaceInToken(
      Tok, WhitespaceOffsetInToken, LeadingWhitespace[LineIndex], "", Prefix,
      InPPDirective, 1, StartOfLineColumn[LineIndex] - Prefix.size());
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
