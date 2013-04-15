//===--- BreakableToken.h - Format C++ code -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Declares BreakableToken, BreakableStringLiteral, and
/// BreakableBlockComment classes, that contain token type-specific logic to
/// break long lines in tokens.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FORMAT_BREAKABLETOKEN_H
#define LLVM_CLANG_FORMAT_BREAKABLETOKEN_H

#include "TokenAnnotator.h"
#include "WhitespaceManager.h"
#include <utility>

namespace clang {
namespace format {

class BreakableToken {
public:
  virtual ~BreakableToken() {}
  virtual unsigned getLineCount() const = 0;
  virtual unsigned getLineSize(unsigned Index) = 0;
  virtual unsigned getLineLengthAfterSplit(unsigned LineIndex,
                                           unsigned TailOffset) = 0;
  virtual unsigned getPrefixLength() = 0;
  virtual unsigned getSuffixLength(unsigned LineIndex) = 0;

  // Contains starting character index and length of split.
  typedef std::pair<StringRef::size_type, unsigned> Split;
  virtual Split getSplit(unsigned LineIndex, unsigned TailOffset,
                         unsigned ColumnLimit) = 0;
  virtual void insertBreak(unsigned LineIndex, unsigned TailOffset, Split Split,
                           bool InPPDirective,
                           WhitespaceManager &Whitespaces) = 0;
  virtual void trimLine(unsigned LineIndex, unsigned TailOffset,
                        unsigned InPPDirective,
                        WhitespaceManager &Whitespaces) = 0;
};

class BreakableStringLiteral : public BreakableToken {
public:
  BreakableStringLiteral(const FormatToken &Tok, unsigned StartColumn)
      : Tok(Tok), StartColumn(StartColumn) {}

  virtual unsigned getLineCount() const { return 1; }

  virtual unsigned getLineSize(unsigned Index) {
    return Tok.TokenLength - 2; // Should be in sync with getLine
  }

  virtual unsigned getLineLengthAfterSplit(unsigned LineIndex,
                                           unsigned TailOffset) {
    return getPrefixLength() + getLine(LineIndex).size() - TailOffset +
           getSuffixLength(LineIndex);
  }

  virtual unsigned getPrefixLength() { return StartColumn + 1; }

  virtual unsigned getSuffixLength(unsigned LineIndex) { return 1; }

  virtual Split getSplit(unsigned LineIndex, unsigned TailOffset,
                         unsigned ColumnLimit) {
    StringRef Text = getLine(LineIndex).substr(TailOffset);
    unsigned DecorationLength = getPrefixLength() + getSuffixLength(0);
    if (ColumnLimit <= DecorationLength)
      return Split(StringRef::npos, 0);
    unsigned MaxSplit = ColumnLimit - DecorationLength;
    assert(MaxSplit < Text.size());
    StringRef::size_type SpaceOffset = Text.rfind(' ', MaxSplit);
    if (SpaceOffset != StringRef::npos && SpaceOffset != 0)
      return Split(SpaceOffset + 1, 0);
    StringRef::size_type SlashOffset = Text.rfind('/', MaxSplit);
    if (SlashOffset != StringRef::npos && SlashOffset != 0)
      return Split(SlashOffset + 1, 0);
    StringRef::size_type SplitPoint = getStartOfCharacter(Text, MaxSplit);
    if (SplitPoint != StringRef::npos && SplitPoint > 1)
      // Do not split at 0.
      return Split(SplitPoint, 0);
    return Split(StringRef::npos, 0);
  }

  virtual void insertBreak(unsigned LineIndex, unsigned TailOffset, Split Split,
                           bool InPPDirective, WhitespaceManager &Whitespaces) {
    unsigned WhitespaceStartColumn = StartColumn + Split.first + 2;
    Whitespaces.breakToken(Tok, TailOffset + Split.first + 1, Split.second,
                           "\"", "\"", InPPDirective, StartColumn,
                           WhitespaceStartColumn);
  }

  virtual void trimLine(unsigned LineIndex, unsigned TailOffset,
                        unsigned InPPDirective,
                        WhitespaceManager &Whitespaces) {}

private:
  StringRef getLine(unsigned Index) {
    // Get string without quotes.
    // FIXME: Handle string prefixes.
    return StringRef(Tok.Tok.getLiteralData() + 1, Tok.TokenLength - 2);
  }

  static StringRef::size_type getStartOfCharacter(StringRef Text,
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

  static unsigned getEscapeSequenceLength(StringRef Text) {
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

  static unsigned getHexLength(StringRef Text) {
    unsigned I = 2; // Point after '\x'.
    while (I < Text.size() && ((Text[I] >= '0' && Text[I] <= '9') ||
                               (Text[I] >= 'a' && Text[I] <= 'f') ||
                               (Text[I] >= 'A' && Text[I] <= 'F'))) {
      ++I;
    }
    return I;
  }

  static unsigned getOctalLength(StringRef Text) {
    unsigned I = 1;
    while (I < Text.size() && I < 4 && (Text[I] >= '0' && Text[I] <= '7')) {
      ++I;
    }
    return I;
  }

  const FormatToken &Tok;
  unsigned StartColumn;
};

class BreakableBlockComment : public BreakableToken {
public:
  BreakableBlockComment(const SourceManager &SourceMgr,
                        const AnnotatedToken &Token, unsigned StartColumn);

  void alignLines(WhitespaceManager &Whitespaces);

  virtual unsigned getLineCount() const { return Lines.size(); }

  virtual unsigned getLineSize(unsigned Index) {
    return getLine(Index).size();
  }

  virtual unsigned getLineLengthAfterSplit(unsigned LineIndex,
                                           unsigned TailOffset) {
    unsigned ContentStartColumn = getPrefixLength();
    if (TailOffset == 0 && LineIndex == 0)
      ContentStartColumn = StartColumn + 2;
    return ContentStartColumn + getLine(LineIndex).size() - TailOffset +
           getSuffixLength(LineIndex);
  }

  virtual unsigned getPrefixLength() {
    return IndentAtLineBreak + (NeedsStar ? 2 : 0);
  }

  virtual unsigned getSuffixLength(unsigned LineIndex) {
    if (LineIndex + 1 < Lines.size())
      return 0;
    return 2;
  }

  virtual Split getSplit(unsigned LineIndex, unsigned TailOffset,
                         unsigned ColumnLimit);

  virtual void insertBreak(unsigned LineIndex, unsigned TailOffset, Split Split,
                           bool InPPDirective, WhitespaceManager &Whitespaces);

  virtual void trimLine(unsigned LineIndex, unsigned TailOffset,
                        unsigned InPPDirective, WhitespaceManager &Whitespaces);

private:
  // Get comment lines without /* */, common prefix and trailing whitespace.
  // Last line is not trimmed, as it is terminated by */, so its trailing
  // whitespace is not really trailing.
  StringRef getLine(unsigned Index) {
    return Index < Lines.size() - 1 ? Lines[Index].rtrim() : Lines[Index];
  }

  const FormatToken &Tok;
  const unsigned StartColumn;
  StringRef TokenText;
  unsigned OriginalStartColumn;
  unsigned CommonPrefixLength;
  unsigned IndentAtLineBreak;
  bool NeedsStar;
  SmallVector<StringRef, 16> Lines;
};

} // namespace format
} // namespace clang

#endif // LLVM_CLANG_FORMAT_BREAKABLETOKEN_H
