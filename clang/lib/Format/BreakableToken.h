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
  BreakableToken(const SourceManager &SourceMgr, const FormatToken &Tok,
                 unsigned StartColumn)
      : Tok(Tok), StartColumn(StartColumn),
        TokenText(SourceMgr.getCharacterData(Tok.getStartOfNonWhitespace()),
                  Tok.TokenLength) {}
  virtual ~BreakableToken() {}
  virtual unsigned getLineCount() const = 0;
  virtual unsigned getLineSize(unsigned Index) const = 0;
  virtual unsigned getLineLengthAfterSplit(unsigned LineIndex,
                                           unsigned TailOffset) const = 0;

  // Contains starting character index and length of split.
  typedef std::pair<StringRef::size_type, unsigned> Split;
  virtual Split getSplit(unsigned LineIndex, unsigned TailOffset,
                         unsigned ColumnLimit) const = 0;
  virtual void insertBreak(unsigned LineIndex, unsigned TailOffset, Split Split,
                           bool InPPDirective,
                           WhitespaceManager &Whitespaces) = 0;
  virtual void trimLine(unsigned LineIndex, unsigned TailOffset,
                        unsigned InPPDirective,
                        WhitespaceManager &Whitespaces) {}
protected:
  const FormatToken &Tok;
  unsigned StartColumn;
  StringRef TokenText;
};

class BreakableStringLiteral : public BreakableToken {
public:
  BreakableStringLiteral(const SourceManager &SourceMgr, const FormatToken &Tok,
                         unsigned StartColumn)
      : BreakableToken(SourceMgr, Tok, StartColumn) {
    assert(TokenText.startswith("\"") && TokenText.endswith("\""));
  }

  virtual unsigned getLineCount() const { return 1; }

  virtual unsigned getLineSize(unsigned Index) const {
    return Tok.TokenLength - 2; // Should be in sync with getLine
  }

  virtual unsigned getLineLengthAfterSplit(unsigned LineIndex,
                                           unsigned TailOffset) const {
    return getDecorationLength() + getLine().size() - TailOffset;
  }

  virtual Split getSplit(unsigned LineIndex, unsigned TailOffset,
                         unsigned ColumnLimit) const {
    StringRef Text = getLine().substr(TailOffset);
    if (ColumnLimit <= getDecorationLength())
      return Split(StringRef::npos, 0);
    unsigned MaxSplit = ColumnLimit - getDecorationLength();
    // FIXME: Reduce unit test case.
    if (Text.empty())
      return Split(StringRef::npos, 0);
    MaxSplit = std::min<unsigned>(MaxSplit, Text.size() - 1);
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
    Whitespaces.breakToken(Tok, 1 + TailOffset + Split.first, Split.second,
                           "\"", "\"", InPPDirective, StartColumn);
  }

private:
  StringRef getLine() const {
    // Get string without quotes.
    // FIXME: Handle string prefixes.
    return TokenText.substr(1, TokenText.size() - 2);
  }

  unsigned getDecorationLength() const { return StartColumn + 2; }

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

};

class BreakableComment : public BreakableToken {
public:
  virtual unsigned getLineSize(unsigned Index) const {
    return getLine(Index).size();
  }

  virtual unsigned getLineCount() const { return Lines.size(); }

  virtual unsigned getLineLengthAfterSplit(unsigned LineIndex,
                                           unsigned TailOffset) const {
    return getContentStartColumn(LineIndex, TailOffset) +
           getLine(LineIndex).size() - TailOffset;
  }

  virtual Split getSplit(unsigned LineIndex, unsigned TailOffset,
                         unsigned ColumnLimit) const;
  virtual void insertBreak(unsigned LineIndex, unsigned TailOffset, Split Split,
                           bool InPPDirective, WhitespaceManager &Whitespaces);

protected:
  BreakableComment(const SourceManager &SourceMgr, const FormatToken &Tok,
                   unsigned StartColumn)
      : BreakableToken(SourceMgr, Tok, StartColumn) {}

  // Get comment lines without /* */, common prefix and trailing whitespace.
  // Last line is not trimmed, as it is terminated by */, so its trailing
  // whitespace is not really trailing.
  StringRef getLine(unsigned Index) const {
    return Index < Lines.size() - 1 ? Lines[Index].rtrim() : Lines[Index];
  }

  unsigned getContentStartColumn(unsigned LineIndex,
                                 unsigned TailOffset) const {
    return (TailOffset == 0 && LineIndex == 0)
               ? StartColumn
               : IndentAtLineBreak + Decoration.size();
  }

  unsigned IndentAtLineBreak;
  StringRef Decoration;
  SmallVector<StringRef, 16> Lines;
};

class BreakableBlockComment : public BreakableComment {
public:
  BreakableBlockComment(const SourceManager &SourceMgr,
                        const AnnotatedToken &Token, unsigned StartColumn);

  void alignLines(WhitespaceManager &Whitespaces);

  virtual unsigned getLineLengthAfterSplit(unsigned LineIndex,
                                           unsigned TailOffset) const {
    return BreakableComment::getLineLengthAfterSplit(LineIndex, TailOffset) +
           (LineIndex + 1 < Lines.size() ? 0 : 2);
  }

  virtual void trimLine(unsigned LineIndex, unsigned TailOffset,
                        unsigned InPPDirective, WhitespaceManager &Whitespaces);

private:
  unsigned OriginalStartColumn;
  unsigned CommonPrefixLength;
};

class BreakableLineComment : public BreakableComment {
public:
  BreakableLineComment(const SourceManager &SourceMgr,
                       const AnnotatedToken &Token, unsigned StartColumn);

private:
  static StringRef getLineCommentPrefix(StringRef Comment);
};

} // namespace format
} // namespace clang

#endif // LLVM_CLANG_FORMAT_BREAKABLETOKEN_H
