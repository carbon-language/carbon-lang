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

#include "Encoding.h"
#include "TokenAnnotator.h"
#include "WhitespaceManager.h"
#include <utility>

namespace clang {
namespace format {

struct FormatStyle;

/// \brief Base class for strategies on how to break tokens.
///
/// FIXME: The interface seems set in stone, so we might want to just pull the
/// strategy into the class, instead of controlling it from the outside.
class BreakableToken {
public:
  /// \brief Contains starting character index and length of split.
  typedef std::pair<StringRef::size_type, unsigned> Split;

  virtual ~BreakableToken() {}

  /// \brief Returns the number of lines in this token in the original code.
  virtual unsigned getLineCount() const = 0;

  /// \brief Returns the number of columns required to format the piece of line
  /// at \p LineIndex, from byte offset \p Offset with length \p Length.
  ///
  /// Note that previous breaks are not taken into account. \p Offset is always
  /// specified from the start of the (original) line.
  /// \p Length can be set to StringRef::npos, which means "to the end of line".
  virtual unsigned
  getLineLengthAfterSplit(unsigned LineIndex, unsigned Offset,
                          StringRef::size_type Length) const = 0;

  /// \brief Returns a range (offset, length) at which to break the line at
  /// \p LineIndex, if previously broken at \p TailOffset. If possible, do not
  /// violate \p ColumnLimit.
  virtual Split getSplit(unsigned LineIndex, unsigned TailOffset,
                         unsigned ColumnLimit) const = 0;

  /// \brief Emits the previously retrieved \p Split via \p Whitespaces.
  virtual void insertBreak(unsigned LineIndex, unsigned TailOffset, Split Split,
                           WhitespaceManager &Whitespaces) = 0;

  /// \brief Replaces the whitespace range described by \p Split with a single
  /// space.
  virtual void replaceWhitespace(unsigned LineIndex, unsigned TailOffset,
                                 Split Split,
                                 WhitespaceManager &Whitespaces) = 0;

  /// \brief Replaces the whitespace between \p LineIndex-1 and \p LineIndex.
  virtual void replaceWhitespaceBefore(unsigned LineIndex,
                                       WhitespaceManager &Whitespaces) {}

protected:
  BreakableToken(const FormatToken &Tok, unsigned IndentLevel,
                 bool InPPDirective, encoding::Encoding Encoding,
                 const FormatStyle &Style)
      : Tok(Tok), IndentLevel(IndentLevel), InPPDirective(InPPDirective),
        Encoding(Encoding), Style(Style) {}

  const FormatToken &Tok;
  const unsigned IndentLevel;
  const bool InPPDirective;
  const encoding::Encoding Encoding;
  const FormatStyle &Style;
};

/// \brief Base class for single line tokens that can be broken.
///
/// \c getSplit() needs to be implemented by child classes.
class BreakableSingleLineToken : public BreakableToken {
public:
  virtual unsigned getLineCount() const;
  virtual unsigned getLineLengthAfterSplit(unsigned LineIndex,
                                           unsigned TailOffset,
                                           StringRef::size_type Length) const;

protected:
  BreakableSingleLineToken(const FormatToken &Tok, unsigned IndentLevel,
                           unsigned StartColumn, StringRef Prefix,
                           StringRef Postfix, bool InPPDirective,
                           encoding::Encoding Encoding,
                           const FormatStyle &Style);

  // The column in which the token starts.
  unsigned StartColumn;
  // The prefix a line needs after a break in the token.
  StringRef Prefix;
  // The postfix a line needs before introducing a break.
  StringRef Postfix;
  // The token text excluding the prefix and postfix.
  StringRef Line;
};

class BreakableStringLiteral : public BreakableSingleLineToken {
public:
  /// \brief Creates a breakable token for a single line string literal.
  ///
  /// \p StartColumn specifies the column in which the token will start
  /// after formatting.
  BreakableStringLiteral(const FormatToken &Tok, unsigned IndentLevel,
                         unsigned StartColumn, StringRef Prefix,
                         StringRef Postfix, bool InPPDirective,
                         encoding::Encoding Encoding, const FormatStyle &Style);

  virtual Split getSplit(unsigned LineIndex, unsigned TailOffset,
                         unsigned ColumnLimit) const;
  virtual void insertBreak(unsigned LineIndex, unsigned TailOffset, Split Split,
                           WhitespaceManager &Whitespaces);
  virtual void replaceWhitespace(unsigned LineIndex, unsigned TailOffset,
                                 Split Split,
                                 WhitespaceManager &Whitespaces) {}
};

class BreakableLineComment : public BreakableSingleLineToken {
public:
  /// \brief Creates a breakable token for a line comment.
  ///
  /// \p StartColumn specifies the column in which the comment will start
  /// after formatting.
  BreakableLineComment(const FormatToken &Token, unsigned IndentLevel,
                       unsigned StartColumn, bool InPPDirective,
                       encoding::Encoding Encoding, const FormatStyle &Style);

  virtual Split getSplit(unsigned LineIndex, unsigned TailOffset,
                         unsigned ColumnLimit) const;
  virtual void insertBreak(unsigned LineIndex, unsigned TailOffset, Split Split,
                           WhitespaceManager &Whitespaces);
  virtual void replaceWhitespace(unsigned LineIndex, unsigned TailOffset,
                                 Split Split,
                                 WhitespaceManager &Whitespaces);
  virtual void replaceWhitespaceBefore(unsigned LineIndex,
                                       WhitespaceManager &Whitespaces);

private:
  // The prefix without an additional space if one was added.
  StringRef OriginalPrefix;
};

class BreakableBlockComment : public BreakableToken {
public:
  /// \brief Creates a breakable token for a block comment.
  ///
  /// \p StartColumn specifies the column in which the comment will start
  /// after formatting, while \p OriginalStartColumn specifies in which
  /// column the comment started before formatting.
  /// If the comment starts a line after formatting, set \p FirstInLine to true.
  BreakableBlockComment(const FormatToken &Token, unsigned IndentLevel,
                        unsigned StartColumn, unsigned OriginaStartColumn,
                        bool FirstInLine, bool InPPDirective,
                        encoding::Encoding Encoding, const FormatStyle &Style);

  virtual unsigned getLineCount() const;
  virtual unsigned getLineLengthAfterSplit(unsigned LineIndex,
                                           unsigned TailOffset,
                                           StringRef::size_type Length) const;
  virtual Split getSplit(unsigned LineIndex, unsigned TailOffset,
                         unsigned ColumnLimit) const;
  virtual void insertBreak(unsigned LineIndex, unsigned TailOffset, Split Split,
                           WhitespaceManager &Whitespaces);
  virtual void replaceWhitespace(unsigned LineIndex, unsigned TailOffset,
                                 Split Split,
                                 WhitespaceManager &Whitespaces);
  virtual void replaceWhitespaceBefore(unsigned LineIndex,
                                       WhitespaceManager &Whitespaces);

private:
  // Rearranges the whitespace between Lines[LineIndex-1] and Lines[LineIndex],
  // so that all whitespace between the lines is accounted to Lines[LineIndex]
  // as leading whitespace:
  // - Lines[LineIndex] points to the text after that whitespace
  // - Lines[LineIndex-1] shrinks by its trailing whitespace
  // - LeadingWhitespace[LineIndex] is updated with the complete whitespace
  //   between the end of the text of Lines[LineIndex-1] and Lines[LineIndex]
  //
  // Sets StartOfLineColumn to the intended column in which the text at
  // Lines[LineIndex] starts (note that the decoration, if present, is not
  // considered part of the text).
  void adjustWhitespace(unsigned LineIndex, int IndentDelta);

  // Returns the column at which the text in line LineIndex starts, when broken
  // at TailOffset. Note that the decoration (if present) is not considered part
  // of the text.
  unsigned getContentStartColumn(unsigned LineIndex, unsigned TailOffset) const;

  // Contains the text of the lines of the block comment, excluding the leading
  // /* in the first line and trailing */ in the last line, and excluding all
  // trailing whitespace between the lines. Note that the decoration (if
  // present) is also not considered part of the text.
  SmallVector<StringRef, 16> Lines;

  // LeadingWhitespace[i] is the number of characters regarded as whitespace in
  // front of Lines[i]. Note that this can include "* " sequences, which we
  // regard as whitespace when all lines have a "*" prefix.
  SmallVector<unsigned, 16> LeadingWhitespace;

  // StartOfLineColumn[i] is the target column at which Line[i] should be.
  // Note that this excludes a leading "* " or "*" in case all lines have
  // a "*" prefix.
  SmallVector<unsigned, 16> StartOfLineColumn;

  // The column at which the text of a broken line should start.
  // Note that an optional decoration would go before that column.
  // IndentAtLineBreak is a uniform position for all lines in a block comment,
  // regardless of their relative position.
  // FIXME: Revisit the decision to do this; the main reason was to support
  // patterns like
  // /**************//**
  //  * Comment
  // We could also support such patterns by special casing the first line
  // instead.
  unsigned IndentAtLineBreak;

  // This is to distinguish between the case when the last line was empty and
  // the case when it started with a decoration ("*" or "* ").
  bool LastLineNeedsDecoration;

  // Either "* " if all lines begin with a "*", or empty.
  StringRef Decoration;
};

} // namespace format
} // namespace clang

#endif // LLVM_CLANG_FORMAT_BREAKABLETOKEN_H
