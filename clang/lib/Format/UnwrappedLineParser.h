//===--- UnwrappedLineParser.cpp - Format C++ code ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the declaration of the UnwrappedLineParser,
/// which turns a stream of tokens into UnwrappedLines.
///
/// This is EXPERIMENTAL code under heavy development. It is not in a state yet,
/// where it can be used to format real code.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FORMAT_UNWRAPPED_LINE_PARSER_H
#define LLVM_CLANG_FORMAT_UNWRAPPED_LINE_PARSER_H

#include "clang/Basic/SourceManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Lex/Lexer.h"

namespace clang {
namespace format {

/// \brief A wrapper around a \c Token storing information about the
/// whitespace characters preceeding it.
struct FormatToken {
  FormatToken() : NewlinesBefore(0), WhiteSpaceLength(0) {
  }

  /// \brief The \c Token.
  Token Tok;

  /// \brief The number of newlines immediately before the \c Token.
  ///
  /// This can be used to determine what the user wrote in the original code
  /// and thereby e.g. leave an empty line between two function definitions.
  unsigned NewlinesBefore;

  /// \brief The location of the start of the whitespace immediately preceeding
  /// the \c Token.
  ///
  /// Used together with \c WhiteSpaceLength to create a \c Replacement.
  SourceLocation WhiteSpaceStart;

  /// \brief The length in characters of the whitespace immediately preceeding
  /// the \c Token.
  unsigned WhiteSpaceLength;
};

/// \brief An unwrapped line is a sequence of \c Token, that we would like to
/// put on a single line if there was no column limit.
///
/// This is used as a main interface between the \c UnwrappedLineParser and the
/// \c UnwrappedLineFormatter. The key property is that changing the formatting
/// within an unwrapped line does not affect any other unwrapped lines.
struct UnwrappedLine {
  UnwrappedLine() : Level(0) {
  }

  /// \brief The \c Token comprising this \c UnwrappedLine.
  SmallVector<FormatToken, 16> Tokens;

  /// \brief The indent level of the \c UnwrappedLine.
  unsigned Level;
};

class UnwrappedLineConsumer {
public:
  virtual void formatUnwrappedLine(const UnwrappedLine &Line) = 0;
};

class UnwrappedLineParser {
public:
  UnwrappedLineParser(Lexer &Lex, SourceManager &SourceMgr,
                      UnwrappedLineConsumer &Callback);

  void parse();

private:
  void parseLevel();
  void parseBlock();
  void parsePPDirective();
  void parseComment();
  void parseStatement();
  void parseParens();
  void parseIfThenElse();
  void parseDoWhile();
  void parseLabel();
  void parseCaseLabel();
  void parseSwitch();
  void parseAccessSpecifier();
  void parseEnum();
  void addUnwrappedLine();
  bool eof() const;
  void nextToken();
  void parseToken();

  /// Returns the text of \c FormatTok.
  StringRef tokenText();

  UnwrappedLine Line;
  FormatToken FormatTok;
  bool GreaterStashed;

  Lexer &Lex;
  SourceManager &SourceMgr;
  IdentifierTable IdentTable;
  UnwrappedLineConsumer &Callback;
};

}  // end namespace format
}  // end namespace clang

#endif // LLVM_CLANG_FORMAT_UNWRAPPED_LINE_PARSER_H
