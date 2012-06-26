//===--- CommentBriefParser.h - Dumb comment parser -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a very simple Doxygen comment parser.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_CLANG_AST_BRIEF_COMMENT_PARSER_H
#define LLVM_CLANG_AST_BRIEF_COMMENT_PARSER_H

#include "clang/AST/CommentLexer.h"

namespace clang {
namespace comments {

/// A very simple comment parser that extracts just the brief description or
/// first paragraph.
class BriefParser {
  Lexer &L;

  /// Current lookahead token.
  Token Tok;

  SourceLocation ConsumeToken() {
    SourceLocation Loc = Tok.getLocation();
    L.lex(Tok);
    return Loc;
  }

public:
  BriefParser(Lexer &L);

  /// Return \\brief paragraph, if it exists; otherwise return the first
  /// paragraph.
  std::string Parse();
};

} // end namespace comments
} // end namespace clang

#endif

