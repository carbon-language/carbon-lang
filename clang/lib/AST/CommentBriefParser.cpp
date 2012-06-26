//===--- CommentBriefParser.cpp - Dumb comment parser ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/CommentBriefParser.h"

namespace clang {
namespace comments {

std::string BriefParser::Parse() {
  std::string FirstParagraph;
  std::string Brief;
  bool InFirstParagraph = true;
  bool InBrief = false;
  bool BriefDone = false;

  while (Tok.isNot(tok::eof)) {
    if (Tok.is(tok::text)) {
      if (InFirstParagraph)
        FirstParagraph += Tok.getText();
      if (InBrief)
        Brief += Tok.getText();
      ConsumeToken();
      continue;
    }

    if (!BriefDone && Tok.is(tok::command) && Tok.getCommandName() == "brief") {
      InBrief = true;
      ConsumeToken();
      continue;
    }

    if (Tok.is(tok::newline)) {
      if (InFirstParagraph)
        FirstParagraph += '\n';
      if (InBrief)
        Brief += '\n';
      ConsumeToken();

      if (Tok.is(tok::newline)) {
        ConsumeToken();
        // We found a paragraph end.
        InFirstParagraph = false;
        if (InBrief) {
          InBrief = false;
          BriefDone = true;
        }
      }
      continue;
    }

    // We didn't handle this token, so just drop it.
    ConsumeToken();
  }

  if (Brief.size() > 0)
    return Brief;

  return FirstParagraph;
}

BriefParser::BriefParser(Lexer &L) : L(L)
{
  // Get lookahead token.
  ConsumeToken();
}

} // end namespace comments
} // end namespace clang


