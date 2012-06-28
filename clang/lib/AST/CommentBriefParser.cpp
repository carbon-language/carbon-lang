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

namespace {
/// Convert all whitespace into spaces, remove leading and trailing spaces,
/// compress multiple spaces into one.
void cleanupBrief(std::string &S) {
  bool PrevWasSpace = true;
  std::string::iterator O = S.begin();
  for (std::string::iterator I = S.begin(), E = S.end();
       I != E; ++I) {
    const char C = *I;
    if (C == ' ' || C == '\n' || C == '\r' ||
        C == '\t' || C == '\v' || C == '\f') {
      if (!PrevWasSpace) {
        *O++ = ' ';
        PrevWasSpace = true;
      }
      continue;
    } else {
      *O++ = C;
      PrevWasSpace = false;
    }
  }
  if (O != S.begin() && *(O - 1) == ' ')
    --O;

  S.resize(O - S.begin());
}
} // unnamed namespace

std::string BriefParser::Parse() {
  std::string Paragraph;
  bool InFirstParagraph = true;
  bool InBrief = false;

  while (Tok.isNot(tok::eof)) {
    if (Tok.is(tok::text)) {
      if (InFirstParagraph || InBrief)
        Paragraph += Tok.getText();
      ConsumeToken();
      continue;
    }

    if (Tok.is(tok::command)) {
      StringRef Name = Tok.getCommandName();
      if (Name == "brief") {
        Paragraph.clear();
        InBrief = true;
        ConsumeToken();
        continue;
      }
      // Check if this command implicitly starts a new paragraph.
      if (Name == "param" || Name == "result" || Name == "return" ||
          Name == "returns") {
        // We found an implicit paragraph end.
        InFirstParagraph = false;
        if (InBrief) {
          InBrief = false;
          break;
        }
      }
    }

    if (Tok.is(tok::newline)) {
      if (InFirstParagraph || InBrief)
        Paragraph += ' ';
      ConsumeToken();

      if (Tok.is(tok::newline)) {
        ConsumeToken();
        // We found a paragraph end.
        InFirstParagraph = false;
        if (InBrief) {
          InBrief = false;
          break;
        }
      }
      continue;
    }

    // We didn't handle this token, so just drop it.
    ConsumeToken();
  }

  cleanupBrief(Paragraph);
  return Paragraph;
}

BriefParser::BriefParser(Lexer &L) : L(L)
{
  // Get lookahead token.
  ConsumeToken();
}

} // end namespace comments
} // end namespace clang


