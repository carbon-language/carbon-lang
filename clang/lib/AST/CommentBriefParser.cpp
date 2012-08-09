//===--- CommentBriefParser.cpp - Dumb comment parser ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/CommentBriefParser.h"
#include "clang/AST/CommentCommandTraits.h"
#include "llvm/ADT/StringSwitch.h"

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

BriefParser::BriefParser(Lexer &L, const CommandTraits &Traits) :
    L(L), Traits(Traits) {
  // Get lookahead token.
  ConsumeToken();
}

std::string BriefParser::Parse() {
  std::string FirstParagraphOrBrief;
  std::string ReturnsParagraph;
  bool InFirstParagraph = true;
  bool InBrief = false;
  bool InReturns = false;

  while (Tok.isNot(tok::eof)) {
    if (Tok.is(tok::text)) {
      if (InFirstParagraph || InBrief)
        FirstParagraphOrBrief += Tok.getText();
      else if (InReturns)
        ReturnsParagraph += Tok.getText();
      ConsumeToken();
      continue;
    }

    if (Tok.is(tok::command)) {
      StringRef Name = Tok.getCommandName();
      if (Traits.isBriefCommand(Name)) {
        FirstParagraphOrBrief.clear();
        InBrief = true;
        ConsumeToken();
        continue;
      }
      if (Traits.isReturnsCommand(Name)) {
        InReturns = true;
        ReturnsParagraph += "Returns ";
      }
      // Block commands implicitly start a new paragraph.
      if (Traits.isBlockCommand(Name)) {
        // We found an implicit paragraph end.
        InFirstParagraph = false;
        if (InBrief)
          break;
      }
    }

    if (Tok.is(tok::newline)) {
      if (InFirstParagraph || InBrief)
        FirstParagraphOrBrief += ' ';
      else if (InReturns)
        ReturnsParagraph += ' ';
      ConsumeToken();

      if (Tok.is(tok::newline)) {
        ConsumeToken();
        // We found a paragraph end.
        InFirstParagraph = false;
        InReturns = false;
        if (InBrief)
          break;
      }
      continue;
    }

    // We didn't handle this token, so just drop it.
    ConsumeToken();
  }

  cleanupBrief(FirstParagraphOrBrief);
  if (!FirstParagraphOrBrief.empty())
    return FirstParagraphOrBrief;

  cleanupBrief(ReturnsParagraph);
  return ReturnsParagraph;
}

} // end namespace comments
} // end namespace clang


