//===--- MacroExpander.cpp - Lex from a macro expansion -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MacroExpander interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/MacroExpander.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/SourceManager.h"
using namespace llvm;
using namespace clang;

MacroExpander::MacroExpander(LexerToken &Tok, Preprocessor &pp)
  : Macro(*Tok.getIdentifierInfo()->getMacroInfo()), PP(pp), CurToken(0),
    InstantiateLoc(Tok.getLocation()),
    AtStartOfLine(Tok.isAtStartOfLine()),
    HasLeadingSpace(Tok.hasLeadingSpace()) {
}

/// Lex - Lex and return a token from this macro stream.
///
void MacroExpander::Lex(LexerToken &Tok) {
  // Lexing off the end of the macro, pop this macro off the expansion stack.
  if (CurToken == Macro.getNumTokens())
    return PP.HandleEndOfMacro(Tok);
  
  // Get the next token to return.
  Tok = Macro.getReplacementToken(CurToken++);
  //Tok.SetLocation(InstantiateLoc);

  // The token's current location indicate where the token was lexed from.  We
  // need this information to compute the spelling of the token, but any
  // diagnostics for the expanded token should appear as if they came from
  // InstantiateLoc.  Pull this information together into a new SourceLocation
  // that captures all of this.
  unsigned CharFilePos = Tok.getLocation().getRawFilePos();
  unsigned CharFileID  = Tok.getLocation().getFileID();
  
  unsigned InstantiationFileID =
    PP.getSourceManager().createFileIDForMacroExp(InstantiateLoc, CharFileID);
  Tok.SetLocation(SourceLocation(InstantiationFileID, CharFilePos));

  // If this is the first token, set the lexical properties of the token to
  // match the lexical properties of the macro identifier.
  if (CurToken == 1) {
    Tok.SetFlagValue(LexerToken::StartOfLine , AtStartOfLine);
    Tok.SetFlagValue(LexerToken::LeadingSpace, HasLeadingSpace);
  }
  
  // Handle recursive expansion!
  if (Tok.getIdentifierInfo())
    return PP.HandleIdentifier(Tok);

  // Otherwise, return a normal token.
}
