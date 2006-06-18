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
using namespace llvm;
using namespace clang;

/// Lex - Lex and return a token from this macro stream.
void MacroExpander::Lex(LexerToken &Tok) {
  // Lexing off the end of the macro, pop this macro off the expansion stack.
  if (CurToken == Macro.getNumTokens())
    return PP.HandleEndOfMacro(Tok);
  
  // Get the next token to return.
  Tok = Macro.getReplacementToken(CurToken++);

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
