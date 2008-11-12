//===--- PTHLexer.cpp - Lex from a token stream ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PTHLexer interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/PTHLexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include "clang/Basic/TokenKinds.h"

using namespace clang;

PTHLexer::PTHLexer(Preprocessor& pp, SourceLocation fileloc,
                   const Token *TokArray, unsigned NumToks)
  : PP(pp), FileLoc(fileloc), Tokens(TokArray), NumTokens(NumToks), CurToken(0){

  assert (Tokens[NumTokens-1].is(tok::eof));
  --NumTokens;
    
  LexingRawMode = false;
  ParsingPreprocessorDirective = false;
  ParsingFilename = false;
}

void PTHLexer::Lex(Token& Tok) {

  if (CurToken == NumTokens) {    
    // If we hit the end of the file while parsing a preprocessor directive,
    // end the preprocessor directive first.  The next token returned will
    // then be the end of file.
    //   OR
    // If we are in raw mode, return this event as an EOF token.  Let the caller
    // that put us in raw mode handle the event.
    if (ParsingPreprocessorDirective || LexingRawMode) {
      // Done parsing the "line".
      ParsingPreprocessorDirective = false;
      Tok = Tokens[CurToken]; // not an out-of-bound access
      // FIXME: eom handling?
    }
    else
      PP.HandleEndOfFile(Tok, false);
    
    return;
  }

  Tok = Tokens[CurToken];
  
  if (ParsingPreprocessorDirective && Tok.isAtStartOfLine()) {
    ParsingPreprocessorDirective = false; // Done parsing the "line".
    MIOpt.ReadToken();
    // FIXME:  Need to replicate:
    // FormTokenWithChars(Tok, CurPtr, tok::eom);
    Tok.setKind(tok::eom);    
    return;
  }
  else // Otherwise, advance to the next token.
    ++CurToken;

  if (Tok.isAtStartOfLine() && Tok.is(tok::hash) && !LexingRawMode) {
    PP.HandleDirective(Tok);
    PP.Lex(Tok);
    return;
  }
    
  MIOpt.ReadToken();
}

void PTHLexer::setEOF(Token& Tok) {
  Tok = Tokens[NumTokens]; // NumTokens is already adjusted, so this isn't
                           // an overflow.
}
