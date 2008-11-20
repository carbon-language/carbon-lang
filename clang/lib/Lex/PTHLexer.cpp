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
#include "clang/Basic/TokenKinds.h"
using namespace clang;

PTHLexer::PTHLexer(Preprocessor& pp, SourceLocation fileloc,
                   const Token *TokArray, unsigned NumTokens)
  : PreprocessorLexer(&pp, fileloc),
    Tokens(TokArray),
    LastToken(NumTokens - 1),
    CurToken(0) {

  assert (NumTokens >= 1);
  assert (Tokens[LastToken].is(tok::eof));
}

void PTHLexer::Lex(Token& Tok) {

  if (CurToken == LastToken) {    
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
      PP->HandleEndOfFile(Tok, false);
    
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
    PP->HandleDirective(Tok);
    PP->Lex(Tok);
    return;
  }
    
  MIOpt.ReadToken();
}

void PTHLexer::setEOF(Token& Tok) {
  Tok = Tokens[LastToken];
}

void PTHLexer::DiscardToEndOfLine() {
  assert(ParsingPreprocessorDirective && ParsingFilename == false &&
         "Must be in a preprocessing directive!");

  // Already at end-of-file?
  if (CurToken == LastToken)
    return;

  // Find the first token that is not the start of the *current* line.
  for ( ++CurToken; CurToken != LastToken ; ++CurToken )
    if (Tokens[CurToken].isAtStartOfLine())
      return;
}

unsigned PTHLexer::isNextPPTokenLParen() {  
  if (CurToken == LastToken)
    return 2;
  
  return Tokens[CurToken].is(tok::l_paren);
}

