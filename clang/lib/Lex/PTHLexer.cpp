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
    LastTokenIdx(NumTokens - 1),
    CurTokenIdx(0) {

  assert(NumTokens >= 1);
  assert(Tokens[LastTokenIdx].is(tok::eof));
}

Token PTHLexer::GetToken() { 
  Token Tok = Tokens[CurTokenIdx];
  
  // If we are in raw mode, zero out identifier pointers.  This is
  // needed for 'pragma poison'.  Note that this requires that the Preprocessor
  // can go back to the original source when it calls getSpelling().
  if (LexingRawMode && Tok.is(tok::identifier))
    Tok.setIdentifierInfo(0);

  return Tok;
}

void PTHLexer::Lex(Token& Tok) {
LexNextToken:
  Tok = GetToken();
  
  if (AtLastToken()) {
    Preprocessor *PPCache = PP;

    if (LexEndOfFile(Tok))
      return;

    assert(PPCache && "Raw buffer::LexEndOfFile should return a token");
    return PPCache->Lex(Tok);
  }
  
  // Don't advance to the next token yet.  Check if we are at the
  // start of a new line and we're processing a directive.  If so, we
  // consume this token twice, once as an tok::eom.
  if (Tok.isAtStartOfLine() && ParsingPreprocessorDirective) {
    ParsingPreprocessorDirective = false;
    Tok.setKind(tok::eom);
    MIOpt.ReadToken();
    return;
  }
  
  // Advance to the next token.
  AdvanceToken();
    
  if (Tok.is(tok::hash)) {    
    if (Tok.isAtStartOfLine() && !LexingRawMode) {
      PP->HandleDirective(Tok);

      if (PP->isCurrentLexer(this))
        goto LexNextToken;

      return PP->Lex(Tok);
    }
  }

  MIOpt.ReadToken();
  
  if (Tok.is(tok::identifier)) {
    if (LexingRawMode) return;
    return PP->HandleIdentifier(Tok);
  }  
}

bool PTHLexer::LexEndOfFile(Token &Tok) {
  
  if (ParsingPreprocessorDirective) {
    ParsingPreprocessorDirective = false;
    Tok.setKind(tok::eom);
    MIOpt.ReadToken();
    return true; // Have a token.
  }
  
  if (LexingRawMode) {
    MIOpt.ReadToken();
    return true;  // Have an eof token.
  }
  
  // FIXME: Issue diagnostics similar to Lexer.
  return PP->HandleEndOfFile(Tok, false);
}

void PTHLexer::setEOF(Token& Tok) {
  Tok = Tokens[LastTokenIdx];
}

void PTHLexer::DiscardToEndOfLine() {
  assert(ParsingPreprocessorDirective && ParsingFilename == false &&
         "Must be in a preprocessing directive!");

  // Already at end-of-file?
  if (AtLastToken())
    return;

  // Find the first token that is not the start of the *current* line.
  Token T;
  for (Lex(T); !AtLastToken(); Lex(T))
    if (GetToken().isAtStartOfLine())
      return;
}
