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

  assert(NumTokens >= 1);
  assert(Tokens[LastToken].is(tok::eof));
}

void PTHLexer::Lex(Token& Tok) {
LexNextToken:
  if (CurToken == LastToken) {
    if (ParsingPreprocessorDirective) {
      ParsingPreprocessorDirective = false;
      Tok = Tokens[LastToken];
      Tok.setKind(tok::eom);
      MIOpt.ReadToken();
      return;
    }
    
    assert(!LexingRawMode && "PTHLexer cannot lex in raw mode.");
    
    // FIXME: Issue diagnostics similar to Lexer.
    PP->HandleEndOfFile(Tok, false);    
    return;
  }

  Tok = Tokens[CurToken];
  
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
  ++CurToken;
    
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

