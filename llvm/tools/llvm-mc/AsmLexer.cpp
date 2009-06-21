//===- AsmLexer.cpp - Lexer for Assembly Files ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the lexer for assembly files.
//
//===----------------------------------------------------------------------===//

#include "AsmLexer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
using namespace llvm;

AsmLexer::AsmLexer(SourceMgr &SM) : SrcMgr(SM) {
  CurBuffer = 0;
  CurBuf = SrcMgr.getMemoryBuffer(CurBuffer);
  CurPtr = CurBuf->getBufferStart();
  TokStart = 0;
}

void AsmLexer::PrintError(const char *Loc, const std::string &Msg) const {
  SrcMgr.PrintError(SMLoc::getFromPointer(Loc), Msg);
}

void AsmLexer::PrintError(SMLoc Loc, const std::string &Msg) const {
  SrcMgr.PrintError(Loc, Msg);
}

int AsmLexer::getNextChar() {
  char CurChar = *CurPtr++;
  switch (CurChar) {
  default:
    return (unsigned char)CurChar;
  case 0: {
    // A nul character in the stream is either the end of the current buffer or
    // a random nul in the file.  Disambiguate that here.
    if (CurPtr-1 != CurBuf->getBufferEnd())
      return 0;  // Just whitespace.
    
    // If this is the end of an included file, pop the parent file off the
    // include stack.
    SMLoc ParentIncludeLoc = SrcMgr.getParentIncludeLoc(CurBuffer);
    if (ParentIncludeLoc != SMLoc()) {
      CurBuffer = SrcMgr.FindBufferContainingLoc(ParentIncludeLoc);
      CurBuf = SrcMgr.getMemoryBuffer(CurBuffer);
      CurPtr = ParentIncludeLoc.getPointer();
      return getNextChar();
    }
    
    // Otherwise, return end of file.
    --CurPtr;  // Another call to lex will return EOF again.  
    return EOF;
  }
  }
}

asmtok::TokKind AsmLexer::LexToken() {
  TokStart = CurPtr;
  // This always consumes at least one character.
  int CurChar = getNextChar();
  
  switch (CurChar) {
  default:
    // Handle letters: [a-zA-Z_]
//    if (isalpha(CurChar) || CurChar == '_' || CurChar == '#')
//      return LexIdentifier();
    
    // Unknown character, emit an error.
    return asmtok::Error;
  case EOF: return asmtok::Eof;
  case 0:
  case ' ':
  case '\t':
  case '\n':
  case '\r':
    // Ignore whitespace.
    return LexToken();
  case ':': return asmtok::Colon;
  case '+': return asmtok::Plus;
  case '-': return asmtok::Minus;
  }
}