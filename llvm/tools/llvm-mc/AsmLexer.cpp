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
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/MC/MCAsmInfo.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
using namespace llvm;

AsmLexer::AsmLexer(const MCAsmInfo &_MAI) : MAI(_MAI)  {
  CurBuf = NULL;
  CurPtr = NULL;
  TokStart = 0;
}

AsmLexer::~AsmLexer() {
}

void AsmLexer::setBuffer(const MemoryBuffer *buf, const char *ptr) {
  CurBuf = buf;
  
  if (ptr)
    CurPtr = ptr;
  else
    CurPtr = CurBuf->getBufferStart();
  
  TokStart = 0;
}

SMLoc AsmLexer::getLoc() const {
  return SMLoc::getFromPointer(TokStart);
}

/// ReturnError - Set the error to the specified string at the specified
/// location.  This is defined to always return AsmToken::Error.
AsmToken AsmLexer::ReturnError(const char *Loc, const std::string &Msg) {
  SetError(SMLoc::getFromPointer(Loc), Msg);
  
  return AsmToken(AsmToken::Error, StringRef(Loc, 0));
}

int AsmLexer::getNextChar() {
  char CurChar = *CurPtr++;
  switch (CurChar) {
  default:
    return (unsigned char)CurChar;
  case 0:
    // A nul character in the stream is either the end of the current buffer or
    // a random nul in the file.  Disambiguate that here.
    if (CurPtr-1 != CurBuf->getBufferEnd())
      return 0;  // Just whitespace.
    
    // Otherwise, return end of file.
    --CurPtr;  // Another call to lex will return EOF again.  
    return EOF;
  }
}

/// LexIdentifier: [a-zA-Z_.][a-zA-Z0-9_$.@]*
AsmToken AsmLexer::LexIdentifier() {
  while (isalnum(*CurPtr) || *CurPtr == '_' || *CurPtr == '$' ||
         *CurPtr == '.' || *CurPtr == '@')
    ++CurPtr;
  return AsmToken(AsmToken::Identifier, StringRef(TokStart, CurPtr - TokStart));
}

/// LexSlash: Slash: /
///           C-Style Comment: /* ... */
AsmToken AsmLexer::LexSlash() {
  switch (*CurPtr) {
  case '*': break; // C style comment.
  case '/': return ++CurPtr, LexLineComment();
  default:  return AsmToken(AsmToken::Slash, StringRef(CurPtr, 1));
  }

  // C Style comment.
  ++CurPtr;  // skip the star.
  while (1) {
    int CurChar = getNextChar();
    switch (CurChar) {
    case EOF:
      return ReturnError(TokStart, "unterminated comment");
    case '*':
      // End of the comment?
      if (CurPtr[0] != '/') break;
      
      ++CurPtr;   // End the */.
      return LexToken();
    }
  }
}

/// LexLineComment: Comment: #[^\n]*
///                        : //[^\n]*
AsmToken AsmLexer::LexLineComment() {
  // FIXME: This is broken if we happen to a comment at the end of a file, which
  // was .included, and which doesn't end with a newline.
  int CurChar = getNextChar();
  while (CurChar != '\n' && CurChar != '\n' && CurChar != EOF)
    CurChar = getNextChar();
  
  if (CurChar == EOF)
    return AsmToken(AsmToken::Eof, StringRef(CurPtr, 0));
  return AsmToken(AsmToken::EndOfStatement, StringRef(CurPtr, 0));
}


/// LexDigit: First character is [0-9].
///   Local Label: [0-9][:]
///   Forward/Backward Label: [0-9][fb]
///   Binary integer: 0b[01]+
///   Octal integer: 0[0-7]+
///   Hex integer: 0x[0-9a-fA-F]+
///   Decimal integer: [1-9][0-9]*
/// TODO: FP literal.
AsmToken AsmLexer::LexDigit() {
  if (*CurPtr == ':')
    return ReturnError(TokStart, "FIXME: local label not implemented");
  if (*CurPtr == 'f' || *CurPtr == 'b')
    return ReturnError(TokStart, "FIXME: directional label not implemented");
  
  // Decimal integer: [1-9][0-9]*
  if (CurPtr[-1] != '0') {
    while (isdigit(*CurPtr))
      ++CurPtr;
    return AsmToken(AsmToken::Integer, StringRef(TokStart, CurPtr - TokStart), 
                    strtoll(TokStart, 0, 10));
  }
  
  if (*CurPtr == 'b') {
    ++CurPtr;
    const char *NumStart = CurPtr;
    while (CurPtr[0] == '0' || CurPtr[0] == '1')
      ++CurPtr;
    
    // Requires at least one binary digit.
    if (CurPtr == NumStart)
      return ReturnError(CurPtr-2, "Invalid binary number");
    return AsmToken(AsmToken::Integer, StringRef(TokStart, CurPtr - TokStart),
                    strtoll(NumStart, 0, 2));
  }
 
  if (*CurPtr == 'x') {
    ++CurPtr;
    const char *NumStart = CurPtr;
    while (isxdigit(CurPtr[0]))
      ++CurPtr;
    
    // Requires at least one hex digit.
    if (CurPtr == NumStart)
      return ReturnError(CurPtr-2, "Invalid hexadecimal number");

    unsigned long long Result;
    if (StringRef(TokStart, CurPtr - TokStart).getAsInteger(0, Result))
      return ReturnError(CurPtr-2, "Invalid hexadecimal number");
      
    return AsmToken(AsmToken::Integer, StringRef(TokStart, CurPtr - TokStart),
                    (int64_t)Result);
  }
  
  // Must be an octal number, it starts with 0.
  while (*CurPtr >= '0' && *CurPtr <= '7')
    ++CurPtr;
  return AsmToken(AsmToken::Integer, StringRef(TokStart, CurPtr - TokStart),
                  strtoll(TokStart, 0, 8));
}

/// LexQuote: String: "..."
AsmToken AsmLexer::LexQuote() {
  int CurChar = getNextChar();
  // TODO: does gas allow multiline string constants?
  while (CurChar != '"') {
    if (CurChar == '\\') {
      // Allow \", etc.
      CurChar = getNextChar();
    }
    
    if (CurChar == EOF)
      return ReturnError(TokStart, "unterminated string constant");

    CurChar = getNextChar();
  }
  
  return AsmToken(AsmToken::String, StringRef(TokStart, CurPtr - TokStart));
}

StringRef AsmLexer::LexUntilEndOfStatement() {
  TokStart = CurPtr;

  while (!isAtStartOfComment(*CurPtr) && // Start of line comment.
	  *CurPtr != ';' &&  // End of statement marker.
         *CurPtr != '\n' &&
         *CurPtr != '\r' &&
         (*CurPtr != 0 || CurPtr != CurBuf->getBufferEnd())) {
    ++CurPtr;
  }
  return StringRef(TokStart, CurPtr-TokStart);
}

bool AsmLexer::isAtStartOfComment(char Char) {
  // FIXME: This won't work for multi-character comment indicators like "//".
  return Char == *MAI.getCommentString();
}

AsmToken AsmLexer::LexToken() {
  TokStart = CurPtr;
  // This always consumes at least one character.
  int CurChar = getNextChar();
  
  if (isAtStartOfComment(CurChar))
    return LexLineComment();

  switch (CurChar) {
  default:
    // Handle identifier: [a-zA-Z_.][a-zA-Z0-9_$.@]*
    if (isalpha(CurChar) || CurChar == '_' || CurChar == '.')
      return LexIdentifier();
    
    // Unknown character, emit an error.
    return ReturnError(TokStart, "invalid character in input");
  case EOF: return AsmToken(AsmToken::Eof, StringRef(TokStart, 0));
  case 0:
  case ' ':
  case '\t':
    // Ignore whitespace.
    return LexToken();
  case '\n': // FALL THROUGH.
  case '\r': // FALL THROUGH.
  case ';': return AsmToken(AsmToken::EndOfStatement, StringRef(TokStart, 1));
  case ':': return AsmToken(AsmToken::Colon, StringRef(TokStart, 1));
  case '+': return AsmToken(AsmToken::Plus, StringRef(TokStart, 1));
  case '-': return AsmToken(AsmToken::Minus, StringRef(TokStart, 1));
  case '~': return AsmToken(AsmToken::Tilde, StringRef(TokStart, 1));
  case '(': return AsmToken(AsmToken::LParen, StringRef(TokStart, 1));
  case ')': return AsmToken(AsmToken::RParen, StringRef(TokStart, 1));
  case '[': return AsmToken(AsmToken::LBrac, StringRef(TokStart, 1));
  case ']': return AsmToken(AsmToken::RBrac, StringRef(TokStart, 1));
  case '{': return AsmToken(AsmToken::LCurly, StringRef(TokStart, 1));
  case '}': return AsmToken(AsmToken::RCurly, StringRef(TokStart, 1));
  case '*': return AsmToken(AsmToken::Star, StringRef(TokStart, 1));
  case ',': return AsmToken(AsmToken::Comma, StringRef(TokStart, 1));
  case '$': return AsmToken(AsmToken::Dollar, StringRef(TokStart, 1));
  case '=': 
    if (*CurPtr == '=')
      return ++CurPtr, AsmToken(AsmToken::EqualEqual, StringRef(TokStart, 2));
    return AsmToken(AsmToken::Equal, StringRef(TokStart, 1));
  case '|': 
    if (*CurPtr == '|')
      return ++CurPtr, AsmToken(AsmToken::PipePipe, StringRef(TokStart, 2));
    return AsmToken(AsmToken::Pipe, StringRef(TokStart, 1));
  case '^': return AsmToken(AsmToken::Caret, StringRef(TokStart, 1));
  case '&': 
    if (*CurPtr == '&')
      return ++CurPtr, AsmToken(AsmToken::AmpAmp, StringRef(TokStart, 2));
    return AsmToken(AsmToken::Amp, StringRef(TokStart, 1));
  case '!': 
    if (*CurPtr == '=')
      return ++CurPtr, AsmToken(AsmToken::ExclaimEqual, StringRef(TokStart, 2));
    return AsmToken(AsmToken::Exclaim, StringRef(TokStart, 1));
  case '%': return AsmToken(AsmToken::Percent, StringRef(TokStart, 1));
  case '/': return LexSlash();
  case '#': return AsmToken(AsmToken::Hash, StringRef(TokStart, 1));
  case '"': return LexQuote();
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
    return LexDigit();
  case '<':
    switch (*CurPtr) {
    case '<': return ++CurPtr, AsmToken(AsmToken::LessLess, 
                                        StringRef(TokStart, 2));
    case '=': return ++CurPtr, AsmToken(AsmToken::LessEqual, 
                                        StringRef(TokStart, 2));
    case '>': return ++CurPtr, AsmToken(AsmToken::LessGreater, 
                                        StringRef(TokStart, 2));
    default: return AsmToken(AsmToken::Less, StringRef(TokStart, 1));
    }
  case '>':
    switch (*CurPtr) {
    case '>': return ++CurPtr, AsmToken(AsmToken::GreaterGreater, 
                                        StringRef(TokStart, 2));
    case '=': return ++CurPtr, AsmToken(AsmToken::GreaterEqual, 
                                        StringRef(TokStart, 2));
    default: return AsmToken(AsmToken::Greater, StringRef(TokStart, 1));
    }
      
  // TODO: Quoted identifiers (objc methods etc)
  // local labels: [0-9][:]
  // Forward/backward labels: [0-9][fb]
  // Integers, fp constants, character constants.
  }
}
