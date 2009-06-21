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
#include <cerrno>
#include <cstdlib>
using namespace llvm;

AsmLexer::AsmLexer(SourceMgr &SM) : SrcMgr(SM) {
  CurBuffer = 0;
  CurBuf = SrcMgr.getMemoryBuffer(CurBuffer);
  CurPtr = CurBuf->getBufferStart();
  TokStart = 0;
}

SMLoc AsmLexer::getLoc() const {
  return SMLoc::getFromPointer(TokStart);
}

void AsmLexer::PrintError(const char *Loc, const std::string &Msg) const {
  SrcMgr.PrintError(SMLoc::getFromPointer(Loc), Msg);
}

void AsmLexer::PrintError(SMLoc Loc, const std::string &Msg) const {
  SrcMgr.PrintError(Loc, Msg);
}

/// ReturnError - Set the error to the specified string at the specified
/// location.  This is defined to always return asmtok::Error.
asmtok::TokKind AsmLexer::ReturnError(const char *Loc, const std::string &Msg) {
  PrintError(Loc, Msg);
  return asmtok::Error;
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

/// LexIdentifier: [a-zA-Z_.][a-zA-Z0-9_$.@]*
asmtok::TokKind AsmLexer::LexIdentifier() {
  while (isalnum(*CurPtr) || *CurPtr == '_' || *CurPtr == '$' ||
         *CurPtr == '.' || *CurPtr == '@')
    ++CurPtr;
  CurStrVal.assign(TokStart, CurPtr);   // Include %
  return asmtok::Identifier;
}

/// LexPercent: Register: %[a-zA-Z0-9]+
asmtok::TokKind AsmLexer::LexPercent() {
  if (!isalnum(*CurPtr))
    return ReturnError(TokStart, "invalid register name");
  while (isalnum(*CurPtr))
    ++CurPtr;
  CurStrVal.assign(TokStart, CurPtr);   // Skip %
  return asmtok::Register;
}

/// LexSlash: Slash: /
///           C-Style Comment: /* ... */
asmtok::TokKind AsmLexer::LexSlash() {
  if (*CurPtr != '*')
    return asmtok::Slash;

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

/// LexHash: Comment: #[^\n]*
asmtok::TokKind AsmLexer::LexHash() {
  int CurChar = getNextChar();
  while (CurChar != '\n' && CurChar != '\n' && CurChar != EOF)
    CurChar = getNextChar();
  
  if (CurChar == EOF)
    return asmtok::Eof;
  return asmtok::EndOfStatement;
}


/// LexDigit: First character is [0-9].
///   Local Label: [0-9][:]
///   Forward/Backward Label: [0-9][fb]
///   Binary integer: 0b[01]+
///   Octal integer: 0[0-7]+
///   Hex integer: 0x[0-9a-fA-F]+
///   Decimal integer: [1-9][0-9]*
/// TODO: FP literal.
asmtok::TokKind AsmLexer::LexDigit() {
  if (*CurPtr == ':')
    return ReturnError(TokStart, "FIXME: local label not implemented");
  if (*CurPtr == 'f' || *CurPtr == 'b')
    return ReturnError(TokStart, "FIXME: directional label not implemented");
  
  // Decimal integer: [1-9][0-9]*
  if (CurPtr[-1] != '0') {
    while (isdigit(*CurPtr))
      ++CurPtr;
    CurIntVal = strtoll(TokStart, 0, 10);
    return asmtok::IntVal;
  }
  
  if (*CurPtr == 'b') {
    ++CurPtr;
    const char *NumStart = CurPtr;
    while (CurPtr[0] == '0' || CurPtr[0] == '1')
      ++CurPtr;
    
    // Requires at least one binary digit.
    if (CurPtr == NumStart)
      return ReturnError(CurPtr-2, "Invalid binary number");
    CurIntVal = strtoll(NumStart, 0, 2);
    return asmtok::IntVal;
  }
 
  if (*CurPtr == 'x') {
    ++CurPtr;
    const char *NumStart = CurPtr;
    while (isxdigit(CurPtr[0]))
      ++CurPtr;
    
    // Requires at least one hex digit.
    if (CurPtr == NumStart)
      return ReturnError(CurPtr-2, "Invalid hexadecimal number");
    
    errno = 0;
    CurIntVal = strtoll(NumStart, 0, 16);
    if (errno == EINVAL)
      return ReturnError(CurPtr-2, "Invalid hexadecimal number");
    if (errno == ERANGE) {
      errno = 0;
      CurIntVal = (int64_t)strtoull(NumStart, 0, 16);
      if (errno == EINVAL)
        return ReturnError(CurPtr-2, "Invalid hexadecimal number");
      if (errno == ERANGE)
        return ReturnError(CurPtr-2, "Hexadecimal number out of range");
    }
    return asmtok::IntVal;
  }
  
  // Must be an octal number, it starts with 0.
  while (*CurPtr >= '0' && *CurPtr <= '7')
    ++CurPtr;
  CurIntVal = strtoll(TokStart, 0, 8);
  return asmtok::IntVal;
}

/// LexQuote: String: "..."
asmtok::TokKind AsmLexer::LexQuote() {
  int CurChar = getNextChar();
  // TODO: does gas allow multiline string constants?
  while (CurChar != '"') {
    if (CurChar == '\\') {
      // Allow \", etc.
      CurChar = getNextChar();
    }
    
    if (CurChar == EOF) {
      PrintError(TokStart, "unterminated string constant");
      return asmtok::Eof;
    }

    CurChar = getNextChar();
  }
  
  CurStrVal.assign(TokStart, CurPtr);   // include quotes.
  return asmtok::String;
}


asmtok::TokKind AsmLexer::LexToken() {
  TokStart = CurPtr;
  // This always consumes at least one character.
  int CurChar = getNextChar();
  
  switch (CurChar) {
  default:
    // Handle identifier: [a-zA-Z_.][a-zA-Z0-9_$.@]*
    if (isalpha(CurChar) || CurChar == '_' || CurChar == '.')
      return LexIdentifier();
    
    // Unknown character, emit an error.
    return ReturnError(TokStart, "invalid character in input");
  case EOF: return asmtok::Eof;
  case 0:
  case ' ':
  case '\t':
    // Ignore whitespace.
    return LexToken();
  case '\n': // FALL THROUGH.
  case '\r': // FALL THROUGH.
  case ';': return asmtok::EndOfStatement;
  case ':': return asmtok::Colon;
  case '+': return asmtok::Plus;
  case '-': return asmtok::Minus;
  case '(': return asmtok::LParen;
  case ')': return asmtok::RParen;
  case '*': return asmtok::Star;
  case ',': return asmtok::Comma;
  case '$': return asmtok::Dollar;
  case '%': return LexPercent();
  case '/': return LexSlash();
  case '#': return LexHash();
  case '"': return LexQuote();
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
    return LexDigit();
      
  // TODO: Quoted identifiers (objc methods etc)
  // local labels: [0-9][:]
  // Forward/backward labels: [0-9][fb]
  // Integers, fp constants, character constants.
  }
}