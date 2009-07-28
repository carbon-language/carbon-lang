//===- AsmLexer.h - Lexer for Assembly Files --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class declares the lexer for assembly files.
//
//===----------------------------------------------------------------------===//

#ifndef ASMLEXER_H
#define ASMLEXER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmLexer.h"
#include "llvm/Support/DataTypes.h"
#include <string>
#include <cassert>

namespace llvm {
class MemoryBuffer;
class SourceMgr;
class SMLoc;

/// AsmToken - Target independent representation for an assembler token.
struct AsmToken {
  enum TokenKind {
    // Markers
    Eof, Error,

    // String values.
    Identifier,
    Register,
    String,
    
    // Integer values.
    Integer,
    
    // No-value.
    EndOfStatement,
    Colon,
    Plus, Minus, Tilde,
    Slash,    // '/'
    LParen, RParen,
    Star, Comma, Dollar, Equal, EqualEqual,
    
    Pipe, PipePipe, Caret, 
    Amp, AmpAmp, Exclaim, ExclaimEqual, Percent, 
    Less, LessEqual, LessLess, LessGreater,
    Greater, GreaterEqual, GreaterGreater
  };

  TokenKind Kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef Str;

  int64_t IntVal;

public:
  AsmToken() {}
  AsmToken(TokenKind _Kind, const StringRef &_Str, int64_t _IntVal = 0)
    : Kind(_Kind), Str(_Str), IntVal(_IntVal) {}

  TokenKind getKind() const { return Kind; }
  bool is(TokenKind K) const { return Kind == K; }
  bool isNot(TokenKind K) const { return Kind != K; }

  SMLoc getLoc() const;

  StringRef getString() const { return Str; }

  // FIXME: Don't compute this in advance, it makes every token larger, and is
  // also not generally what we want (it is nicer for recovery etc. to lex 123br
  // as a single token, then diagnose as an invalid number).
  int64_t getIntVal() const { 
    assert(Kind == Integer && "This token isn't an integer");
    return IntVal; 
  }
};

/// AsmLexer - Lexer class for assembly files.
class AsmLexer : public MCAsmLexer {
  SourceMgr &SrcMgr;
  
  const char *CurPtr;
  const MemoryBuffer *CurBuf;
  
  const char *TokStart;

  /// The current token.
  AsmToken CurTok;
  
  /// This is the current buffer index we're lexing from as managed by the
  /// SourceMgr object.
  int CurBuffer;
  
  void operator=(const AsmLexer&); // DO NOT IMPLEMENT
  AsmLexer(const AsmLexer&);       // DO NOT IMPLEMENT
public:
  AsmLexer(SourceMgr &SrcMgr);
  ~AsmLexer();
  
  AsmToken::TokenKind Lex() {
    return CurTok = LexToken(), getKind();
  }
  
  AsmToken::TokenKind getKind() const { return CurTok.getKind(); }
  bool is(AsmToken::TokenKind K) const { return CurTok.is(K); }
  bool isNot(AsmToken::TokenKind K) const { return CurTok.isNot(K); }

  /// getCurStrVal - Get the string for the current token, this includes all
  /// characters (for example, the quotes on strings) in the token.
  ///
  /// The returned StringRef points into the source manager's memory buffer, and
  /// is safe to store across calls to Lex().
  StringRef getCurStrVal() const {
    return CurTok.getString();
  }
  int64_t getCurIntVal() const {
    return CurTok.getIntVal();
  }
  
  SMLoc getLoc() const;

  const AsmToken &getTok() const;
  
  /// EnterIncludeFile - Enter the specified file. This returns true on failure.
  bool EnterIncludeFile(const std::string &Filename);
  
  void PrintMessage(SMLoc Loc, const std::string &Msg, const char *Type) const;
  
private:
  int getNextChar();
  AsmToken ReturnError(const char *Loc, const std::string &Msg);

  /// LexToken - Read the next token and return its code.
  AsmToken LexToken();
  AsmToken LexIdentifier();
  AsmToken LexPercent();
  AsmToken LexSlash();
  AsmToken LexLineComment();
  AsmToken LexDigit();
  AsmToken LexQuote();
};
  
} // end namespace llvm

#endif
