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
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/System/DataTypes.h"
#include <string>
#include <cassert>

namespace llvm {
class MemoryBuffer;
class SMLoc;
class MCAsmInfo;

/// AsmLexer - Lexer class for assembly files.
class AsmLexer : public MCAsmLexer {
  const MCAsmInfo &MAI;
  
  const char *CurPtr;
  const MemoryBuffer *CurBuf;
  
  void operator=(const AsmLexer&); // DO NOT IMPLEMENT
  AsmLexer(const AsmLexer&);       // DO NOT IMPLEMENT

protected:
  /// LexToken - Read the next token and return its code.
  virtual AsmToken LexToken();

public:
  AsmLexer(const MCAsmInfo &MAI);
  ~AsmLexer();
  
  void setBuffer(const MemoryBuffer *buf, const char *ptr = NULL);
  
  virtual StringRef LexUntilEndOfStatement();

  bool isAtStartOfComment(char Char);
  
  const MCAsmInfo &getMAI() const { return MAI; }

private:
  int getNextChar();
  AsmToken ReturnError(const char *Loc, const std::string &Msg);

  AsmToken LexIdentifier();
  AsmToken LexSlash();
  AsmToken LexLineComment();
  AsmToken LexDigit();
  AsmToken LexQuote();
};
  
} // end namespace llvm

#endif
