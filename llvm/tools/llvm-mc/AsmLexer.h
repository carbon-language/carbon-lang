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
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/System/DataTypes.h"
#include <string>
#include <cassert>

namespace llvm {
class MemoryBuffer;
class SourceMgr;
class SMLoc;
class MCAsmInfo;

/// AsmLexer - Lexer class for assembly files.
class AsmLexer : public MCAsmLexer {
  SourceMgr &SrcMgr;
  const MCAsmInfo &MAI;
  
  const char *CurPtr;
  const MemoryBuffer *CurBuf;
  
  const char *TokStart;

  /// This is the current buffer index we're lexing from as managed by the
  /// SourceMgr object.
  int CurBuffer;
  
  void operator=(const AsmLexer&); // DO NOT IMPLEMENT
  AsmLexer(const AsmLexer&);       // DO NOT IMPLEMENT

protected:
  /// LexToken - Read the next token and return its code.
  virtual AsmToken LexToken();

public:
  AsmLexer(SourceMgr &SrcMgr, const MCAsmInfo &MAI);
  ~AsmLexer();
  
  SMLoc getLoc() const;
  
  StringRef LexUntilEndOfStatement();

  bool isAtStartOfComment(char Char);

  /// EnterIncludeFile - Enter the specified file. This returns true on failure.
  bool EnterIncludeFile(const std::string &Filename);
  
  void PrintMessage(SMLoc Loc, const std::string &Msg, const char *Type) const;
  
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
