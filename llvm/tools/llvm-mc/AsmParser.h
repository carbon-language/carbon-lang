//===- AsmParser.h - Parser for Assembly Files ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class declares the parser for assembly files.
//
//===----------------------------------------------------------------------===//

#ifndef ASMPARSER_H
#define ASMPARSER_H

#include "AsmLexer.h"

namespace llvm {
  
class AsmParser {
  AsmLexer Lexer;
  struct X86Operand;
  
public:
  AsmParser(SourceMgr &SM) : Lexer(SM) {}
  ~AsmParser() {}
  
  bool Run();
  
private:
  bool ParseStatement();
  
  bool Error(SMLoc L, const char *Msg);
  bool TokError(const char *Msg);
  
  void EatToEndOfStatement();
  
  bool ParseX86Operand(X86Operand &Op);
  bool ParseX86MemOperand(X86Operand &Op);
  bool ParseExpression(int64_t &Res);
};

} // end namespace llvm

#endif
