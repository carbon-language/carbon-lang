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
class MCContext;
class MCInst;
class MCStreamer;
  
class AsmParser {
  AsmLexer Lexer;
  MCContext &Ctx;
  MCStreamer &Out;
  
  struct X86Operand;
  
public:
  AsmParser(SourceMgr &SM, MCContext &ctx, MCStreamer &OutStr)
    : Lexer(SM), Ctx(ctx), Out(OutStr) {}
  ~AsmParser() {}
  
  bool Run();
  
private:
  bool ParseStatement();
  
  bool Error(SMLoc L, const char *Msg);
  bool TokError(const char *Msg);
  
  void EatToEndOfStatement();
  
  bool ParseExpression(int64_t &Res);
  bool ParsePrimaryExpr(int64_t &Res);
  bool ParseBinOpRHS(unsigned Precedence, int64_t &Res);
  bool ParseParenExpr(int64_t &Res);
  
  // X86 specific.
  bool ParseX86InstOperands(MCInst &Inst);
  bool ParseX86Operand(X86Operand &Op);
  bool ParseX86MemOperand(X86Operand &Op);
  
  // Directive Parsing.
  bool ParseDirectiveSection();
  
};

} // end namespace llvm

#endif
