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
#include "llvm/MC/MCStreamer.h"

namespace llvm {
class AsmExpr;
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

  void Warning(SMLoc L, const char *Msg);
  bool Error(SMLoc L, const char *Msg);
  bool TokError(const char *Msg);
  
  void EatToEndOfStatement();
  
  bool ParseAssignment(const char *Name, bool IsDotSet);

  /// ParseExpression - Parse a general assembly expression.
  ///
  /// @param Res - The resulting expression. The pointer value is null on error.
  /// @result - False on success.
  bool ParseExpression(AsmExpr *&Res);
  
  /// ParseAbsoluteExpr - Parse an expression which must evaluate to an absolute
  /// value.
  ///
  /// @param Res - The value of the absolute expression. The result is undefined
  /// on error.
  /// @result - False on success.
  bool ParseAbsoluteExpression(int64_t &Res);

  bool ParsePrimaryExpr(AsmExpr *&Res);
  bool ParseBinOpRHS(unsigned Precedence, AsmExpr *&Res);
  bool ParseParenExpr(AsmExpr *&Res);
  
  // X86 specific.
  bool ParseX86InstOperands(MCInst &Inst);
  bool ParseX86Operand(X86Operand &Op);
  bool ParseX86MemOperand(X86Operand &Op);
  
  // Directive Parsing.
  bool ParseDirectiveDarwinSection(); // Darwin specific ".section".
  bool ParseDirectiveSectionSwitch(const char *Section,
                                   const char *Directives = 0);
  bool ParseDirectiveAscii(bool ZeroTerminated); // ".ascii", ".asciiz"
  bool ParseDirectiveValue(unsigned Size); // ".byte", ".long", ...
  bool ParseDirectiveFill(); // ".fill"
  bool ParseDirectiveSpace(); // ".space"
  bool ParseDirectiveSet(); // ".set"
  bool ParseDirectiveOrg(); // ".org"
  // ".align{,32}", ".p2align{,w,l}"
  bool ParseDirectiveAlign(bool IsPow2, unsigned ValueSize);

  /// ParseDirectiveSymbolAttribute - Parse a directive like ".globl" which
  /// accepts a single symbol (which should be a label or an external).
  bool ParseDirectiveSymbolAttribute(MCStreamer::SymbolAttr Attr);
  
};

} // end namespace llvm

#endif
