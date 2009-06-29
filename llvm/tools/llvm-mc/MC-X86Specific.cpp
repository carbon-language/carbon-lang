//===- MC-X86Specific.cpp - X86-Specific code for MC ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements X86-specific parsing, encoding and decoding stuff for
// MC.
//
//===----------------------------------------------------------------------===//

#include "AsmParser.h"
#include "llvm/MC/MCInst.h"
using namespace llvm;

/// X86Operand - Instances of this class represent one X86 machine instruction.
struct AsmParser::X86Operand {
  enum {
    Register,
    Immediate,
    Memory
  } Kind;
  
  union {
    struct {
      unsigned RegNo;
    } Reg;

    struct {
      // FIXME: Should be a general expression.
      int64_t Val;
    } Imm;
    
    struct {
      unsigned SegReg;
      int64_t Disp;     // FIXME: Should be a general expression.
      unsigned BaseReg;
      unsigned Scale;
      unsigned ScaleReg;
    } Mem;
  };
  
  static X86Operand CreateReg(unsigned RegNo) {
    X86Operand Res;
    Res.Kind = Register;
    Res.Reg.RegNo = RegNo;
    return Res;
  }
  static X86Operand CreateImm(int64_t Val) {
    X86Operand Res;
    Res.Kind = Immediate;
    Res.Imm.Val = Val;
    return Res;
  }
  static X86Operand CreateMem(unsigned SegReg, int64_t Disp, unsigned BaseReg,
                              unsigned Scale, unsigned ScaleReg) {
    X86Operand Res;
    Res.Kind = Memory;
    Res.Mem.SegReg   = SegReg;
    Res.Mem.Disp     = Disp;
    Res.Mem.BaseReg  = BaseReg;
    Res.Mem.Scale    = Scale;
    Res.Mem.ScaleReg = ScaleReg;
    return Res;
  }
  
  void AddToMCInst(MCInst &I) {
    // FIXME: Add in x86 order here.
  }
};

bool AsmParser::ParseX86Operand(X86Operand &Op) {
  switch (Lexer.getKind()) {
  default:
    return ParseX86MemOperand(Op);
  case asmtok::Register:
    // FIXME: Decode reg #.
    // FIXME: if a segment register, this could either be just the seg reg, or
    // the start of a memory operand.
    Op = X86Operand::CreateReg(123);
    Lexer.Lex(); // Eat register.
    return false;
  case asmtok::Dollar: {
    // $42 -> immediate.
    Lexer.Lex();
    int64_t Val;
    if (ParseAbsoluteExpression(Val))
      return TokError("expected integer constant");
    Op = X86Operand::CreateReg(Val);
    return false;
  case asmtok::Star:
    Lexer.Lex(); // Eat the star.
    
    if (Lexer.is(asmtok::Register)) {
      Op = X86Operand::CreateReg(123);
      Lexer.Lex(); // Eat register.
    } else if (ParseX86MemOperand(Op))
      return true;

    // FIXME: Note that these are 'dereferenced' so that clients know the '*' is
    // there.
    return false;
  }
  }
}

/// ParseX86MemOperand: segment: disp(basereg, indexreg, scale)
bool AsmParser::ParseX86MemOperand(X86Operand &Op) {
  // FIXME: If SegReg ':'  (e.g. %gs:), eat and remember.
  unsigned SegReg = 0;
  
  // We have to disambiguate a parenthesized expression "(4+5)" from the start
  // of a memory operand with a missing displacement "(%ebx)" or "(,%eax)".  The
  // only way to do this without lookahead is to eat the ( and see what is after
  // it.
  int64_t Disp = 0;
  if (Lexer.isNot(asmtok::LParen)) {
    if (ParseAbsoluteExpression(Disp)) return true;
    
    // After parsing the base expression we could either have a parenthesized
    // memory address or not.  If not, return now.  If so, eat the (.
    if (Lexer.isNot(asmtok::LParen)) {
      Op = X86Operand::CreateMem(SegReg, Disp, 0, 0, 0);
      return false;
    }
    
    // Eat the '('.
    Lexer.Lex();
  } else {
    // Okay, we have a '('.  We don't know if this is an expression or not, but
    // so we have to eat the ( to see beyond it.
    Lexer.Lex(); // Eat the '('.
    
    if (Lexer.is(asmtok::Register) || Lexer.is(asmtok::Comma)) {
      // Nothing to do here, fall into the code below with the '(' part of the
      // memory operand consumed.
    } else {
      // It must be an parenthesized expression, parse it now.
      if (ParseAbsoluteExpression(Disp))
        return true;
      
      // After parsing the base expression we could either have a parenthesized
      // memory address or not.  If not, return now.  If so, eat the (.
      if (Lexer.isNot(asmtok::LParen)) {
        Op = X86Operand::CreateMem(SegReg, Disp, 0, 0, 0);
        return false;
      }
      
      // Eat the '('.
      Lexer.Lex();
    }
  }
  
  // If we reached here, then we just ate the ( of the memory operand.  Process
  // the rest of the memory operand.
  unsigned BaseReg = 0, ScaleReg = 0, Scale = 0;
  
  if (Lexer.is(asmtok::Register)) {
    BaseReg = 123; // FIXME: decode reg #
    Lexer.Lex();  // eat the register.
  }
  
  if (Lexer.is(asmtok::Comma)) {
    Lexer.Lex(); // eat the comma.
    
    if (Lexer.is(asmtok::Register)) {
      ScaleReg = 123; // FIXME: decode reg #
      Lexer.Lex();  // eat the register.
      Scale = 1;      // If not specified, the scale defaults to 1.
    }
    
    if (Lexer.is(asmtok::Comma)) {
      Lexer.Lex(); // eat the comma.

      // If present, get and validate scale amount.
      if (Lexer.is(asmtok::IntVal)) {
        int64_t ScaleVal = Lexer.getCurIntVal();
        if (ScaleVal != 1 && ScaleVal != 2 && ScaleVal != 4 && ScaleVal != 8)
          return TokError("scale factor in address must be 1, 2, 4 or 8");
        Lexer.Lex();  // eat the scale.
        Scale = (unsigned)ScaleVal;
      }
    }
  }
  
  // Ok, we've eaten the memory operand, verify we have a ')' and eat it too.
  if (Lexer.isNot(asmtok::RParen))
    return TokError("unexpected token in memory operand");
  Lexer.Lex(); // Eat the ')'.
  
  Op = X86Operand::CreateMem(SegReg, Disp, BaseReg, Scale, ScaleReg);
  return false;
}

/// ParseX86InstOperands - Parse the operands of an X86 instruction and return
/// them as the operands of an MCInst.
bool AsmParser::ParseX86InstOperands(MCInst &Inst) {
  // If no operands are present, just return.
  if (Lexer.is(asmtok::EndOfStatement))
    return false;

  // Read the first operand.
  X86Operand Op;
  if (ParseX86Operand(Op))
    return true;
  Op.AddToMCInst(Inst);
  
  while (Lexer.is(asmtok::Comma)) {
    Lexer.Lex();  // Eat the comma.
    
    // Parse and remember the operand.
    Op = X86Operand();
    if (ParseX86Operand(Op))
      return true;
    Op.AddToMCInst(Inst);
  }
  return false;
}
