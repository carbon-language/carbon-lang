//===- AsmParser.cpp - Parser for Assembly Files --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the parser for assembly files.
//
//===----------------------------------------------------------------------===//

#include "AsmParser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

bool AsmParser::Error(SMLoc L, const char *Msg) {
  Lexer.PrintMessage(L, Msg);
  return true;
}

bool AsmParser::TokError(const char *Msg) {
  Lexer.PrintMessage(Lexer.getLoc(), Msg);
  return true;
}

bool AsmParser::Run() {
  // Prime the lexer.
  Lexer.Lex();
  
  while (Lexer.isNot(asmtok::Eof))
    if (ParseStatement())
      return true;
  
  return false;
}

/// EatToEndOfStatement - Throw away the rest of the line for testing purposes.
void AsmParser::EatToEndOfStatement() {
  while (Lexer.isNot(asmtok::EndOfStatement) &&
         Lexer.isNot(asmtok::Eof))
    Lexer.Lex();
  
  // Eat EOL.
  if (Lexer.is(asmtok::EndOfStatement))
    Lexer.Lex();
}


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
};

bool AsmParser::ParseX86Operand(X86Operand &Op) {
  switch (Lexer.getKind()) {
  default:
    return ParseX86MemOperand(Op);
  case asmtok::Register:
    // FIXME: Decode reg #.
    // FIXME: if a segment register, this could either be just the seg reg, or
    // the start of a memory operand.
    Op = X86Operand::CreateReg(0);
    Lexer.Lex(); // Eat register.
    return false;
  case asmtok::Dollar: {
    // $42 -> immediate.
    Lexer.Lex();
    int64_t Val;
    if (ParseExpression(Val))
      return TokError("expected integer constant");
    Op = X86Operand::CreateReg(Val);
    return false;
  }
      
  //case asmtok::Star:
  // * %eax
  // * <memaddress>
  // Note that these are both "dereferenced".
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
    if (ParseExpression(Disp)) return true;
    
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
      // FIXME: Call ParseParenExpression with the leading ( consumed.
      return TokError("FIXME: Paren expr not implemented yet!");
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


/// ParseExpression - Parse an expression and return it.
/// FIXME: This should handle real expressions, we do something trivial for now.
bool AsmParser::ParseExpression(int64_t &Res) {
  switch (Lexer.getKind()) {
  default:
    return TokError("unknown token in expression");
  case asmtok::Identifier:
    // This is a label, this should be parsed as part of an expression, to
    // handle things like LFOO+4
    Res = 0; // FIXME.
    Lexer.Lex(); // Eat identifier.
    return false;
  case asmtok::IntVal:
    Res = Lexer.getCurIntVal();
    Lexer.Lex(); // Eat identifier.
    return false;
  }
}
  
  
  
  
/// ParseStatement:
///   ::= EndOfStatement
///   ::= Label* Directive ...Operands... EndOfStatement
///   ::= Label* Identifier OperandList* EndOfStatement
bool AsmParser::ParseStatement() {
  switch (Lexer.getKind()) {
  default:
    return TokError("unexpected token at start of statement");
  case asmtok::EndOfStatement:
    Lexer.Lex();
    return false;
  case asmtok::Identifier:
    break;
  // TODO: Recurse on local labels etc.
  }
  
  // If we have an identifier, handle it as the key symbol.
  SMLoc IDLoc = Lexer.getLoc();
  std::string IDVal = Lexer.getCurStrVal();
  
  // Consume the identifier, see what is after it.
  if (Lexer.Lex() == asmtok::Colon) {
    // identifier ':'   -> Label.
    Lexer.Lex();
    return ParseStatement();
  }
  
  // Otherwise, we have a normal instruction or directive.  
  if (IDVal[0] == '.') {
    Lexer.PrintMessage(IDLoc, "warning: ignoring directive for now");
    EatToEndOfStatement();
    return false;
  }

  // If it's an instruction, parse an operand list.
  std::vector<X86Operand> Operands;
  
  // Read the first operand, if present.  Note that we require a newline at the
  // end of file, so we don't have to worry about Eof here.
  if (Lexer.isNot(asmtok::EndOfStatement)) {
    X86Operand Op;
    if (ParseX86Operand(Op))
      return true;
    Operands.push_back(Op);
  }

  while (Lexer.is(asmtok::Comma)) {
    Lexer.Lex();  // Eat the comma.
    
    // Parse and remember the operand.
    X86Operand Op;
    if (ParseX86Operand(Op))
      return true;
    Operands.push_back(Op);
  }
  
  if (Lexer.isNot(asmtok::EndOfStatement))
    return TokError("unexpected token in operand list");

  // Eat the end of statement marker.
  Lexer.Lex();
  
  // Instruction is good, process it.
  outs() << "Found instruction: " << IDVal << " with " << Operands.size()
         << " operands.\n";
  
  // Skip to end of line for now.
  return false;
}
