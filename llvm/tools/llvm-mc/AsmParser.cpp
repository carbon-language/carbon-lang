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
    Immediate
  } Kind;
  
  union {
    struct {
      unsigned RegNo;
    } Reg;

    struct {
      // FIXME: Should be a general expression.
      int64_t Val;
    } Imm;
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
};

bool AsmParser::ParseX86Operand(X86Operand &Op) {
  switch (Lexer.getKind()) {
  default:
    return TokError("unknown token at start of instruction operand");
  case asmtok::Register:
    // FIXME: Decode reg #.
    Op = X86Operand::CreateReg(0);
    Lexer.Lex(); // Eat register.
    return false;
  case asmtok::Dollar:
    // $42 -> immediate.
    Lexer.Lex();
    // FIXME: Parse an arbitrary expression here, like $(4+5)
    if (Lexer.isNot(asmtok::IntVal))
      return TokError("expected integer constant");
    
    Op = X86Operand::CreateReg(Lexer.getCurIntVal());
    Lexer.Lex(); // Eat register.
    return false;
  case asmtok::Identifier:
    // This is a label, this should be parsed as part of an expression, to
    // handle things like LFOO+4
    Op = X86Operand::CreateImm(0); // FIXME.
    Lexer.Lex(); // Eat identifier.
    return false;
      
  //case asmtok::Star:
  // * %eax
  // * <memaddress>
  // Note that these are both "dereferenced".
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
