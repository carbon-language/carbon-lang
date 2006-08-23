//===--- Expr.cpp - Expression AST Node Implementation --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Expr class and subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Expr.h"
#include <iostream>
using namespace llvm;
using namespace clang;

void Expr::dump() const {
  if (this == 0)
    std::cerr << "<null expr>";
  else
    dump_impl();
}


void IntegerConstant::dump_impl() const {
  std::cerr << "1";
}

void FloatingConstant::dump_impl() const {
  std::cerr << "1.0";
}

void ParenExpr::dump_impl() const {
  std::cerr << "'('";
  Val->dump();
  std::cerr << "')'";
}

/// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
/// corresponds to, e.g. "sizeof" or "[pre]++".
const char *UnaryOperator::getOpcodeStr(Opcode Op) {
  switch (Op) {
  default: assert(0 && "Unknown binary operator");
  case PostInc: return "[post]++";
  case PostDec: return "[post]--";
  case PreInc:  return "[pre]++";
  case PreDec:  return "[pre]--";
  case AddrOf:  return "&";
  case Deref:   return "*";
  case Plus:    return "+";
  case Minus:   return "-";
  case Not:     return "~";
  case LNot:    return "!";
  case Real:    return "__real";
  case Imag:    return "__imag";
  }
}

void UnaryOperator::dump_impl() const {
  std::cerr << "(" << getOpcodeStr(Opc);
  Input->dump();
  std::cerr << ")";
}

/// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
/// corresponds to, e.g. "<<=".
const char *BinaryOperator::getOpcodeStr(Opcode Op) {
  switch (Op) {
  default: assert(0 && "Unknown binary operator");
  case Mul:       return "*";
  case Div:       return "/";
  case Rem:       return "%";
  case Add:       return "+";
  case Sub:       return "-";
  case Shl:       return "<<";
  case Shr:       return ">>";
  case LT:        return "<";
  case GT:        return ">";
  case LE:        return "<=";
  case GE:        return ">=";
  case EQ:        return "==";
  case NE:        return "!=";
  case And:       return "&";
  case Xor:       return "^";
  case Or:        return "|";
  case LAnd:      return "&&";
  case LOr:       return "||";
  case Assign:    return "=";
  case MulAssign: return "*=";
  case DivAssign: return "/=";
  case RemAssign: return "%=";
  case AddAssign: return "+=";
  case SubAssign: return "-=";
  case ShlAssign: return "<<=";
  case ShrAssign: return ">>=";
  case AndAssign: return "&=";
  case XorAssign: return "^=";
  case OrAssign:  return "|=";
  case Comma:     return ",";
  }
}

void BinaryOperator::dump_impl() const {
  std::cerr << "(";
  LHS->dump();
  std::cerr << " " << getOpcodeStr(Opc) << " ";
  RHS->dump();
  std::cerr << ")";
}

void ConditionalOperator::dump_impl() const {
  std::cerr << "(";
  Cond->dump();
  std::cerr << " ? ";
  LHS->dump();
  std::cerr << " : ";
  RHS->dump();
  std::cerr << ")";
}
