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
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/IdentifierTable.h"
#include <iostream>
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// Visitor Implementation.
//===----------------------------------------------------------------------===//

#define MAKE_VISITOR(CLASS) \
  void CLASS::visit(StmtVisitor *V) { return V->Visit ## CLASS(this); }

MAKE_VISITOR(Expr)
MAKE_VISITOR(DeclRefExpr)
MAKE_VISITOR(IntegerConstant)
MAKE_VISITOR(FloatingConstant)
MAKE_VISITOR(StringExpr)
MAKE_VISITOR(ParenExpr)
MAKE_VISITOR(UnaryOperator)
MAKE_VISITOR(SizeOfAlignOfTypeExpr)
MAKE_VISITOR(ArraySubscriptExpr)
MAKE_VISITOR(CallExpr)
MAKE_VISITOR(MemberExpr)
MAKE_VISITOR(CastExpr)
MAKE_VISITOR(BinaryOperator)
MAKE_VISITOR(ConditionalOperator)

#undef MAKE_VISITOR

//===----------------------------------------------------------------------===//
// Primary Expressions.
//===----------------------------------------------------------------------===//

void DeclRefExpr::dump_impl() const {
  std::cerr << "x";
}

void IntegerConstant::dump_impl() const {
  std::cerr << "1";
}

void FloatingConstant::dump_impl() const {
  std::cerr << "1.0";
}



StringExpr::StringExpr(const char *strData, unsigned byteLength, bool Wide) {
  // OPTIMIZE: could allocate this appended to the StringExpr.
  char *AStrData = new char[byteLength];
  memcpy(AStrData, strData, byteLength);
  StrData = AStrData;
  ByteLength = byteLength;
  isWide = Wide;
}

StringExpr::~StringExpr() {
  delete[] StrData;
}

void StringExpr::dump_impl() const {
  if (isWide) std::cerr << 'L';
  std::cerr << '"' << StrData << '"';
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
  default: assert(0 && "Unknown unary operator");
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
  case SizeOf:  return "sizeof";
  case AlignOf: return "alignof";
  case Extension: return "__extension__";
  }
}

void UnaryOperator::dump_impl() const {
  std::cerr << getOpcodeStr(Opc);
  Input->dump();
}

void SizeOfAlignOfTypeExpr::dump_impl() const {
  std::cerr << (isSizeof ? "sizeof(" : "alignof(");
  // FIXME: print type.
  std::cerr << "ty)";
}

//===----------------------------------------------------------------------===//
// Postfix Operators.
//===----------------------------------------------------------------------===//

void ArraySubscriptExpr::dump_impl() const {
  Base->dump();
  std::cerr << "[";
  Idx->dump();
  std::cerr << "]";
}

CallExpr::CallExpr(Expr *fn, Expr **args, unsigned numargs)
  : Fn(fn), NumArgs(numargs) {
  Args = new Expr*[numargs];
  for (unsigned i = 0; i != numargs; ++i)
    Args[i] = args[i];
}

void CallExpr::dump_impl() const {
  Fn->dump();
  std::cerr << "(";
  for (unsigned i = 0, e = getNumArgs(); i != e; ++i) {
    if (i) std::cerr << ", ";
    getArg(i)->dump();
  }
  std::cerr << ")";
}


void MemberExpr::dump_impl() const {
  Base->dump();
  std::cerr << (isArrow ? "->" : ".");
  
  if (MemberDecl)
    /*TODO: Print MemberDecl*/;
  std::cerr << "member";
}


void CastExpr::dump_impl() const {
  std::cerr << "'('";
  // TODO PRINT TYPE
  std::cerr << "<type>";
  std::cerr << "')'";
  Op->dump();
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
  LHS->dump();
  std::cerr << " " << getOpcodeStr(Opc) << " ";
  RHS->dump();
}

void ConditionalOperator::dump_impl() const {
  Cond->dump();
  std::cerr << " ? ";
  LHS->dump();
  std::cerr << " : ";
  RHS->dump();
}
