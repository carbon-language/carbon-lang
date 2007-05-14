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
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// Primary Expressions.
//===----------------------------------------------------------------------===//

StringLiteral::StringLiteral(const char *strData, unsigned byteLength, 
                             bool Wide, QualType t) : 
  Expr(StringLiteralClass, t) {
  // OPTIMIZE: could allocate this appended to the StringLiteral.
  char *AStrData = new char[byteLength];
  memcpy(AStrData, strData, byteLength);
  StrData = AStrData;
  ByteLength = byteLength;
  IsWide = Wide;
}

StringLiteral::~StringLiteral() {
  delete[] StrData;
}

bool UnaryOperator::isPostfix(Opcode Op) {
  switch (Op) {
  case PostInc:
  case PostDec:
    return true;
  default:
    return false;
  }
}

/// getOpcodeStr - Turn an Opcode enum value into the punctuation char it
/// corresponds to, e.g. "sizeof" or "[pre]++".
const char *UnaryOperator::getOpcodeStr(Opcode Op) {
  switch (Op) {
  default: assert(0 && "Unknown unary operator");
  case PostInc: return "++";
  case PostDec: return "--";
  case PreInc:  return "++";
  case PreDec:  return "--";
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

//===----------------------------------------------------------------------===//
// Postfix Operators.
//===----------------------------------------------------------------------===//

CallExpr::CallExpr(Expr *fn, Expr **args, unsigned numargs, QualType t)
  : Expr(CallExprClass, t), Fn(fn), NumArgs(numargs) {
  Args = new Expr*[numargs];
  for (unsigned i = 0; i != numargs; ++i)
    Args[i] = args[i];
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

/// isLvalue - C99 6.3.2.1: an lvalue is an expression with an object type or an
/// incomplete type other than void. Nonarray expressions that can be lvalues:
///  - name, where name must be a variable
///  - e[i]
///  - (e), where e must be an lvalue
///  - e.name, where e must be an lvalue
///  - e->name
///  - *e, the type of e cannot be a function type
///  - string-constant
///
bool Expr::isLvalue() {
  // first, check the type (C99 6.3.2.1)
  if (!TR->isObjectType())
    return false;
  if (TR->isIncompleteType() && TR->isVoidType())
    return false;
  
  // now, check the expression
  switch (getStmtClass()) {
  case StringLiteralClass: // C99 6.5.1p4
    return true;
  case ArraySubscriptExprClass:
    return true;
  case DeclRefExprClass: // C99 6.5.1p2
    const DeclRefExpr *d = cast<DeclRefExpr>(this);
    return isa<VarDecl>(d->getDecl());
  case MemberExprClass: // C99 6.5.2.3p4
    const MemberExpr *m = cast<MemberExpr>(this);
    if (m->isArrow())
      return true;
    return m->getBase()->isLvalue();
  case UnaryOperatorClass: // C99 6.5.3p4
    const UnaryOperator *u = cast<UnaryOperator>(this);
    return u->getOpcode() == UnaryOperator::Deref;
  case ParenExprClass: // C99 6.5.1p5
    return cast<ParenExpr>(this)->getSubExpr()->isLvalue();
  default: 
    return false;
  }
}

/// isModifiableLvalue - C99 6.3.2.1: an lvalue that does not have array type,
/// does not have an incomplete type, does not have a const-qualified type, and
/// if it is a structure or union, does not have any member (including, 
/// recursively, any member or element of all contained aggregates or unions)
/// with a const-qualified type.
bool Expr::isModifiableLvalue() {
  if (!isLvalue())
    return false;
  
  if (TR.isConstQualified())
    return false;
  if (TR->isArrayType())
    return false;
  if (TR->isIncompleteType())
    return false;
  if (const RecordType *r = dyn_cast<RecordType>(TR.getCanonicalType()))
    return r->isModifiableLvalue();
  return true;    
}

bool Expr::isConstantExpr() const {
  switch (getStmtClass()) {
  case IntegerLiteralClass:
  case FloatingLiteralClass:
  case CharacterLiteralClass:
  case StringLiteralClass:
    return true;
  case DeclRefExprClass:
    return isa<EnumConstantDecl>(cast<DeclRefExpr>(this)->getDecl());
  case UnaryOperatorClass:
    return cast<UnaryOperator>(this)->getSubExpr()->isConstantExpr();
  case BinaryOperatorClass:
    return cast<BinaryOperator>(this)->getLHS()->isConstantExpr() &&
           cast<BinaryOperator>(this)->getRHS()->isConstantExpr();
  case ParenExprClass:
    return cast<ParenExpr>(this)->getSubExpr()->isConstantExpr();
  case CastExprClass:
    return cast<CastExpr>(this)->getSubExpr()->isConstantExpr();
  case SizeOfAlignOfTypeExprClass:
    return cast<SizeOfAlignOfTypeExpr>(this)->getArgumentType()
                                            ->isConstantSizeType();
  default: 
    return false;
  }
}

bool Expr::isIntegerConstantExpr() const {
  switch (getStmtClass()) {
  case IntegerLiteralClass:
  case CharacterLiteralClass:
    return true;
  case DeclRefExprClass:
    return isa<EnumConstantDecl>(cast<DeclRefExpr>(this)->getDecl());
  case UnaryOperatorClass:
    return cast<UnaryOperator>(this)->getSubExpr()->isIntegerConstantExpr();
  case BinaryOperatorClass:
    return cast<BinaryOperator>(this)->getLHS()->isIntegerConstantExpr() &&
           cast<BinaryOperator>(this)->getRHS()->isIntegerConstantExpr();
  case ParenExprClass:
    return cast<ParenExpr>(this)->getSubExpr()->isIntegerConstantExpr();
  case CastExprClass:
    return cast<CastExpr>(this)->getSubExpr()->isIntegerConstantExpr();
  case SizeOfAlignOfTypeExprClass:
    return cast<SizeOfAlignOfTypeExpr>(this)->getArgumentType()
                                            ->isConstantSizeType();
  default: 
    return false;
  }
}

bool Expr::isNullPointerConstant() const {
  const IntegerLiteral *constant = dyn_cast<IntegerLiteral>(this);
  if (!constant || constant->getValue() != 0)
    return false;
  return true;
}
