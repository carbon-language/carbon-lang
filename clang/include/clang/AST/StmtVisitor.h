//===--- StmtVisitor.h - Visitor for Stmt subclasses ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the StmtVisitor interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTVISITOR_H
#define LLVM_CLANG_AST_STMTVISITOR_H

#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"

namespace clang {
  
#define DISPATCH(NAME, CLASS) \
  return static_cast<ImplClass*>(this)->Visit ## NAME(static_cast<CLASS*>(S))
  
/// StmtVisitor - This class implements a simple visitor for Stmt subclasses.
/// Since Expr derives from Stmt, this also includes support for visiting Exprs.
template<typename ImplClass, typename RetTy=void>
class StmtVisitor {
public:
  RetTy Visit(Stmt *S) {
    
    // If we have a binary expr, dispatch to the subcode of the binop.  A smart
    // optimizer (e.g. LLVM) will fold this comparison into the switch stmt
    // below.
    if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(S)) {
      switch (BinOp->getOpcode()) {
      default: assert(0 && "Unknown binary operator!");
      case BinaryOperator::PtrMemD:   DISPATCH(BinPtrMemD,   BinaryOperator);
      case BinaryOperator::PtrMemI:   DISPATCH(BinPtrMemI,   BinaryOperator);
      case BinaryOperator::Mul:       DISPATCH(BinMul,       BinaryOperator);
      case BinaryOperator::Div:       DISPATCH(BinDiv,       BinaryOperator);
      case BinaryOperator::Rem:       DISPATCH(BinRem,       BinaryOperator);
      case BinaryOperator::Add:       DISPATCH(BinAdd,       BinaryOperator);
      case BinaryOperator::Sub:       DISPATCH(BinSub,       BinaryOperator);
      case BinaryOperator::Shl:       DISPATCH(BinShl,       BinaryOperator);
      case BinaryOperator::Shr:       DISPATCH(BinShr,       BinaryOperator);

      case BinaryOperator::LT:        DISPATCH(BinLT,        BinaryOperator);
      case BinaryOperator::GT:        DISPATCH(BinGT,        BinaryOperator);
      case BinaryOperator::LE:        DISPATCH(BinLE,        BinaryOperator);
      case BinaryOperator::GE:        DISPATCH(BinGE,        BinaryOperator);
      case BinaryOperator::EQ:        DISPATCH(BinEQ,        BinaryOperator);
      case BinaryOperator::NE:        DISPATCH(BinNE,        BinaryOperator);
        
      case BinaryOperator::And:       DISPATCH(BinAnd,       BinaryOperator);
      case BinaryOperator::Xor:       DISPATCH(BinXor,       BinaryOperator);
      case BinaryOperator::Or :       DISPATCH(BinOr,        BinaryOperator);
      case BinaryOperator::LAnd:      DISPATCH(BinLAnd,      BinaryOperator);
      case BinaryOperator::LOr :      DISPATCH(BinLOr,       BinaryOperator);
      case BinaryOperator::Assign:    DISPATCH(BinAssign,    BinaryOperator);
      case BinaryOperator::MulAssign:
        DISPATCH(BinMulAssign, CompoundAssignOperator);
      case BinaryOperator::DivAssign:
        DISPATCH(BinDivAssign, CompoundAssignOperator);
      case BinaryOperator::RemAssign:
        DISPATCH(BinRemAssign, CompoundAssignOperator);
      case BinaryOperator::AddAssign:
        DISPATCH(BinAddAssign, CompoundAssignOperator);
      case BinaryOperator::SubAssign:
        DISPATCH(BinSubAssign, CompoundAssignOperator);
      case BinaryOperator::ShlAssign:
        DISPATCH(BinShlAssign, CompoundAssignOperator);
      case BinaryOperator::ShrAssign:
        DISPATCH(BinShrAssign, CompoundAssignOperator);
      case BinaryOperator::AndAssign:
        DISPATCH(BinAndAssign, CompoundAssignOperator);
      case BinaryOperator::OrAssign:
        DISPATCH(BinOrAssign,  CompoundAssignOperator);
      case BinaryOperator::XorAssign:
        DISPATCH(BinXorAssign, CompoundAssignOperator);
      case BinaryOperator::Comma:     DISPATCH(BinComma,     BinaryOperator);
      }
    } else if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(S)) {
      switch (UnOp->getOpcode()) {
      default: assert(0 && "Unknown unary operator!");
      case UnaryOperator::PostInc:      DISPATCH(UnaryPostInc,   UnaryOperator);
      case UnaryOperator::PostDec:      DISPATCH(UnaryPostDec,   UnaryOperator);
      case UnaryOperator::PreInc:       DISPATCH(UnaryPreInc,    UnaryOperator);
      case UnaryOperator::PreDec:       DISPATCH(UnaryPreDec,    UnaryOperator);
      case UnaryOperator::AddrOf:       DISPATCH(UnaryAddrOf,    UnaryOperator);
      case UnaryOperator::Deref:        DISPATCH(UnaryDeref,     UnaryOperator);
      case UnaryOperator::Plus:         DISPATCH(UnaryPlus,      UnaryOperator);
      case UnaryOperator::Minus:        DISPATCH(UnaryMinus,     UnaryOperator);
      case UnaryOperator::Not:          DISPATCH(UnaryNot,       UnaryOperator);
      case UnaryOperator::LNot:         DISPATCH(UnaryLNot,      UnaryOperator);
      case UnaryOperator::Real:         DISPATCH(UnaryReal,      UnaryOperator);
      case UnaryOperator::Imag:         DISPATCH(UnaryImag,      UnaryOperator);
      case UnaryOperator::Extension:    DISPATCH(UnaryExtension, UnaryOperator);
      case UnaryOperator::OffsetOf:     DISPATCH(UnaryOffsetOf,  UnaryOperator);
      }
    }
    
    // Top switch stmt: dispatch to VisitFooStmt for each FooStmt.
    switch (S->getStmtClass()) {
    default: assert(0 && "Unknown stmt kind!");
#define STMT(CLASS, PARENT)                              \
    case Stmt::CLASS ## Class: DISPATCH(CLASS, CLASS);
#include "clang/AST/StmtNodes.def"
    }
  }
  
  // If the implementation chooses not to implement a certain visit method, fall
  // back on VisitExpr or whatever else is the superclass.
#define STMT(CLASS, PARENT)                                   \
  RetTy Visit ## CLASS(CLASS *S) { DISPATCH(PARENT, PARENT); }
#include "clang/AST/StmtNodes.def"

  // If the implementation doesn't implement binary operator methods, fall back
  // on VisitBinaryOperator.
#define BINOP_FALLBACK(NAME) \
  RetTy VisitBin ## NAME(BinaryOperator *S) { \
    DISPATCH(BinaryOperator, BinaryOperator); \
  }
  BINOP_FALLBACK(PtrMemD)                    BINOP_FALLBACK(PtrMemI)
  BINOP_FALLBACK(Mul)   BINOP_FALLBACK(Div)  BINOP_FALLBACK(Rem)
  BINOP_FALLBACK(Add)   BINOP_FALLBACK(Sub)  BINOP_FALLBACK(Shl)
  BINOP_FALLBACK(Shr)
  
  BINOP_FALLBACK(LT)    BINOP_FALLBACK(GT)   BINOP_FALLBACK(LE)
  BINOP_FALLBACK(GE)    BINOP_FALLBACK(EQ)   BINOP_FALLBACK(NE)
  BINOP_FALLBACK(And)   BINOP_FALLBACK(Xor)  BINOP_FALLBACK(Or)
  BINOP_FALLBACK(LAnd)  BINOP_FALLBACK(LOr)

  BINOP_FALLBACK(Assign)
  BINOP_FALLBACK(Comma)
#undef BINOP_FALLBACK

  // If the implementation doesn't implement compound assignment operator
  // methods, fall back on VisitCompoundAssignOperator.
#define CAO_FALLBACK(NAME) \
  RetTy VisitBin ## NAME(CompoundAssignOperator *S) { \
    DISPATCH(CompoundAssignOperator, CompoundAssignOperator); \
  }
  CAO_FALLBACK(MulAssign) CAO_FALLBACK(DivAssign) CAO_FALLBACK(RemAssign)
  CAO_FALLBACK(AddAssign) CAO_FALLBACK(SubAssign) CAO_FALLBACK(ShlAssign)
  CAO_FALLBACK(ShrAssign) CAO_FALLBACK(AndAssign) CAO_FALLBACK(OrAssign)
  CAO_FALLBACK(XorAssign)
#undef CAO_FALLBACK
  
  // If the implementation doesn't implement unary operator methods, fall back
  // on VisitUnaryOperator.
#define UNARYOP_FALLBACK(NAME) \
  RetTy VisitUnary ## NAME(UnaryOperator *S) { \
    DISPATCH(UnaryOperator, UnaryOperator);    \
  }
  UNARYOP_FALLBACK(PostInc)   UNARYOP_FALLBACK(PostDec)
  UNARYOP_FALLBACK(PreInc)    UNARYOP_FALLBACK(PreDec)
  UNARYOP_FALLBACK(AddrOf)    UNARYOP_FALLBACK(Deref)
  
  UNARYOP_FALLBACK(Plus)      UNARYOP_FALLBACK(Minus)
  UNARYOP_FALLBACK(Not)       UNARYOP_FALLBACK(LNot)
  UNARYOP_FALLBACK(Real)      UNARYOP_FALLBACK(Imag)
  UNARYOP_FALLBACK(Extension) UNARYOP_FALLBACK(OffsetOf)
#undef UNARYOP_FALLBACK
  
  // Base case, ignore it. :)
  RetTy VisitStmt(Stmt *Node) { return RetTy(); }
};

#undef DISPATCH

}  // end namespace clang

#endif
