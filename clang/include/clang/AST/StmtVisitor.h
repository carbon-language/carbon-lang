//===--- StmtVisitor.h - Visitor for Stmt subclasses ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the StmtVisitor interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTVISITOR_H
#define LLVM_CLANG_AST_STMTVISITOR_H

#include "clang/AST/ExprCXX.h"

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
      case BinaryOperator::Mul:       DISPATCH(BinMul,       BinaryOperator);
      case BinaryOperator::Div:       DISPATCH(BinDiv,       BinaryOperator);
      case BinaryOperator::Rem:       DISPATCH(BinRem,       BinaryOperator);
      case BinaryOperator::Add:       DISPATCH(BinAdd,       BinaryOperator);
      case BinaryOperator::Sub:       DISPATCH(BinSub,       BinaryOperator);
      case BinaryOperator::Shl:       DISPATCH(BinShl,       BinaryOperator);
      case BinaryOperator::Shr:       DISPATCH(BinShr,       BinaryOperator);
      case BinaryOperator::And:       DISPATCH(BinAnd,       BinaryOperator);
      case BinaryOperator::Xor:       DISPATCH(BinXor,       BinaryOperator);
      case BinaryOperator::Or :       DISPATCH(BinOr,        BinaryOperator);
      case BinaryOperator::Assign:    DISPATCH(BinAssign,    BinaryOperator);
      case BinaryOperator::MulAssign: DISPATCH(BinMulAssign, BinaryOperator);
      case BinaryOperator::DivAssign: DISPATCH(BinDivAssign, BinaryOperator);
      case BinaryOperator::RemAssign: DISPATCH(BinRemAssign, BinaryOperator);
      case BinaryOperator::AddAssign: DISPATCH(BinAddAssign, BinaryOperator);
      case BinaryOperator::SubAssign: DISPATCH(BinSubAssign, BinaryOperator);
      case BinaryOperator::ShlAssign: DISPATCH(BinShlAssign, BinaryOperator);
      case BinaryOperator::ShrAssign: DISPATCH(BinShrAssign, BinaryOperator);
      case BinaryOperator::AndAssign: DISPATCH(BinAndAssign, BinaryOperator);
      case BinaryOperator::OrAssign:  DISPATCH(BinOrAssign,  BinaryOperator);
      case BinaryOperator::XorAssign: DISPATCH(BinXorAssign, BinaryOperator);
      case BinaryOperator::Comma:     DISPATCH(BinComma,     BinaryOperator);
      }
    }
    
    // Top switch stmt: dispatch to VisitFooStmt for each FooStmt.
    switch (S->getStmtClass()) {
    default: assert(0 && "Unknown stmt kind!");
#define STMT(N, CLASS, PARENT)                              \
    case Stmt::CLASS ## Class: DISPATCH(CLASS, CLASS);
#include "clang/AST/StmtNodes.def"
    }
  }
  
  // If the implementation chooses not to implement a certain visit method, fall
  // back on VisitExpr or whatever else is the superclass.
#define STMT(N, CLASS, PARENT)                                   \
  RetTy Visit ## CLASS(CLASS *S) { DISPATCH(PARENT, PARENT); }
#include "clang/AST/StmtNodes.def"

  // If the implementation doesn't implement binary operator methods, fall back
  // on VisitBinaryOperator.
  RetTy VisitBinMul(BinaryOperator *S){DISPATCH(BinaryOperator,BinaryOperator);}
  RetTy VisitBinDiv(BinaryOperator *S){DISPATCH(BinaryOperator,BinaryOperator);}
  RetTy VisitBinRem(BinaryOperator *S){DISPATCH(BinaryOperator,BinaryOperator);}
  RetTy VisitBinAdd(BinaryOperator *S){DISPATCH(BinaryOperator,BinaryOperator);}
  RetTy VisitBinSub(BinaryOperator *S){DISPATCH(BinaryOperator,BinaryOperator);}
  RetTy VisitBinShl(BinaryOperator *S){DISPATCH(BinaryOperator,BinaryOperator);}
  RetTy VisitBinShr(BinaryOperator *S){DISPATCH(BinaryOperator,BinaryOperator);}
  RetTy VisitBinAnd(BinaryOperator *S){DISPATCH(BinaryOperator,BinaryOperator);}
  RetTy VisitBinXor(BinaryOperator *S){DISPATCH(BinaryOperator,BinaryOperator);}
  RetTy VisitBinOr(BinaryOperator *S){DISPATCH(BinaryOperator,BinaryOperator);}
  RetTy VisitBinAssign(BinaryOperator *S) {
    DISPATCH(BinaryOperator,BinaryOperator);
  }
  RetTy VisitBinMulAssign(BinaryOperator *S) {
    DISPATCH(BinaryOperator,BinaryOperator);
  }
  RetTy VisitBinDivAssign(BinaryOperator *S) {
    DISPATCH(BinaryOperator,BinaryOperator);
  }
  RetTy VisitBinRemAssign(BinaryOperator *S) {
    DISPATCH(BinaryOperator,BinaryOperator);
  }
  RetTy VisitBinAddAssign(BinaryOperator *S) {
    DISPATCH(BinaryOperator,BinaryOperator);
  }
  RetTy VisitBinSubAssign(BinaryOperator *S) {
    DISPATCH(BinaryOperator,BinaryOperator);
  }
  RetTy VisitBinShlAssign(BinaryOperator *S) {
    DISPATCH(BinaryOperator,BinaryOperator);
  }
  RetTy VisitBinShrAssign(BinaryOperator *S) {
    DISPATCH(BinaryOperator,BinaryOperator);
  }
  RetTy VisitBinAndAssign(BinaryOperator *S) {
    DISPATCH(BinaryOperator,BinaryOperator);
  }
  RetTy VisitBinOrAssign(BinaryOperator *S) {
    DISPATCH(BinaryOperator,BinaryOperator);
  }
  RetTy VisitBinXorAssign(BinaryOperator *S) {
    DISPATCH(BinaryOperator,BinaryOperator);
  }
  RetTy VisitBinComma(BinaryOperator *S) {
    DISPATCH(BinaryOperator,BinaryOperator);
  }
  
  // Base case, ignore it. :)
  RetTy VisitStmt(Stmt *Node) { return RetTy(); }
};

#undef DISPATCH

}  // end namespace clang

#endif
