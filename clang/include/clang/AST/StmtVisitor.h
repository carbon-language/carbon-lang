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
#define LLVM_CLANG_AST_STMT_H

namespace llvm {
namespace clang {
  class CompoundStmt;
  class IfStmt;
  class ReturnStmt;

  class DeclRefExpr;
  class IntegerConstant;
  class FloatingConstant;
  class StringExpr;
  class ParenExpr;
  class UnaryOperator;
  class SizeOfAlignOfTypeExpr;
  class ArraySubscriptExpr;
  class CallExpr;
  class MemberExpr;
  class CastExpr;
  class BinaryOperator;
  class ConditionalOperator;
  
/// StmtVisitor - This class implements a simple visitor for Stmt subclasses.
/// Since Expr derives from Stmt, this also includes support for visiting Exprs.
struct StmtVisitor {
  virtual ~StmtVisitor();
  
  /// VisitNull - Visit a null pointer.
  ///
  virtual void VisitNull() {}

  // Visitation methods for various subclasses.
  virtual void VisitCompoundStmt(CompoundStmt *Node) {}
  virtual void VisitIfStmt      (IfStmt       *Node) {}
  virtual void VisitReturnStmt  (ReturnStmt   *Node) {}
  
  virtual void VisitDeclRefExpr(DeclRefExpr *Node) {}
  virtual void VisitIntegerConstant(IntegerConstant *Node) {}
  virtual void VisitFloatingConstant(FloatingConstant *Node) {}
  virtual void VisitStringExpr(StringExpr *Node) {}
  virtual void VisitParenExpr(ParenExpr *Node) {}
  virtual void VisitUnaryOperator(UnaryOperator *Node) {}
  virtual void VisitSizeOfAlignOfTypeExpr(SizeOfAlignOfTypeExpr *Node) {}
  virtual void VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {}
  virtual void VisitCallExpr(CallExpr *Node) {}
  virtual void VisitMemberExpr(MemberExpr *Node) {}
  virtual void VisitCastExpr(CastExpr *Node) {}
  virtual void VisitBinaryOperator(BinaryOperator *Node) {}
  virtual void VisitConditionalOperator(ConditionalOperator *Node) {}
};
  
}
}

#endif
