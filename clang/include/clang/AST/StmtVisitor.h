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

namespace llvm {
namespace clang {
  class Stmt;
  class Expr;
  class CompoundStmt;
  class IfStmt;
  class DoStmt;
  class WhileStmt;
  class ForStmt;
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
class StmtVisitor {
public:
  virtual ~StmtVisitor();
  
  virtual void VisitStmt(Stmt *Node) {}
  virtual void VisitExpr(Expr *Node);
  
  // Visitation methods for various Stmt subclasses.
  virtual void VisitCompoundStmt(CompoundStmt *Node);
  virtual void VisitIfStmt(IfStmt *Node);
  virtual void VisitWhileStmt(WhileStmt *Node);
  virtual void VisitDoStmt(DoStmt *Node);
  virtual void VisitForStmt(ForStmt *Node);
  virtual void VisitReturnStmt(ReturnStmt *Node);
  
  // Visitation methods for various Expr subclasses.
  virtual void VisitDeclRefExpr(DeclRefExpr *Node);
  virtual void VisitIntegerConstant(IntegerConstant *Node);
  virtual void VisitFloatingConstant(FloatingConstant *Node);
  virtual void VisitStringExpr(StringExpr *Node);
  virtual void VisitParenExpr(ParenExpr *Node);
  virtual void VisitUnaryOperator(UnaryOperator *Node);
  virtual void VisitSizeOfAlignOfTypeExpr(SizeOfAlignOfTypeExpr *Node);
  virtual void VisitArraySubscriptExpr(ArraySubscriptExpr *Node);
  virtual void VisitCallExpr(CallExpr *Node);
  virtual void VisitMemberExpr(MemberExpr *Node);
  virtual void VisitCastExpr(CastExpr *Node);
  virtual void VisitBinaryOperator(BinaryOperator *Node);
  virtual void VisitConditionalOperator(ConditionalOperator *Node);
};
  
}
}

#endif
