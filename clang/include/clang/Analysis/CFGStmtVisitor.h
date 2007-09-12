//===--- CFGStmtVisitor.h - Visitor for Stmts in a CFG ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CFGStmtVisitor interface, which extends 
//  StmtVisitor.  This interface is useful for visiting statements in a CFG
//  where some statements have implicit control-flow and thus should
//  be treated specially.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_CFGSTMTVISITOR_H
#define LLVM_CLANG_ANALYSIS_CFGSTMTVISITOR_H

#include "clang/AST/StmtVisitor.h"
#include "clang/AST/CFG.h"

namespace clang {

#define DISPATCH_CASE(CLASS) \
case Stmt::CLASS ## Class: return \
static_cast<ImplClass*>(this)->BlockStmt_Visit ## CLASS(static_cast<CLASS*>(S));  

#define DEFAULT_BLOCKSTMT_VISIT(CLASS) RetTy BlockStmt_Visit ## CLASS(CLASS *S)\
{ return\
  static_cast<ImplClass*>(this)->BlockStmt_VisitImplicitControlFlowStmt(S); }  

template <typename ImplClass, typename RetTy=void>
class CFGStmtVisitor : public StmtVisitor<ImplClass,RetTy> {
public:
  /// BlockVisit_XXX - Visitor methods for visiting the "root" statements in
  /// CFGBlocks.  Root statements are the statements that appear explicitly in 
  /// the list of statements in a CFGBlock.  For substatements, or when there
  /// is no implementation provided for a BlockStmt_XXX method, we default
  /// to using StmtVisitor's Visit method.
  RetTy BlockStmt_Visit(Stmt* S) {
    switch (S->getStmtClass()) {    
      DISPATCH_CASE(CallExpr)
      DISPATCH_CASE(StmtExpr)
      DISPATCH_CASE(ConditionalOperator)

      case Stmt::BinaryOperatorClass: {
        BinaryOperator* B = cast<BinaryOperator>(S);
        if (B->isLogicalOp())
          return static_cast<ImplClass*>(this)->BlockStmt_VisitLogicalOp(B);
        else if (B->getOpcode() == BinaryOperator::Comma)
          return static_cast<ImplClass*>(this)->BlockStmt_VisitComma(B);
        // Fall through.
      }
      
      default:
        return static_cast<ImplClass*>(this)->BlockStmt_VisitStmt(S);                             
    }
  }

  DEFAULT_BLOCKSTMT_VISIT(CallExpr)
  DEFAULT_BLOCKSTMT_VISIT(StmtExpr)
  DEFAULT_BLOCKSTMT_VISIT(ConditionalOperator)
  
  RetTy BlockStmt_VisitImplicitControlFlowStmt(Stmt* S) {
    return static_cast<ImplClass*>(this)->BlockStmt_VisitStmt(S);
  }
  
  RetTy BlockStmt_VisitStmt(Stmt* S) {
    return static_cast<ImplClass*>(this)->Visit(S);
  }

  RetTy BlockStmt_VisitLogicalOp(BinaryOperator* B) {
    return 
     static_cast<ImplClass*>(this)->BlockStmt_VisitImplicitControlFlowStmt(B);
  }
  
  RetTy BlockStmt_VisitComma(BinaryOperator* B) {
    return
     static_cast<ImplClass*>(this)->BlockStmt_VisitImplicitControlFlowStmt(B);
  }

  //===--------------------------------------------------------------------===//
  // Utility methods.  Not called by default (but subclasses may use them).
  //===--------------------------------------------------------------------===//
    
  /// VisitChildren: Call "Visit" on each child of S.
  void VisitChildren(Stmt* S) {
    for (Stmt::child_iterator I=S->child_begin(), E=S->child_end(); I != E;++I)
      static_cast<ImplClass*>(this)->Visit(*I);    
  }
};  
      
#undef DEFAULT_BLOCKSTMT_VISIT
#undef DISPATCH_CASE

}  // end namespace clang

#endif
