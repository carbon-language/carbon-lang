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
  
/// StmtVisitor - This class implements a simple visitor for Stmt subclasses.
/// Since Expr derives from Stmt, this also includes support for visiting Exprs.
template<typename ImplClass>
class StmtVisitor {
public:
  void Visit(Stmt *S) {
    // Top switch stmt: dispatch to VisitFooStmt for each FooStmt.
    switch (S->getStmtClass()) {
    default: assert(0 && "Unknown stmt kind!");
#define STMT(N, CLASS, PARENT)                              \
    case Stmt::CLASS ## Class:                              \
      return static_cast<ImplClass*>(this)->Visit ## CLASS( \
             static_cast<CLASS*>(S));
#include "clang/AST/StmtNodes.def"
    }
  }
  
  // If the implementation chooses not to implement a certain visit method, fall
  // back on VisitExpr or whatever else is the superclass.
#define STMT(N, CLASS, PARENT)                                   \
  void Visit ## CLASS(CLASS *Node) {                             \
    return static_cast<ImplClass*>(this)->Visit ## PARENT(Node); \
  }
#include "clang/AST/StmtNodes.def"

  // Base case, ignore it. :)
  void VisitStmt(Stmt *Node) {}
};
  
}

#endif
