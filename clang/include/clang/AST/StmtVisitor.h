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
  // Add prototypes for all AST node classes.
#define STMT(N, CLASS, PARENT) \
  class CLASS;
#include "clang/AST/StmtNodes.def"
  
/// StmtVisitor - This class implements a simple visitor for Stmt subclasses.
/// Since Expr derives from Stmt, this also includes support for visiting Exprs.
class StmtVisitor {
public:
  virtual ~StmtVisitor();
  
  virtual void VisitStmt(Stmt *Node) {}
  
  // Implement all the methods with the StmtNodes.def file.
#define STMT(N, CLASS, PARENT) \
  virtual void Visit##CLASS(CLASS *Node);
#include "clang/AST/StmtNodes.def"
};
  
}
}

#endif
