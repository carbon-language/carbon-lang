//===--- Stmt.h - Classes for representing statements -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Stmt interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMT_H
#define LLVM_CLANG_AST_STMT_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"
#include <iosfwd>

namespace llvm {
namespace clang {
  class Expr;
  class StmtVisitor;
  
/// Stmt - This represents one statement.
///
class Stmt {
public:
  Stmt() {}
  virtual ~Stmt() {}
  
  void dump() const;
  void print(std::ostream &OS) const;
  
  // Implement visitor support.
  virtual void visit(StmtVisitor &Visitor);
};

/// CompoundStmt - This represents a group of statements like { stmt stmt }.
///
class CompoundStmt : public Stmt {
  SmallVector<Stmt*, 16> Body;
public:
  CompoundStmt(Stmt **StmtStart, unsigned NumStmts)
    : Body(StmtStart, StmtStart+NumStmts) {}
  
  typedef SmallVector<Stmt*, 16>::iterator body_iterator;
  body_iterator body_begin() { return Body.begin(); }
  body_iterator body_end() { return Body.end(); }
  
  virtual void visit(StmtVisitor &Visitor);
};

/// IfStmt - This represents an if/then/else.
///
class IfStmt : public Stmt {
  Expr *Cond;
  Stmt *Then, *Else;
public:
  IfStmt(Expr *cond, Stmt *then, Stmt *elsev = 0)
    : Cond(cond), Then(then), Else(elsev) {}
  
  const Expr *getCond() const { return Cond; }
  const Stmt *getThen() const { return Then; }
  const Stmt *getElse() const { return Else; }

  Expr *getCond() { return Cond; }
  Stmt *getThen() { return Then; }
  Stmt *getElse() { return Else; }
  
  virtual void visit(StmtVisitor &Visitor);
};



/// ReturnStmt - This represents a return, optionally of an expression.
///
class ReturnStmt : public Stmt {
  Expr *RetExpr;
public:
  ReturnStmt(Expr *E = 0) : RetExpr(E) {}
  
  Expr *getRetValue() { return RetExpr; }
  
  virtual void visit(StmtVisitor &Visitor);
};


  
}  // end namespace clang
}  // end namespace llvm

#endif
