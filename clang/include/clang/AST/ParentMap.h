//===--- ParentMap.h - Mappings from Stmts to their Parents -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ParentMap class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARENTMAP_H
#define LLVM_CLANG_PARENTMAP_H

namespace clang {
class Stmt;
class Expr;
  
class ParentMap {
  void* Impl;
public:
  ParentMap(Stmt* ASTRoot);
  ~ParentMap();

  Stmt *getParent(Stmt*) const;
  Stmt *getParentIgnoreParens(Stmt *) const;

  const Stmt *getParent(const Stmt* S) const {
    return getParent(const_cast<Stmt*>(S));
  }
  
  const Stmt *getParentIgnoreParens(const Stmt *S) const {
    return getParentIgnoreParens(const_cast<Stmt*>(S));
  }

  bool hasParent(Stmt* S) const {
    return getParent(S) != 0;
  }
  
  bool isConsumedExpr(Expr *E) const;
  
  bool isConsumedExpr(const Expr *E) const {
    return isConsumedExpr(const_cast<Expr*>(E));
  }
};
  
} // end clang namespace
#endif
