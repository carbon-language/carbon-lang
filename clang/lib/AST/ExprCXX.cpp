//===--- ExprCXX.cpp - (C++) Expression AST Node Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclesses of Expr class declared in ExprCXX.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ExprCXX.h"
using namespace clang;

//===----------------------------------------------------------------------===//
//  Child Iterators for iterating over subexpressions/substatements
//===----------------------------------------------------------------------===//


// CXXCastExpr
Stmt::child_iterator CXXCastExpr::child_begin() {
  return reinterpret_cast<Stmt**>(&Op);
}
Stmt::child_iterator CXXCastExpr::child_end() {
  return reinterpret_cast<Stmt**>(&Op)+1;
}

// CXXBoolLiteralExpr
Stmt::child_iterator CXXBoolLiteralExpr::child_begin() { 
  return child_iterator();
}
Stmt::child_iterator CXXBoolLiteralExpr::child_end() {
  return child_iterator();
}

// CXXThrowExpr
Stmt::child_iterator CXXThrowExpr::child_begin() {
  return reinterpret_cast<Stmt**>(&Op);
}
Stmt::child_iterator CXXThrowExpr::child_end() {
  // If Op is 0, we are processing throw; which has no children.
  if (Op == 0)
    return reinterpret_cast<Stmt**>(&Op)+0;
  return reinterpret_cast<Stmt**>(&Op)+1;
}

// CXXDefaultArgExpr
Stmt::child_iterator CXXDefaultArgExpr::child_begin() {
  return child_iterator();
}
Stmt::child_iterator CXXDefaultArgExpr::child_end() {
  return child_iterator();
}
