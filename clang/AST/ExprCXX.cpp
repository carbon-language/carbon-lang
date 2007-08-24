//===--- ExprCXX.cpp - (C++) Expression AST Node Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
  return child_begin()+1;
}

// CXXBoolLiteralExpr
Stmt::child_iterator CXXBoolLiteralExpr::child_begin() { return NULL; }
Stmt::child_iterator CXXBoolLiteralExpr::child_end() { return NULL; }
