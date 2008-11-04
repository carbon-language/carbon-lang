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

void CXXConditionDeclExpr::Destroy(ASTContext& C) {
  getVarDecl()->Destroy(C);
  delete this;
}


//===----------------------------------------------------------------------===//
//  Child Iterators for iterating over subexpressions/substatements
//===----------------------------------------------------------------------===//


// CXXBoolLiteralExpr
Stmt::child_iterator CXXBoolLiteralExpr::child_begin() { 
  return child_iterator();
}
Stmt::child_iterator CXXBoolLiteralExpr::child_end() {
  return child_iterator();
}

// CXXThisExpr
Stmt::child_iterator CXXThisExpr::child_begin() { return child_iterator(); }
Stmt::child_iterator CXXThisExpr::child_end() { return child_iterator(); }

// CXXThrowExpr
Stmt::child_iterator CXXThrowExpr::child_begin() { return &Op; }
Stmt::child_iterator CXXThrowExpr::child_end() {
  // If Op is 0, we are processing throw; which has no children.
  return Op ? &Op+1 : &Op;
}

// CXXDefaultArgExpr
Stmt::child_iterator CXXDefaultArgExpr::child_begin() {
  return child_iterator();
}
Stmt::child_iterator CXXDefaultArgExpr::child_end() {
  return child_iterator();
}

// CXXZeroInitValueExpr
Stmt::child_iterator CXXZeroInitValueExpr::child_begin() { 
  return child_iterator();
}
Stmt::child_iterator CXXZeroInitValueExpr::child_end() {
  return child_iterator();
}

// CXXConditionDeclExpr
Stmt::child_iterator CXXConditionDeclExpr::child_begin() {
  return getVarDecl();
}
Stmt::child_iterator CXXConditionDeclExpr::child_end() {
  return child_iterator();
}

//===----------------------------------------------------------------------===//
//  Named casts
//===----------------------------------------------------------------------===//

/// getCastName - Get the name of the C++ cast being used, e.g.,
/// "static_cast", "dynamic_cast", "reinterpret_cast", or
/// "const_cast". The returned pointer must not be freed.
const char *CXXNamedCastExpr::getCastName() const {
  switch (getStmtClass()) {
  case CXXStaticCastExprClass:      return "static_cast";
  case CXXDynamicCastExprClass:     return "dynamic_cast";
  case CXXReinterpretCastExprClass: return "reinterpret_cast";
  case CXXConstCastExprClass:       return "const_cast";
  default:                          return "<invalid cast>";
  }
}
