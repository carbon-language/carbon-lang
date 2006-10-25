//===--- Stmt.cpp - Statement AST Node Implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt class and statement subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include <iostream>
using namespace llvm;
using namespace clang;

void Stmt::dump() const {
  if (this == 0) {
    std::cerr << "<null>";
    return;
  }
  bool isExpr = dynamic_cast<const Expr*>(this) != 0;
  if (isExpr) std::cerr << "(";
  dump_impl();
  if (isExpr) std::cerr << ")";
}
