//===--- Attr.cpp -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Steve Naroff and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Attribute class interfaces
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/Lex/IdentifierTable.h"
using namespace llvm;
using namespace clang;

Attr::Attr(SourceLocation L, IdentifierInfo *aName, 
                IdentifierInfo *pname, Expr **elist, unsigned numargs)
  : AttrName(aName), AttrLoc(L), ParmName(pname), NumArgs(numargs) {
  Args = new Expr*[numargs];
  for (unsigned i = 0; i != numargs; ++i)
    Args[i] = elist[i];
}
