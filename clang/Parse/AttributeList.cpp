//===--- AttributeList.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Steve Naroff and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AttributeList class implementation
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/AttributeList.h"
#include "clang/Lex/IdentifierTable.h"
using namespace clang;

AttributeList::AttributeList(IdentifierInfo *aName, SourceLocation aLoc,
                             IdentifierInfo *pName, SourceLocation pLoc,
                             Action::ExprTy **elist, unsigned numargs, 
                             AttributeList *n)
  : AttrName(aName), AttrLoc(aLoc), ParmName(pName), ParmLoc(pLoc),
    NumArgs(numargs), Next(n) {
  Args = new Action::ExprTy*[numargs];
  for (unsigned i = 0; i != numargs; ++i)
    Args[i] = elist[i];
}
