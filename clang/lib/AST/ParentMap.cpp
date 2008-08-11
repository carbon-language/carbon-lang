//===--- ParentMap.cpp - Mappings from Stmts to their Parents ---*- C++ -*-===//
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

#include "clang/AST/ParentMap.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/DenseMap.h"

using namespace clang;

typedef llvm::DenseMap<Stmt*, Stmt*> MapTy;

static void BuildParentMap(MapTy& M, Stmt* S) {
  for (Stmt::child_iterator I=S->child_begin(), E=S->child_end(); I!=E; ++I)
    if (*I) {
      M[*I] = S;
      BuildParentMap(M, *I);
    }
}

ParentMap::ParentMap(Stmt* S) : Impl(0) {
  if (S) {
    MapTy *M = new MapTy();
    BuildParentMap(*M, S);
    Impl = M;    
  }
}

ParentMap::~ParentMap() {
  delete (MapTy*) Impl;
}

Stmt* ParentMap::getParent(Stmt* S) const {
  MapTy* M = (MapTy*) Impl;
  MapTy::iterator I = M->find(S);
  return I == M->end() ? 0 : I->second;
}

bool ParentMap::isSubExpr(Stmt* S) const {
  if (!isa<Expr>(S))
    return false;
  
  Stmt* P = getParent(S);
  return P ? !isa<CompoundStmt>(P) : false;
}
