//===--- StmtIterator.cpp - Iterators for Statements ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines internal methods for StmtIterator.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtIterator.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"

using namespace clang;

void StmtIteratorBase::NextDecl() {
  assert (FirstDecl && Ptr.D);

  do Ptr.D = Ptr.D->getNextDeclarator();
  while (Ptr.D != NULL && !isa<VarDecl>(Ptr.D));
  
  if (Ptr.D == NULL) FirstDecl = NULL;
}

StmtIteratorBase::StmtIteratorBase(ScopedDecl* d) {
  assert (d);
  
  while (d != NULL) {
    if (VarDecl* V = dyn_cast<VarDecl>(d))
      if (V->getInit()) break;
    
    d = d->getNextDeclarator();
  }
  
  FirstDecl = d;
  Ptr.D = d;
}

void StmtIteratorBase::PrevDecl() {
  assert (FirstDecl);
  assert (Ptr.D != FirstDecl);
  
  // March through the list of decls until we find the decl just before
  // the one we currently point 
  
  ScopedDecl* d = FirstDecl;
  ScopedDecl* lastVD = d;
  
  while (d->getNextDeclarator() != Ptr.D) {
    if (VarDecl* V = dyn_cast<VarDecl>(d))
      if (V->getInit())
        lastVD = d;

    d = d->getNextDeclarator();
  }
  
  Ptr.D = lastVD;
}

Stmt* StmtIteratorBase::GetInitializer() const {
  return cast<VarDecl>(Ptr.D)->getInit();
}
