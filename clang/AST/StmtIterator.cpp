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
#include "clang/AST/Stmt.h"
#include "clang/AST/Decl.h"

using namespace clang;

void StmtIterator::NextDecl() {
  assert (D);
  do D = D->getNextDeclarator();
  while (D != NULL && !isa<VarDecl>(D));
  
  if (!D) S = NULL;
}

void StmtIterator::PrevDecl() {
  assert (isa<DeclStmt>(*S));
  DeclStmt* DS = cast<DeclStmt>(*S);

  ScopedDecl* d = DS->getDecl();
  assert (d);
  
  if (d == D) { assert(false) ; return; }
  
  // March through the list of decls until we find the decl just before
  // the one we currently point 
  
  while (d->getNextDeclarator() != D)
    d = d->getNextDeclarator();
  
  D = d;
}

Stmt*& StmtIterator::GetInitializer() const {
  assert (D && isa<VarDecl>(D));
  assert (cast<VarDecl>(D)->Init);
  return reinterpret_cast<Stmt*&>(cast<VarDecl>(D)->Init);
}
