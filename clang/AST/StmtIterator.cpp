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

void StmtIterator::NextDecl() { assert(false); }
void StmtIterator::PrevDecl() { assert(false); }

Stmt*& StmtIterator::GetInitializer() const {
  assert (D && isa<VarDecl>(D));
  assert (cast<VarDecl>(D)->Init);
  return reinterpret_cast<Stmt*&>(cast<VarDecl>(D)->Init);
}
