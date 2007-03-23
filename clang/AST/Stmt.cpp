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
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtVisitor.h"
using namespace llvm;
using namespace clang;

// Implement all the AST node visit methods using the StmtNodes.def database.
#define STMT(N, CLASS, PARENT) \
void CLASS::visit(StmtVisitor &V) { return V.Visit##CLASS(this); }

STMT(0, Stmt, )
#include "clang/AST/StmtNodes.def"

static struct StmtClassNameTable {
  int enumValue;
  const char *className;
} sNames[] = {
#define STMT(N, CLASS, PARENT) { N, #CLASS },
#include "clang/AST/StmtNodes.def"
  { 0, 0 }
};
  
const char *Stmt::getStmtClassName() const {
  for (int i = 0; sNames[i].className; i++) {
    if (sClass == sNames[i].enumValue)
      return sNames[i].className;
  }
  return 0; // should never happen....
}
  
