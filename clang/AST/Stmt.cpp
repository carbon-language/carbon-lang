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
#include "clang/Lex/IdentifierTable.h"
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
  unsigned counter;
  unsigned size;
} sNames[] = {
#define STMT(N, CLASS, PARENT) { N, #CLASS, 0, sizeof(CLASS) },
#include "clang/AST/StmtNodes.def"
  { 0, 0, 0, 0 }
};
  
const char *Stmt::getStmtClassName() const {
  for (int i = 0; sNames[i].className; i++) {
    if (sClass == sNames[i].enumValue)
      return sNames[i].className;
  }
  return 0; // should never happen....
}

void Stmt::PrintStats() {
  unsigned sum = 0;
  fprintf(stderr, "*** Stmt/Expr Stats:\n");
  for (int i = 0; sNames[i].className; i++) {
    sum += sNames[i].counter;
  }
  fprintf(stderr, "  %d stmts/exprs total.\n", sum);
  sum = 0;
  for (int i = 0; sNames[i].className; i++) {
    fprintf(stderr, "    %d %s, %d each (%d bytes)\n", 
      sNames[i].counter, sNames[i].className, sNames[i].size, sNames[i].counter*sNames[i].size);
    sum += sNames[i].counter*sNames[i].size;
  }
  fprintf(stderr, "Total bytes = %d\n", sum);
}

void Stmt::addStmtClass(StmtClass s) {
  for (int i = 0; sNames[i].className; i++) {
    if (s == sNames[i].enumValue)
      sNames[i].counter++;
  }
}

static bool StatSwitch = false;

bool Stmt::CollectingStats(bool enable) {
  if (enable) StatSwitch = true;
	return StatSwitch;
}



const char *LabelStmt::getName() const {
  return getID()->getName();
}

