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
using namespace clang;

static struct StmtClassNameTable {
  const char *Name;
  unsigned Counter;
  unsigned Size;
} StmtClassInfo[Stmt::lastExprConstant];

static StmtClassNameTable &getStmtInfoTableEntry(Stmt::StmtClass E) {
  static bool Initialized = false;
  if (Initialized)
    return StmtClassInfo[E];

  // Intialize the table on the first use.
  Initialized = true;
#define STMT(N, CLASS, PARENT) \
  StmtClassInfo[N].Name = #CLASS; \
  StmtClassInfo[N].Size = sizeof(CLASS);
#include "clang/AST/StmtNodes.def"
  
  return StmtClassInfo[E];
}

const char *Stmt::getStmtClassName() const {
  return getStmtInfoTableEntry(sClass).Name;
}

void Stmt::PrintStats() {
  // Ensure the table is primed.
  getStmtInfoTableEntry(Stmt::NullStmtClass);
  
  unsigned sum = 0;
  fprintf(stderr, "*** Stmt/Expr Stats:\n");
  for (int i = 0; i != Stmt::lastExprConstant; i++) {
    if (StmtClassInfo[i].Name == 0) continue;
    sum += StmtClassInfo[i].Counter;
  }
  fprintf(stderr, "  %d stmts/exprs total.\n", sum);
  sum = 0;
  for (int i = 0; i != Stmt::lastExprConstant; i++) {
    if (StmtClassInfo[i].Name == 0) continue;
    fprintf(stderr, "    %d %s, %d each (%d bytes)\n", 
            StmtClassInfo[i].Counter, StmtClassInfo[i].Name,
            StmtClassInfo[i].Size,
            StmtClassInfo[i].Counter*StmtClassInfo[i].Size);
    sum += StmtClassInfo[i].Counter*StmtClassInfo[i].Size;
  }
  fprintf(stderr, "Total bytes = %d\n", sum);
}

void Stmt::addStmtClass(StmtClass s) {
  ++getStmtInfoTableEntry(s).Counter;
}

static bool StatSwitch = false;

bool Stmt::CollectingStats(bool enable) {
  if (enable) StatSwitch = true;
  return StatSwitch;
}


const char *LabelStmt::getName() const {
  return getID()->getName();
}

//===----------------------------------------------------------------------===//
//  Child Iterators for iterating over subexpressions/substatements
//===----------------------------------------------------------------------===//

// DeclStmt
Stmt::child_iterator DeclStmt::child_begin() { return NULL; }
Stmt::child_iterator DeclStmt::child_end() { return NULL; }

// NullStmt
Stmt::child_iterator NullStmt::child_begin() { return NULL; }
Stmt::child_iterator NullStmt::child_end() { return NULL; }

// CompoundStmt
Stmt::child_iterator CompoundStmt::child_begin() { return &Body[0]; }
Stmt::child_iterator CompoundStmt::child_end() { return &Body[0]+Body.size(); }

// SwitchCase
Stmt::child_iterator SwitchCase::child_begin() { return &SubStmt; }
Stmt::child_iterator SwitchCase::child_end() { return child_begin()+1; }

// LabelStmt
Stmt::child_iterator LabelStmt::child_begin() { return &SubStmt; }
Stmt::child_iterator LabelStmt::child_end() { return child_begin()+1; }

// IfStmt
Stmt::child_iterator IfStmt::child_begin() { return &SubExprs[0]; }
Stmt::child_iterator IfStmt::child_end() { return &SubExprs[0]+END_EXPR; }

// SwitchStmt
Stmt::child_iterator SwitchStmt::child_begin() { return &SubExprs[0]; }
Stmt::child_iterator SwitchStmt::child_end() { return &SubExprs[0]+END_EXPR; }

// WhileStmt
Stmt::child_iterator WhileStmt::child_begin() { return &SubExprs[0]; }
Stmt::child_iterator WhileStmt::child_end() { return &SubExprs[0]+END_EXPR; }

// DoStmt
Stmt::child_iterator DoStmt::child_begin() { return &SubExprs[0]; }
Stmt::child_iterator DoStmt::child_end() { return &SubExprs[0]+END_EXPR; }

// ForStmt
Stmt::child_iterator ForStmt::child_begin() { return &SubExprs[0]; }
Stmt::child_iterator ForStmt::child_end() { return &SubExprs[0]+END_EXPR; }

// GotoStmt
Stmt::child_iterator GotoStmt::child_begin() { return NULL; }
Stmt::child_iterator GotoStmt::child_end() { return NULL; }

// IndirectGotoStmt
Stmt::child_iterator IndirectGotoStmt::child_begin() { 
  return reinterpret_cast<Stmt**>(&Target); 
}

Stmt::child_iterator IndirectGotoStmt::child_end() { return child_begin()+1; }

// ContinueStmt
Stmt::child_iterator ContinueStmt::child_begin() { return NULL; }
Stmt::child_iterator ContinueStmt::child_end() { return NULL; }

// BreakStmt
Stmt::child_iterator BreakStmt::child_begin() { return NULL; }
Stmt::child_iterator BreakStmt::child_end() { return NULL; }

// ReturnStmt
Stmt::child_iterator ReturnStmt::child_begin() { 
  return reinterpret_cast<Stmt**>(&RetExpr); 
}

Stmt::child_iterator ReturnStmt::child_end() { return child_begin()+1; }

