//===--- Stmt.cpp - Statement AST Node Implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt class and statement subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Stmt.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/IdentifierTable.h"
using namespace clang;

static struct StmtClassNameTable {
  const char *Name;
  unsigned Counter;
  unsigned Size;
} StmtClassInfo[Stmt::lastExprConstant+1];

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

void Stmt::DestroyChildren() {
  for (child_iterator I = child_begin(), E = child_end(); I !=E; ++I)
    delete *I; // Handles the case when *I == NULL.
}

void Stmt::PrintStats() {
  // Ensure the table is primed.
  getStmtInfoTableEntry(Stmt::NullStmtClass);
  
  unsigned sum = 0;
  fprintf(stderr, "*** Stmt/Expr Stats:\n");
  for (int i = 0; i != Stmt::lastExprConstant+1; i++) {
    if (StmtClassInfo[i].Name == 0) continue;
    sum += StmtClassInfo[i].Counter;
  }
  fprintf(stderr, "  %d stmts/exprs total.\n", sum);
  sum = 0;
  for (int i = 0; i != Stmt::lastExprConstant+1; i++) {
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

// This is defined here to avoid polluting Stmt.h with importing Expr.h
SourceRange ReturnStmt::getSourceRange() const { 
  if (RetExpr)
    return SourceRange(RetLoc, RetExpr->getLocEnd());
  else
    return SourceRange(RetLoc);
}

bool Stmt::hasImplicitControlFlow() const {
  switch (sClass) {
    default:
      return false;
      
    case CallExprClass:
    case ConditionalOperatorClass:
    case ChooseExprClass:
    case StmtExprClass:
    case DeclStmtClass:
      return true;    
      
    case Stmt::BinaryOperatorClass: {
      const BinaryOperator* B = cast<BinaryOperator>(this);
      if (B->isLogicalOp() || B->getOpcode() == BinaryOperator::Comma)
        return true;
      else
        return false;
    }
  }
}

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

AsmStmt::AsmStmt(SourceLocation asmloc, bool issimple, bool isvolatile,
                 unsigned numoutputs, unsigned numinputs,
                 std::string *names, StringLiteral **constraints,
                 Expr **exprs, StringLiteral *asmstr, unsigned numclobbers,
                 StringLiteral **clobbers, SourceLocation rparenloc)
  : Stmt(AsmStmtClass), AsmLoc(asmloc), RParenLoc(rparenloc), AsmStr(asmstr)
  , IsSimple(issimple), IsVolatile(isvolatile)
  , NumOutputs(numoutputs), NumInputs(numinputs) {
  for (unsigned i = 0, e = numinputs + numoutputs; i != e; i++) {
    Names.push_back(names[i]);
    Exprs.push_back(exprs[i]);
    Constraints.push_back(constraints[i]);    
  }
  
  for (unsigned i = 0; i != numclobbers; i++)
    Clobbers.push_back(clobbers[i]);
}

ObjCForCollectionStmt::ObjCForCollectionStmt(Stmt *Elem, Expr *Collect,
                                             Stmt *Body,  SourceLocation FCL,
                                             SourceLocation RPL) 
: Stmt(ObjCForCollectionStmtClass) {
  SubExprs[ELEM] = Elem;
  SubExprs[COLLECTION] = reinterpret_cast<Stmt*>(Collect);
  SubExprs[BODY] = Body;
  ForLoc = FCL;
  RParenLoc = RPL;
}


ObjCAtCatchStmt::ObjCAtCatchStmt(SourceLocation atCatchLoc, 
                                 SourceLocation rparenloc, 
                                 Stmt *catchVarStmtDecl, Stmt *atCatchStmt, 
                                 Stmt *atCatchList)
: Stmt(ObjCAtCatchStmtClass) {
  SubExprs[SELECTOR] = catchVarStmtDecl;
  SubExprs[BODY] = atCatchStmt;
  if (!atCatchList)
    SubExprs[NEXT_CATCH] = NULL;
  else {
    ObjCAtCatchStmt *AtCatchList = static_cast<ObjCAtCatchStmt*>(atCatchList);

    while (ObjCAtCatchStmt* NextCatch = AtCatchList->getNextCatchStmt())      
      AtCatchList = NextCatch;
    
    AtCatchList->SubExprs[NEXT_CATCH] = this;
  }
  AtCatchLoc = atCatchLoc;
  RParenLoc = rparenloc;
}


//===----------------------------------------------------------------------===//
//  Child Iterators for iterating over subexpressions/substatements
//===----------------------------------------------------------------------===//

// DeclStmt
Stmt::child_iterator DeclStmt::child_begin() { return getDecl(); }
Stmt::child_iterator DeclStmt::child_end() { return child_iterator(); }

// NullStmt
Stmt::child_iterator NullStmt::child_begin() { return child_iterator(); }
Stmt::child_iterator NullStmt::child_end() { return child_iterator(); }

// CompoundStmt
Stmt::child_iterator CompoundStmt::child_begin() { return &Body[0]; }
Stmt::child_iterator CompoundStmt::child_end() { return &Body[0]+Body.size(); }

// CaseStmt
Stmt::child_iterator CaseStmt::child_begin() { return &SubExprs[0]; }
Stmt::child_iterator CaseStmt::child_end() { return &SubExprs[END_EXPR]; }

// DefaultStmt
Stmt::child_iterator DefaultStmt::child_begin() { return &SubStmt; }
Stmt::child_iterator DefaultStmt::child_end() { return &SubStmt+1; }

// LabelStmt
Stmt::child_iterator LabelStmt::child_begin() { return &SubStmt; }
Stmt::child_iterator LabelStmt::child_end() { return &SubStmt+1; }

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

// ObjCForCollectionStmt
Stmt::child_iterator ObjCForCollectionStmt::child_begin() { 
  return &SubExprs[0]; 
}
Stmt::child_iterator ObjCForCollectionStmt::child_end() { 
  return &SubExprs[0]+END_EXPR; 
}

// GotoStmt
Stmt::child_iterator GotoStmt::child_begin() { return child_iterator(); }
Stmt::child_iterator GotoStmt::child_end() { return child_iterator(); }

// IndirectGotoStmt
Stmt::child_iterator IndirectGotoStmt::child_begin() { 
  return reinterpret_cast<Stmt**>(&Target); 
}

Stmt::child_iterator IndirectGotoStmt::child_end() { return ++child_begin(); }

// ContinueStmt
Stmt::child_iterator ContinueStmt::child_begin() { return child_iterator(); }
Stmt::child_iterator ContinueStmt::child_end() { return child_iterator(); }

// BreakStmt
Stmt::child_iterator BreakStmt::child_begin() { return child_iterator(); }
Stmt::child_iterator BreakStmt::child_end() { return child_iterator(); }

// ReturnStmt
Stmt::child_iterator ReturnStmt::child_begin() {
  if (RetExpr) return reinterpret_cast<Stmt**>(&RetExpr);
  else return child_iterator();
}

Stmt::child_iterator ReturnStmt::child_end() { 
  if (RetExpr) return reinterpret_cast<Stmt**>(&RetExpr)+1;
  else return child_iterator();
}

// AsmStmt
Stmt::child_iterator AsmStmt::child_begin() { return child_iterator(); }
Stmt::child_iterator AsmStmt::child_end() { return child_iterator(); }

// ObjCAtCatchStmt
Stmt::child_iterator ObjCAtCatchStmt::child_begin() { return &SubExprs[0]; }
Stmt::child_iterator ObjCAtCatchStmt::child_end() { 
  return &SubExprs[0]+END_EXPR; 
}

// ObjCAtFinallyStmt
Stmt::child_iterator ObjCAtFinallyStmt::child_begin() { return &AtFinallyStmt; }
Stmt::child_iterator ObjCAtFinallyStmt::child_end() { return &AtFinallyStmt+1; }

// ObjCAtTryStmt
Stmt::child_iterator ObjCAtTryStmt::child_begin() { return &SubStmts[0]; }
Stmt::child_iterator ObjCAtTryStmt::child_end()   { 
  return &SubStmts[0]+END_EXPR; 
}

// ObjCAtThrowStmt
Stmt::child_iterator ObjCAtThrowStmt::child_begin() {
  return &Throw;
}

Stmt::child_iterator ObjCAtThrowStmt::child_end() {
  return &Throw+1;
}

// ObjCAtSynchronizedStmt
Stmt::child_iterator ObjCAtSynchronizedStmt::child_begin() {
  return &SubStmts[0];
}

Stmt::child_iterator ObjCAtSynchronizedStmt::child_end() {
  return &SubStmts[0]+END_EXPR;
}

