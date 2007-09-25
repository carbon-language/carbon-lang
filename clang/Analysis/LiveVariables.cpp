//=- LiveVariables.cpp - Live Variable Analysis for Source CFGs -*- C++ --*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for
// details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Live Variables analysis for source-level CFGs.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/LiveVariables.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/Expr.h"
#include "clang/AST/CFG.h"
#include "clang/Analysis/Visitors/CFGRecStmtDeclVisitor.h"
#include "clang/Analysis/FlowSensitive/DataflowSolver.h"
#include "clang/Lex/IdentifierTable.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <string.h>
#include <stdio.h>

using namespace clang;

//===----------------------------------------------------------------------===//
// Dataflow initialization logic.
//===----------------------------------------------------------------------===//      

namespace {
struct RegisterDecls : public CFGRecStmtDeclVisitor<RegisterDecls> {
  LiveVariables::AnalysisDataTy& AD;
  void VisitVarDecl(VarDecl* VD) { AD.RegisterDecl(VD); }

  RegisterDecls(LiveVariables::AnalysisDataTy& ad) : AD(ad) {}
};  
} // end anonymous namespace

void LiveVariables::InitializeValues(const CFG& cfg) {
  RegisterDecls R(getAnalysisData());
  cfg.VisitBlockStmts(R);
}

//===----------------------------------------------------------------------===//
// Transfer functions.
//===----------------------------------------------------------------------===//      

namespace {
class TransferFuncs : public CFGStmtVisitor<TransferFuncs> {
  LiveVariables::AnalysisDataTy& AD;
  LiveVariables::ValTy Live;
public:
  TransferFuncs(LiveVariables::AnalysisDataTy& ad) : AD(ad) {}

  LiveVariables::ValTy& getVal() { return Live; }
  
  void VisitDeclRefExpr(DeclRefExpr* DR);
  void VisitBinaryOperator(BinaryOperator* B);
  void VisitAssign(BinaryOperator* B);
  void VisitDeclStmt(DeclStmt* DS);
  void VisitUnaryOperator(UnaryOperator* U);
  void VisitStmt(Stmt* S) { VisitChildren(S); }
  void Visit(Stmt *S) {
    if (AD.Observer) AD.Observer->ObserveStmt(S,AD,Live);
    static_cast<CFGStmtVisitor<TransferFuncs>*>(this)->Visit(S);
  }
};

void TransferFuncs::VisitDeclRefExpr(DeclRefExpr* DR) {
  if (VarDecl* V = dyn_cast<VarDecl>(DR->getDecl())) 
    Live.set(AD[V]);   // Register a use of the variable.
}
  
void TransferFuncs::VisitBinaryOperator(BinaryOperator* B) {     
  if (B->isAssignmentOp()) VisitAssign(B);
  else VisitStmt(B);
}

void TransferFuncs::VisitUnaryOperator(UnaryOperator* U) {
  switch (U->getOpcode()) {
  case UnaryOperator::PostInc:
  case UnaryOperator::PostDec:
  case UnaryOperator::PreInc:
  case UnaryOperator::PreDec:
  case UnaryOperator::AddrOf:
    // Walk through the subexpressions, blasting through ParenExprs
    // until we either find a DeclRefExpr or some non-DeclRefExpr
    // expression.
    for (Stmt* S = U->getSubExpr() ;;) {
      if (ParenExpr* P = dyn_cast<ParenExpr>(S)) { S=P->getSubExpr(); continue;} 
      else if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(S)) {
        // Treat the --/++/& operator as a kill.
        Live.reset(AD[DR->getDecl()]);
        if (AD.Observer) AD.Observer->ObserverKill(DR);
        VisitDeclRefExpr(DR);          
      }
      else Visit(S);

      break;   
    }        
    break;
  
  default:
    Visit(U->getSubExpr());
    break;
  }
}

void TransferFuncs::VisitAssign(BinaryOperator* B) {    
  Stmt* LHS = B->getLHS();

  if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(LHS)) { // Assigning to a var?    
    Live.reset(AD[DR->getDecl()]);
    if (AD.Observer) AD.Observer->ObserverKill(DR);
    // Handle things like +=, etc., which also generate "uses"
    // of a variable.  Do this just by visiting the subexpression.
    if (B->getOpcode() != BinaryOperator::Assign) Visit(LHS);
  }
  else // Not assigning to a variable.  Process LHS as usual.
    Visit(LHS);
  
  Visit(B->getRHS());
}

void TransferFuncs::VisitDeclStmt(DeclStmt* DS) {
  // Declarations effectively "kill" a variable since they cannot
  // possibly be live before they are declared.
  for (ScopedDecl* D = DS->getDecl(); D != NULL ; D = D->getNextDeclarator())
    Live.reset(AD[D]);
}
} // end anonymous namespace

namespace {
struct Merge {
  void operator()(LiveVariables::ValTy& Dst, LiveVariables::ValTy& Src) {
    Src |= Dst;
  }
};
} // end anonymous namespace

namespace {
typedef DataflowSolver<LiveVariables,TransferFuncs,Merge> Solver;
}

void LiveVariables::runOnCFG(const CFG& cfg) {
  Solver S(*this);
  S.runOnCFG(cfg);
}

void LiveVariables::runOnAllBlocks(const CFG& cfg,
                                   LiveVariables::ObserverTy& Obs) {
  Solver S(*this);
  ObserverTy* OldObserver = getAnalysisData().Observer;
  getAnalysisData().Observer = &Obs;
  S.runOnAllBlocks(cfg);
  getAnalysisData().Observer = OldObserver;
}

//===----------------------------------------------------------------------===//
// liveness queries
//

bool LiveVariables::isLive(const CFGBlock* B, const VarDecl* D) const {
  return getBlockData(B)[ getAnalysisData()[D] ];
}

bool LiveVariables::isLive(const ValTy& Live, const VarDecl* D) const {
  return Live[ getAnalysisData()[D] ];
}

//===----------------------------------------------------------------------===//
// printing liveness state for debugging
//

void LiveVariables::dumpLiveness(const ValTy& V, SourceManager& SM) const {
  const AnalysisDataTy& AD = getAnalysisData();
  
  for (AnalysisDataTy::iterator I = AD.begin(), E = AD.end(); I!=E; ++I)
    if (V[I->second]) {      
      SourceLocation PhysLoc = SM.getPhysicalLoc(I->first->getLocation());
    
      fprintf(stderr, "  %s <%s:%u:%u>\n", 
              I->first->getIdentifier()->getName(),
              SM.getSourceName(PhysLoc),
              SM.getLineNumber(PhysLoc),
              SM.getColumnNumber(PhysLoc));
    }
}                                  

void LiveVariables::dumpBlockLiveness(SourceManager& M) const {
  for (BlockDataMapTy::iterator I = getBlockDataMap().begin(),
       E = getBlockDataMap().end(); I!=E; ++I) {
    fprintf(stderr, "\n[ B%d (live variables at block exit) ]\n",
            I->first->getBlockID());
            
    dumpLiveness(I->second,M);
  }

  fprintf(stderr,"\n");
}
