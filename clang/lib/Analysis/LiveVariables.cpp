//=- LiveVariables.cpp - Live Variable Analysis for Source CFGs -*- C++ --*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Live Variables analysis for source-level CFGs.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/Expr.h"
#include "clang/AST/CFG.h"
#include "clang/Analysis/Visitors/CFGRecStmtDeclVisitor.h"
#include "clang/Analysis/FlowSensitive/DataflowSolver.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"

#include <string.h>
#include <stdio.h>

using namespace clang;

//===----------------------------------------------------------------------===//
// Useful constants.
//===----------------------------------------------------------------------===//      

static const bool Alive = true;
static const bool Dead = false; 

//===----------------------------------------------------------------------===//
// Dataflow initialization logic.
//===----------------------------------------------------------------------===//      

namespace {
class VISIBILITY_HIDDEN RegisterDecls 
  : public CFGRecStmtDeclVisitor<RegisterDecls> {
    
  LiveVariables::AnalysisDataTy& AD;
  
  typedef llvm::SmallVector<VarDecl*, 20> AlwaysLiveTy;
  AlwaysLiveTy AlwaysLive;

    
public:
  RegisterDecls(LiveVariables::AnalysisDataTy& ad) : AD(ad) {}

  ~RegisterDecls() {

    AD.AlwaysLive.resetValues(AD);
    
    for (AlwaysLiveTy::iterator I = AlwaysLive.begin(), E = AlwaysLive.end();
         I != E; ++ I)       
      AD.AlwaysLive(*I, AD) = Alive;      
  }

  void VisitImplicitParamDecl(ImplicitParamDecl* IPD) {
    // Register the VarDecl for tracking.
    AD.Register(IPD);
  }

  void VisitVarDecl(VarDecl* VD) {
    // Register the VarDecl for tracking.
    AD.Register(VD);
    
    // Does the variable have global storage?  If so, it is always live.
    if (VD->hasGlobalStorage())
      AlwaysLive.push_back(VD);    
  }
  
  void VisitUnaryOperator(UnaryOperator* U) {
    // Check for '&'.  Any VarDecl whose value has its address-taken we
    // treat as always being live (flow-insensitive).

    Expr* E = U->getSubExpr()->IgnoreParenCasts();
    
    if (U->getOpcode() == UnaryOperator::AddrOf)
      if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(E))
        if (VarDecl* VD = dyn_cast<VarDecl>(DR->getDecl())) {
          AD.Register(VD);
          AlwaysLive.push_back(VD);
          return;
        }
      
    Visit(E);
  }
    
  CFG& getCFG() { return AD.getCFG(); }
};
} // end anonymous namespace


LiveVariables::LiveVariables(CFG& cfg) {
  // Register all referenced VarDecls.
  getAnalysisData().setCFG(&cfg);
  RegisterDecls R(getAnalysisData());
  cfg.VisitBlockStmts(R);
}

//===----------------------------------------------------------------------===//
// Transfer functions.
//===----------------------------------------------------------------------===//      

namespace {

class VISIBILITY_HIDDEN TransferFuncs : public CFGRecStmtVisitor<TransferFuncs>{
  LiveVariables::AnalysisDataTy& AD;
  LiveVariables::ValTy LiveState;
public:
  TransferFuncs(LiveVariables::AnalysisDataTy& ad) : AD(ad) {}

  LiveVariables::ValTy& getVal() { return LiveState; }
  CFG& getCFG() { return AD.getCFG(); }
  
  void VisitDeclRefExpr(DeclRefExpr* DR);
  void VisitBinaryOperator(BinaryOperator* B);
  void VisitAssign(BinaryOperator* B);
  void VisitDeclStmt(DeclStmt* DS);
  void VisitUnaryOperator(UnaryOperator* U);
  void Visit(Stmt *S);  
  void VisitTerminator(CFGBlock* B); 
  
  void SetTopValue(LiveVariables::ValTy& V) {
    V = AD.AlwaysLive;    
  }
  
};
      
void TransferFuncs::Visit(Stmt *S) {
  
  if (S == getCurrentBlkStmt()) {
    
    if (AD.Observer)
      AD.Observer->ObserveStmt(S,AD,LiveState);
    
    if (getCFG().isBlkExpr(S)) LiveState(S,AD) = Dead;
    StmtVisitor<TransferFuncs,void>::Visit(S);
  }
  else if (!getCFG().isBlkExpr(S)) {
    
    if (AD.Observer)
      AD.Observer->ObserveStmt(S,AD,LiveState);
    
    StmtVisitor<TransferFuncs,void>::Visit(S);
    
  }
  else
    // For block-level expressions, mark that they are live.
    LiveState(S,AD) = Alive;
}
  
void TransferFuncs::VisitTerminator(CFGBlock* B) {
    
  const Expr* E = B->getTerminatorCondition();

  if (!E)
    return;
  
  assert (getCFG().isBlkExpr(E));
  LiveState(E, AD) = Alive;
}

void TransferFuncs::VisitDeclRefExpr(DeclRefExpr* DR) {
  if (VarDecl* V = dyn_cast<VarDecl>(DR->getDecl())) 
    LiveState(V,AD) = Alive;
}
  
void TransferFuncs::VisitBinaryOperator(BinaryOperator* B) {     
  if (B->isAssignmentOp()) VisitAssign(B);
  else VisitStmt(B);
}

void TransferFuncs::VisitUnaryOperator(UnaryOperator* U) {
  Expr *E = U->getSubExpr();
  
  switch (U->getOpcode()) {
  case UnaryOperator::SizeOf: return;      
  case UnaryOperator::PostInc:
  case UnaryOperator::PostDec:
  case UnaryOperator::PreInc:
  case UnaryOperator::PreDec:
    // Walk through the subexpressions, blasting through ParenExprs
    // until we either find a DeclRefExpr or some non-DeclRefExpr
    // expression.
    if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(E->IgnoreParens())) 
      if (VarDecl* VD = dyn_cast<VarDecl>(DR->getDecl())) {
        // Treat the --/++ operator as a kill.
        if (AD.Observer) { AD.Observer->ObserverKill(DR); }
        LiveState(VD, AD) = Alive;
        return VisitDeclRefExpr(DR);
      }

    // Fall-through.
  
  default:
    return Visit(E);
  }
}
  
void TransferFuncs::VisitAssign(BinaryOperator* B) {    
  Expr* LHS = B->getLHS();

  // Assigning to a variable?
  if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(LHS->IgnoreParens())) {
    
    // Update liveness inforamtion.
    unsigned bit = AD.getIdx(DR->getDecl());
    LiveState.getDeclBit(bit) = Dead | AD.AlwaysLive.getDeclBit(bit);
    
    if (AD.Observer) { AD.Observer->ObserverKill(DR); }
    
    // Handle things like +=, etc., which also generate "uses"
    // of a variable.  Do this just by visiting the subexpression.
    if (B->getOpcode() != BinaryOperator::Assign)
      VisitDeclRefExpr(DR);
  }
  else // Not assigning to a variable.  Process LHS as usual.
    Visit(LHS);
  
  Visit(B->getRHS());
}

void TransferFuncs::VisitDeclStmt(DeclStmt* DS) {
  // Declarations effectively "kill" a variable since they cannot
  // possibly be live before they are declared.
  for (ScopedDecl* D = DS->getDecl(); D != NULL; D = D->getNextDeclarator())
    if (VarDecl* VD = dyn_cast<VarDecl>(D)) {
      
      // Update liveness information.
      unsigned bit = AD.getIdx(VD);
      LiveState.getDeclBit(bit) = Dead | AD.AlwaysLive.getDeclBit(bit);
      
      if (Expr* Init = VD->getInit())
        Visit(Init);
    }
}
  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Merge operator: if something is live on any successor block, it is live
//  in the current block (a set union).
//===----------------------------------------------------------------------===//      

namespace {

struct Merge {
  typedef ExprDeclBitVector_Types::ValTy ValTy; 
    
  void operator()(ValTy& Dst, const ValTy& Src) {
    Dst.OrDeclBits(Src);
    Dst.AndExprBits(Src);
  }
};
  
typedef DataflowSolver<LiveVariables, TransferFuncs, Merge> Solver;
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// External interface to run Liveness analysis.
//===----------------------------------------------------------------------===//      

void LiveVariables::runOnCFG(CFG& cfg) {
  Solver S(*this);
  S.runOnCFG(cfg);
}

void LiveVariables::runOnAllBlocks(const CFG& cfg,
                                   LiveVariables::ObserverTy* Obs,
                                   bool recordStmtValues) {
  Solver S(*this);
  ObserverTy* OldObserver = getAnalysisData().Observer;
  getAnalysisData().Observer = Obs;
  S.runOnAllBlocks(cfg, recordStmtValues);
  getAnalysisData().Observer = OldObserver;
}

//===----------------------------------------------------------------------===//
// liveness queries
//

bool LiveVariables::isLive(const CFGBlock* B, const VarDecl* D) const {
  DeclBitVector_Types::Idx i = getAnalysisData().getIdx(D);
  return i.isValid() ? getBlockData(B).getBit(i) : false;
}

bool LiveVariables::isLive(const ValTy& Live, const VarDecl* D) const {
  DeclBitVector_Types::Idx i = getAnalysisData().getIdx(D);
  return i.isValid() ? Live.getBit(i) : false;
}

bool LiveVariables::isLive(const Stmt* Loc, const Stmt* StmtVal) const {
  return getStmtData(Loc)(StmtVal,getAnalysisData());
}

bool LiveVariables::isLive(const Stmt* Loc, const VarDecl* D) const {
  return getStmtData(Loc)(D,getAnalysisData());
}

//===----------------------------------------------------------------------===//
// printing liveness state for debugging
//

void LiveVariables::dumpLiveness(const ValTy& V, SourceManager& SM) const {
  const AnalysisDataTy& AD = getAnalysisData();
  
  for (AnalysisDataTy::decl_iterator I = AD.begin_decl(),
                                     E = AD.end_decl(); I!=E; ++I)
    if (V.getDeclBit(I->second)) {      
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
