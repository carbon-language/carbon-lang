//==- LiveVariables.cpp - Live Variable Analysis for Source CFGs -*- C++ --*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/IdentifierTable.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <string.h>
#include <stdio.h>

using namespace clang;

//===----------------------------------------------------------------------===//
// RegisterDecls - Utility class to create VarInfo objects for all
//                 Decls referenced in a function.
//

namespace {

class RegisterDecls : public StmtVisitor<RegisterDecls,void> {
  LiveVariables& L;
  const CFG& cfg;
public:  
  RegisterDecls(LiveVariables& l, const CFG& c)
    : L(l), cfg(c) {}
    
  void VisitStmt(Stmt* S);
  void VisitDeclRefExpr(DeclRefExpr* DR);
  void VisitDeclStmt(DeclStmt* DS);
  void Register(Decl* D);
  void RegisterDeclChain(Decl* D);
  void RegisterUsedDecls();
};

void RegisterDecls::VisitStmt(Stmt* S) {
  for (Stmt::child_iterator I = S->child_begin(),E = S->child_end(); I != E;++I)
    Visit(*I);
}

void RegisterDecls::VisitDeclRefExpr(DeclRefExpr* DR) {
  RegisterDeclChain(DR->getDecl());
}

void RegisterDecls::VisitDeclStmt(DeclStmt* DS) {
  RegisterDeclChain(DS->getDecl());
}

void RegisterDecls::RegisterDeclChain(Decl* D) {
  for (; D != NULL ; D = D->getNextDeclarator())
    Register(D);
}

void RegisterDecls::Register(Decl* D) {
  LiveVariables::VPair& VP = L.getVarInfoMap()[const_cast<const Decl*>(D)];

  VP.V.AliveBlocks.reserve(cfg.getNumBlockIDs());
  VP.Idx = L.getNumDecls()++;
}

void RegisterDecls::RegisterUsedDecls() {
  for (CFG::const_iterator BI = cfg.begin(), BE = cfg.end(); BI != BE; ++BI)
    for (CFGBlock::const_iterator SI=BI->begin(),SE = BI->end();SI != SE;++SI)
      Visit(const_cast<Stmt*>(*SI));
}
  
  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// WorkList - Data structure representing the liveness algorithm worklist.
//

namespace {

class WorkListTy {
  typedef llvm::SmallPtrSet<const CFGBlock*,20> BlockSet;
  BlockSet wlist;
public:
  void enqueue(const CFGBlock* B) { wlist.insert(B); }
      
  const CFGBlock* dequeue() {
    assert (!wlist.empty());
    const CFGBlock* B = *wlist.begin();
    wlist.erase(B);
    return B;          
  }
  
  bool isEmpty() const { return wlist.empty(); }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TFuncs
//

namespace {

class LivenessTFuncs : public StmtVisitor<LivenessTFuncs,void> {
  LiveVariables& L;
  llvm::BitVector Live;
  llvm::BitVector KilledAtLeastOnce;
  Stmt* CurrentStmt;
  const CFGBlock* CurrentBlock;
  bool blockPreviouslyProcessed;
  LiveVariablesObserver* Observer;
public:
  LivenessTFuncs(LiveVariables& l, LiveVariablesObserver* A = NULL)
    : L(l), CurrentStmt(NULL), CurrentBlock(NULL),
      blockPreviouslyProcessed(false), Observer(A)
  {
    Live.resize(l.getNumDecls());
    KilledAtLeastOnce.resize(l.getNumDecls());
  }

  void VisitStmt(Stmt* S);
  void VisitDeclRefExpr(DeclRefExpr* DR);
  void VisitBinaryOperator(BinaryOperator* B);
  void VisitAssign(BinaryOperator* B);
  void VisitStmtExpr(StmtExpr* S);
  void VisitDeclStmt(DeclStmt* DS);
  void VisitUnaryOperator(UnaryOperator* U);

  unsigned getIdx(const Decl* D) {
    LiveVariables::VarInfoMap& V = L.getVarInfoMap();
    LiveVariables::VarInfoMap::iterator I = V.find(D);
    assert (I != V.end());
    return I->second.Idx;
  }
  
  bool ProcessBlock(const CFGBlock* B);
  llvm::BitVector* getBlockEntryLiveness(const CFGBlock* B);
  LiveVariables::VarInfo& KillVar(Decl* D);
};

void LivenessTFuncs::VisitStmt(Stmt* S) {
  if (Observer)
    Observer->ObserveStmt(S,L,Live);
    
  // Evaluate the transfer functions for all subexpressions.  Note that
  // each invocation of "Visit" will have a side-effect: "Liveness" and "Kills"
  // will be updated.  
  for (Stmt::child_iterator I = S->child_begin(),E = S->child_end(); I != E;++I)
    Visit(*I);
}

void LivenessTFuncs::VisitDeclRefExpr(DeclRefExpr* DR) {
  if (Observer)
    Observer->ObserveStmt(DR,L,Live);
    
  // Register a use of the variable.
  Live.set(getIdx(DR->getDecl()));
}

void LivenessTFuncs::VisitStmtExpr(StmtExpr* S) {
  // Do nothing.  The substatements of S are segmented into separate
  // statements in the CFG.
}
  
void LivenessTFuncs::VisitBinaryOperator(BinaryOperator* B) {
  switch (B->getOpcode()) {
    case BinaryOperator::LAnd:
    case BinaryOperator::LOr:
    case BinaryOperator::Comma:
      // Do nothing.  These operations are broken up into multiple
      // statements in the CFG.  All these expressions do is return
      // the value of their subexpressions, but these subexpressions will
      // be evalualated elsewhere in the CFG.
      break;
      
    // FIXME: handle '++' and '--'
    default:        
      if (B->isAssignmentOp()) VisitAssign(B);
      else VisitStmt(B);
  }
}

void LivenessTFuncs::VisitUnaryOperator(UnaryOperator* U) {
  switch (U->getOpcode()) {
    case UnaryOperator::PostInc:
    case UnaryOperator::PostDec:
    case UnaryOperator::PreInc:
    case UnaryOperator::PreDec:
        case UnaryOperator::AddrOf:
      // Walk through the subexpressions, blasting through ParenExprs until
      // we either find a DeclRefExpr or some non-DeclRefExpr expression.
      for (Stmt* S = U->getSubExpr() ; ; ) {
        if (ParenExpr* P = dyn_cast<ParenExpr>(S)) {
          S = P->getSubExpr();
          continue;
        }
        else if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(S)) {
          // Treat the --/++/& operator as a kill.
          LiveVariables::VarInfo& V = KillVar(DR->getDecl());

          if (!blockPreviouslyProcessed)
            V.AddKill(CurrentStmt,DR); 
        
          VisitDeclRefExpr(DR);          
        }
        else
          Visit(S);
          
        break;                  
      }        
      break;      
    
    default:
      VisitStmt(U->getSubExpr());
      break;
  }
}

LiveVariables::VarInfo& LivenessTFuncs::KillVar(Decl* D) {
  LiveVariables::VarInfoMap::iterator I =  L.getVarInfoMap().find(D);
  
  assert (I != L.getVarInfoMap().end() && 
          "Declaration not managed by variable map in LiveVariables");
  
  // Mark the variable dead, and remove the current block from
  // the set of blocks where the variable may be alive the entire time.
  Live.reset(I->second.Idx);
  I->second.V.AliveBlocks.reset(CurrentBlock->getBlockID());
  
  return I->second.V;
}  

void LivenessTFuncs::VisitAssign(BinaryOperator* B) {
  if (Observer)
    Observer->ObserveStmt(B,L,Live);
    
  // Check if we are assigning to a variable.
  Stmt* LHS = B->getLHS();
  
  if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(LHS)) {
    LiveVariables::VarInfo& V = KillVar(DR->getDecl());
    
    // We only need to register kills once, so we check if this block
    // has been previously processed.
    if (!blockPreviouslyProcessed)
      V.AddKill(CurrentStmt,DR);
      
    if (B->getOpcode() != BinaryOperator::Assign)
      Visit(LHS);
  }
  else
    Visit(LHS);
  
  Visit(B->getRHS());    
}

void LivenessTFuncs::VisitDeclStmt(DeclStmt* DS) {
  if (Observer)
    Observer->ObserveStmt(DS,L,Live);
    
  // Declarations effectively "kill" a variable since they cannot possibly
  // be live before they are declared.  Declarations, however, are not kills
  // in the sense that the value is obliterated, so we do not register
  // DeclStmts as a "kill site" for a variable.
  for (Decl* D = DS->getDecl(); D != NULL ; D = D->getNextDeclarator())
    KillVar(D);
}

llvm::BitVector* LivenessTFuncs::getBlockEntryLiveness(const CFGBlock* B) {
  LiveVariables::BlockLivenessMap& BMap = L.getLiveAtBlockEntryMap();

  LiveVariables::BlockLivenessMap::iterator I = BMap.find(B);
  return (I == BMap.end()) ? NULL : &(I->second);  
}

bool LivenessTFuncs::ProcessBlock(const CFGBlock* B) {

  CurrentBlock = B;
  Live.reset();
  KilledAtLeastOnce.reset();
  
  // Check if this block has been previously processed.
  LiveVariables::BlockLivenessMap& BMap = L.getLiveAtBlockEntryMap();
  LiveVariables::BlockLivenessMap::iterator BI = BMap.find(B);
    
  blockPreviouslyProcessed = BI != BMap.end();
  
  // Merge liveness information from all predecessors.
  for (CFGBlock::const_succ_iterator I=B->succ_begin(),E=B->succ_end();I!=E;++I)
    if (llvm::BitVector* V = getBlockEntryLiveness(*I))    
      Live |= *V;

  if (Observer)
    Observer->ObserveBlockExit(B,L,Live);
      
  // Tentatively mark all variables alive at the end of the current block
  // as being alive during the whole block.  We then cull these out as
  // we process the statements of this block.
  for (LiveVariables::VarInfoMap::iterator
         I=L.getVarInfoMap().begin(), E=L.getVarInfoMap().end(); I != E; ++I)
    if (Live[I->second.Idx])
      I->second.V.AliveBlocks.set(B->getBlockID());                              
  
  // March up the statements and process the transfer functions.
  for (CFGBlock::const_reverse_iterator I=B->rbegin(), E=B->rend(); I!=E; ++I) {
    CurrentStmt = *I;
    Visit(CurrentStmt);    
  }

  // Compare the computed "Live" values with what we already have
  // for the entry to this block.
  bool hasChanged = false;

  
  if (!blockPreviouslyProcessed) {
    // We have not previously calculated liveness information for this block.
    // Lazily instantiate a bitvector, and copy the bits from Live.
    hasChanged = true;
    llvm::BitVector& V = BMap[B];
    V.resize(L.getNumDecls());
    V = Live;
  }
  else if (BI->second != Live) {
    hasChanged = true;
    BI->second = Live;
  }
  
  return hasChanged;
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// runOnCFG - Method to run the actual liveness computation.
//

void LiveVariables::runOnCFG(const CFG& cfg, LiveVariablesObserver* Observer) {
  // Scan a CFG for DeclRefStmts.  For each one, create a VarInfo object.
  {
    RegisterDecls R(*this,cfg);
    R.RegisterUsedDecls();
  }
  
  // Create the worklist and enqueue the exit block.
  WorkListTy WorkList;
  WorkList.enqueue(&cfg.getExit());
  
  // Create the state for transfer functions.
  LivenessTFuncs TF(*this,Observer);
  
  // Process the worklist until it is empty.
  
  while (!WorkList.isEmpty()) {
    const CFGBlock* B = WorkList.dequeue();
    if (TF.ProcessBlock(B))
      for (CFGBlock::const_pred_iterator I = B->pred_begin(), E = B->pred_end();
           I != E; ++I)
        WorkList.enqueue(*I);    
  }
  
  // Go through each block and reserve a bitvector.  This is needed if
  // a block was never visited by the worklist algorithm.
  for (CFG::const_iterator I = cfg.begin(), E = cfg.end(); I != E; ++I)
    LiveAtBlockEntryMap[&(*I)].resize(NumDecls);        
}


void LiveVariables::runOnBlock(const CFGBlock* B,
                               LiveVariablesObserver* Observer)
{
  LivenessTFuncs TF(*this,Observer);
  TF.ProcessBlock(B);
}

//===----------------------------------------------------------------------===//
// liveness queries
//

bool LiveVariables::isLive(const CFGBlock* B, const Decl* D) const {
  BlockLivenessMap::const_iterator I = LiveAtBlockEntryMap.find(B);
  assert (I != LiveAtBlockEntryMap.end());
  
  VarInfoMap::const_iterator VI = VarInfos.find(D);
  assert (VI != VarInfos.end());
  
  return I->second[VI->second.Idx];
}

bool LiveVariables::isLive(llvm::BitVector& Live, const Decl* D) const {
  VarInfoMap::const_iterator VI = VarInfos.find(D);
  assert (VI != VarInfos.end());
  return Live[VI->second.Idx];
}

bool LiveVariables::KillsVar(const Stmt* S, const Decl* D) const {
  VarInfoMap::const_iterator VI = VarInfos.find(D);
  assert (VI != VarInfos.end());
  
  for (VarInfo::KillsSet::const_iterator
         I = VI->second.V.Kills.begin(), E = VI->second.V.Kills.end(); I!=E;++I)
    if (I->first == S)
      return true;
      
  return false;        
}

LiveVariables::VarInfo& LiveVariables::getVarInfo(const Decl* D) {
  VarInfoMap::iterator VI = VarInfos.find(D);
  assert (VI != VarInfos.end());
  return VI->second.V;
}

const LiveVariables::VarInfo& LiveVariables::getVarInfo(const Decl* D) const {
  return const_cast<LiveVariables*>(this)->getVarInfo(D);
}

//===----------------------------------------------------------------------===//
// Defaults for LiveVariablesObserver

void LiveVariablesObserver::ObserveStmt(Stmt* S, LiveVariables& L,
                                        llvm::BitVector& V) {}

void LiveVariablesObserver::ObserveBlockExit(const CFGBlock* B,
                                             LiveVariables& L,
                                             llvm::BitVector& V) {}
                            
//===----------------------------------------------------------------------===//
// printing liveness state for debugging
//

void LiveVariables::dumpLiveness(const llvm::BitVector& V,
                                 SourceManager& SM) const {

  for (VarInfoMap::iterator I = VarInfos.begin(), E=VarInfos.end(); I!=E; ++I) {
    if (V[I->second.Idx]) {
      
      SourceLocation PhysLoc = SM.getPhysicalLoc(I->first->getLocation());

      fprintf(stderr, "  %s <%s:%u:%u>\n", 
              I->first->getIdentifier()->getName(),
              SM.getSourceName(PhysLoc),
              SM.getLineNumber(PhysLoc),
              SM.getColumnNumber(PhysLoc));
    }
  }  
}                                  

void LiveVariables::dumpBlockLiveness(SourceManager& M) const {
  for (BlockLivenessMap::iterator I = LiveAtBlockEntryMap.begin(),
                                  E = LiveAtBlockEntryMap.end();
       I != E; ++I) {

    fprintf(stderr,
            "\n[ B%d (live variables at block entry) ]\n",
            I->first->getBlockID());
            
    dumpLiveness(I->second,M);
  }
}