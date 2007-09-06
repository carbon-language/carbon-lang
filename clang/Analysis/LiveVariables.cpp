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
#include "clang/AST/Expr.h"
#include "clang/AST/CFG.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/IdentifierTable.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <iostream>

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
  void Register(Decl* D);
  void RegisterUsedDecls();
};

void RegisterDecls::VisitStmt(Stmt* S) {
  for (Stmt::child_iterator I = S->child_begin(),E = S->child_end(); I != E;++I)
    Visit(*I);
}

void RegisterDecls::VisitDeclRefExpr(DeclRefExpr* DR) {
  for (Decl* D = DR->getDecl() ; D != NULL ; D = D->getNextDeclarator())
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
  llvm::BitVector Killed;
public:
  LivenessTFuncs(LiveVariables& l) : L(l) {
    Live.resize(l.getNumDecls());
    Killed.resize(l.getNumDecls());
  }

  void VisitStmt(Stmt* S);
  void VisitDeclRefExpr(DeclRefExpr* DR);
  void VisitBinaryOperator(BinaryOperator* B);
  void VisitAssign(BinaryOperator* B);
  void VisitStmtExpr(StmtExpr* S);

  unsigned getIdx(const Decl* D) {
    LiveVariables::VarInfoMap& V = L.getVarInfoMap();
    LiveVariables::VarInfoMap::iterator I = V.find(D);
    assert (I != V.end());
    return I->second.Idx;
  }
  
  bool ProcessBlock(const CFGBlock* B);
  llvm::BitVector* getLiveness(const CFGBlock* B);

};

void LivenessTFuncs::VisitStmt(Stmt* S) {
  // Evaluate the transfer functions for all subexpressions.  Note that
  // each invocation of "Visit" will have a side-effect: "Liveness" and "Kills"
  // will be updated.
  for (Stmt::child_iterator I = S->child_begin(),E = S->child_end(); I != E;++I)
    Visit(*I);
}

void LivenessTFuncs::VisitDeclRefExpr(DeclRefExpr* DR) {
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
      // the value of their subexpressions, but these expressions will
      // be evalualated elsewhere in the CFG.
      break;
      
    // FIXME: handle '++' and '--'
    default:
      if (B->isAssignmentOp()) VisitAssign(B);
      else Visit(B);
  }
}


void LivenessTFuncs::VisitAssign(BinaryOperator* B) {
  Stmt* LHS = B->getLHS();

  if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(LHS)) {
    unsigned i = getIdx(DR->getDecl());
    Live.reset(i);
    Killed.set(i);
  }
  else Visit(LHS);
  
  Visit(B->getRHS());    
}


llvm::BitVector* LivenessTFuncs::getLiveness(const CFGBlock* B) {
  LiveVariables::BlockLivenessMap& BMap = L.getLiveAtBlockEntryMap();

  LiveVariables::BlockLivenessMap::iterator I = BMap.find(B);
  return (I == BMap.end()) ? NULL : &(I->second);  
}

bool LivenessTFuncs::ProcessBlock(const CFGBlock* B) {
  // First: merge all predecessors.
  Live.reset();
  Killed.reset();
  
  for (CFGBlock::const_succ_iterator I=B->succ_begin(),E=B->succ_end();I!=E;++I)
    if (llvm::BitVector* V = getLiveness(*I))    
      Live |= *V;
  
  // Second: march up the statements and process the transfer functions.
  for (CFGBlock::const_reverse_iterator I=B->rbegin(), E=B->rend(); I!=E; ++I) {
    Visit(*I);    
  }

  // Third: compare the computed "Live" values with what we already have
  // for this block.
  bool hasChanged = false;
  
  LiveVariables::BlockLivenessMap& BMap = L.getLiveAtBlockEntryMap();
  LiveVariables::BlockLivenessMap::iterator I = BMap.find(B);
  if (I == BMap.end()) {
    hasChanged = true;
    llvm::BitVector& V = BMap[B];
    V.resize(L.getNumDecls());
    V |= Live;
  }
  else if (I->second != Live) {
    hasChanged = true;
    I->second = Live;
  }
  
  return hasChanged;
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// runOnCFG - Method to run the actual liveness computation.
//

void LiveVariables::runOnCFG(const CFG& cfg) {
  // Scan a CFG for DeclRefStmts.  For each one, create a VarInfo object.
  {
    RegisterDecls R(*this,cfg);
    R.RegisterUsedDecls();
  }
  
  // Create the worklist and enqueue the exit block.
  WorkListTy WorkList;
  WorkList.enqueue(&cfg.getExit());
  
  // Create the state for transfer functions.
  LivenessTFuncs TF(*this);
  
  // Process the worklist until it is empty.
  
  while (!WorkList.isEmpty()) {
    const CFGBlock* B = WorkList.dequeue();
    if (TF.ProcessBlock(B))
      for (CFGBlock::const_pred_iterator I = B->pred_begin(), E = B->pred_end();
           I != E; ++I)
        WorkList.enqueue(*I);    
  }
  
  // Go through each block and reserve a bitvector.
  for (CFG::const_iterator I = cfg.begin(), E = cfg.end(); I != E; ++I)
    LiveAtBlockEntryMap[&(*I)].resize(NumDecls);        
}

//===----------------------------------------------------------------------===//
// printing liveness state for debugging
//

void LiveVariables::printLiveness(const llvm::BitVector& V,
                                  std::ostream& OS) const {

  for (VarInfoMap::iterator I = VarInfos.begin(), E=VarInfos.end(); I!=E; ++I) {
    if (V[I->second.Idx]) {
      OS << I->first->getIdentifier()->getName() << "\n";
    }
  }  
}                                  

void LiveVariables::printBlockLiveness(std::ostream& OS) const {
  for (BlockLivenessMap::iterator I = LiveAtBlockEntryMap.begin(),
                                  E = LiveAtBlockEntryMap.end();
       I != E; ++I) {
    OS << "\n[ B" << I->first->getBlockID() 
       << " (live variables at block entry) ]\n";
    printLiveness(I->second, OS);           
  }
}

void LiveVariables::DumpBlockLiveness() const {
  printBlockLiveness(std::cerr);
}