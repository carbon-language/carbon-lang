//==- Dominators.cpp - Construct the Dominance Tree Given CFG ----*- C++ --*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple, fast dominance algorithm for source-level CFGs.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/Dominators.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"

using namespace clang;

DominatorTree::~DominatorTree() {
  IDoms.clear();
  RootNode = 0;
}

CFGBlock * DominatorTree::getNode(const CFGBlock *B) const {
  CFGBlockMapTy::const_iterator I = IDoms.find(B);
  return I != IDoms.end() ? I->second : 0; 
}

bool DominatorTree::properlyDominates(const CFGBlock *A, 
                                      const CFGBlock *B) const {
  if (0 == A || 0 == B || A == B)
    return false;

  // The EntryBlock dominates every other block.
  if (A == RootNode)
    return true;

  // Note: The dominator of the EntryBlock is itself.
  CFGBlock *IDom = getNode(B);
  while (IDom != A && IDom != RootNode)
    IDom = getNode(IDom);

  return IDom != RootNode;
}

bool DominatorTree::dominates(const CFGBlock *A,
                              const CFGBlock *B) const {
  if (A == B)
    return true;

  return properlyDominates(A, B);
}

const CFGBlock * DominatorTree::findNearestCommonDominator
      (const CFGBlock *A, const CFGBlock *B) const {
  //If A dominates B, then A is the nearest common dominator
  if (dominates(A, B))
    return A;

  //If B dominates A, then B is the nearest common dominator
  if (dominates(B, A))
    return B;

  //Collect all A's dominators
  llvm::SmallPtrSet<CFGBlock *, 16> ADoms;
  ADoms.insert(RootNode);
  CFGBlock *ADom = getNode(A);
  while (ADom != RootNode) {
    ADoms.insert(ADom);
    ADom = getNode(ADom);
  }  

  //Check all B's dominators against ADoms
  CFGBlock *BDom = getNode(B);
  while (BDom != RootNode){
    if (ADoms.count(BDom) != 0)
      return BDom;

    BDom = getNode(BDom);
  }

  //The RootNode dominates every other node
  return RootNode;
}

/// Constructs immediate dominator tree for a given CFG based on the algorithm
/// described in this paper:
///
///  A Simple, Fast Dominance Algorithm
///  Keith D. Cooper, Timothy J. Harvey and Ken Kennedy
///  Software-Practice and Expreience, 2001;4:1-10.
///
/// This implementation is simple and runs faster in practice than the classis
/// Lengauer-Tarjan algorithm. For detailed discussions, refer to the paper. 
void DominatorTree::BuildDominatorTree() {
  CFG *cfg = AC.getCFG();
  CFGBlock *EntryBlk = &cfg->getEntry();

  //Sort all CFGBlocks in reverse order
  PostOrderCFGView *rpocfg = AC.getAnalysis<PostOrderCFGView>();

  //Set the root of the dominance tree
  RootNode = EntryBlk;
  
  //Compute the immediate dominator for each CFGBlock
  IDoms[EntryBlk] = EntryBlk;
  bool changed = true;
  while (changed){
    changed = false;

    for (PostOrderCFGView::iterator I = rpocfg->begin(),
        E = rpocfg->end(); I != E; ++I){
      if (EntryBlk == *I)
        continue;
      if (const CFGBlock *B = *I) {
        //Compute immediate dominance information for CFGBlock B
        CFGBlock *IDom = 0;
        for (CFGBlock::const_pred_iterator J = B->pred_begin(),
            K = B->pred_end(); J != K; ++J)
          if( CFGBlock *P = *J) {
            if (IDoms.find(P) == IDoms.end())
              continue;
            if (!IDom)
              IDom = P;
            else {
              //intersect IDom and P
              CFGBlock *B1 = IDom, *B2 = P;
              while (B1 != B2) {
                while ((rpocfg->getComparator())(B2,B1))
                  B1 = IDoms[B1];
                while ((rpocfg->getComparator())(B1,B2))
                  B2 = IDoms[B2];
              }
              IDom = B1;
            }
          }
        if (IDoms[B] != IDom) {
          IDoms[B] = IDom;
          changed = true;
        } 
      }
    }
  }//while
}

void DominatorTree::dump() {
  CFG *cfg = AC.getCFG();

  llvm::errs() << "Immediate dominance tree (Node#,IDom#):\n";
  for (CFG::const_iterator I = cfg->begin(),
      E = cfg->end(); I != E; ++I) {
    assert(IDoms[(*I)] && 
       "Failed to find the immediate dominator for all CFG blocks.");
    llvm::errs() << "(" << (*I)->getBlockID()
                 << "," << IDoms[(*I)]->getBlockID() << ")\n";
  }
}

