//===- LoadValueNumbering.cpp - Load Value #'ing Implementation -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements a value numbering pass that value #'s load instructions.
// To do this, it finds lexically identical load instructions, and uses alias
// analysis to determine which loads are guaranteed to produce the same value.
//
// This pass builds off of another value numbering pass to implement value
// numbering for non-load instructions.  It uses Alias Analysis so that it can
// disambiguate the load instructions.  The more powerful these base analyses
// are, the more powerful the resultant analysis will be.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoadValueNumbering.h"
#include "llvm/Analysis/ValueNumbering.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/iMemory.h"
#include "llvm/BasicBlock.h"
#include "llvm/Support/CFG.h"
#include <algorithm>
#include <set>

namespace llvm {

namespace {
  // FIXME: This should not be a FunctionPass.
  struct LoadVN : public FunctionPass, public ValueNumbering {
    
    /// Pass Implementation stuff.  This doesn't do any analysis.
    ///
    bool runOnFunction(Function &) { return false; }
    
    /// getAnalysisUsage - Does not modify anything.  It uses Value Numbering
    /// and Alias Analysis.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    
    /// getEqualNumberNodes - Return nodes with the same value number as the
    /// specified Value.  This fills in the argument vector with any equal
    /// values.
    ///
    virtual void getEqualNumberNodes(Value *V1,
                                     std::vector<Value*> &RetVals) const;
  private:
    /// haveEqualValueNumber - Given two load instructions, determine if they
    /// both produce the same value on every execution of the program, assuming
    /// that their source operands always give the same value.  This uses the
    /// AliasAnalysis implementation to invalidate loads when stores or function
    /// calls occur that could modify the value produced by the load.
    ///
    bool haveEqualValueNumber(LoadInst *LI, LoadInst *LI2, AliasAnalysis &AA,
                              DominatorSet &DomSetInfo) const;
    bool haveEqualValueNumber(LoadInst *LI, StoreInst *SI, AliasAnalysis &AA,
                              DominatorSet &DomSetInfo) const;
  };

  // Register this pass...
  RegisterOpt<LoadVN> X("load-vn", "Load Value Numbering");

  // Declare that we implement the ValueNumbering interface
  RegisterAnalysisGroup<ValueNumbering, LoadVN> Y;
}

Pass *createLoadValueNumberingPass() { return new LoadVN(); }


/// getAnalysisUsage - Does not modify anything.  It uses Value Numbering and
/// Alias Analysis.
///
void LoadVN::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<AliasAnalysis>();
  AU.addRequired<ValueNumbering>();
  AU.addRequired<DominatorSet>();
  AU.addRequired<TargetData>();
}

// getEqualNumberNodes - Return nodes with the same value number as the
// specified Value.  This fills in the argument vector with any equal values.
//
void LoadVN::getEqualNumberNodes(Value *V,
                                 std::vector<Value*> &RetVals) const {
  // If the alias analysis has any must alias information to share with us, we
  // can definitely use it.
  if (isa<PointerType>(V->getType()))
    getAnalysis<AliasAnalysis>().getMustAliases(V, RetVals);

  if (LoadInst *LI = dyn_cast<LoadInst>(V)) {
    // Volatile loads cannot be replaced with the value of other loads.
    if (LI->isVolatile())
      return getAnalysis<ValueNumbering>().getEqualNumberNodes(V, RetVals);

    // If we have a load instruction, find all of the load and store
    // instructions that use the same source operand.  We implement this
    // recursively, because there could be a load of a load of a load that are
    // all identical.  We are guaranteed that this cannot be an infinite
    // recursion because load instructions would have to pass through a PHI node
    // in order for there to be a cycle.  The PHI node would be handled by the
    // else case here, breaking the infinite recursion.
    //
    std::vector<Value*> PointerSources;
    getEqualNumberNodes(LI->getOperand(0), PointerSources);
    PointerSources.push_back(LI->getOperand(0));

    Function *F = LI->getParent()->getParent();

    // Now that we know the set of equivalent source pointers for the load
    // instruction, look to see if there are any load or store candidates that
    // are identical.
    //
    std::vector<LoadInst*> CandidateLoads;
    std::vector<StoreInst*> CandidateStores;

    while (!PointerSources.empty()) {
      Value *Source = PointerSources.back();
      PointerSources.pop_back();                // Get a source pointer...

      for (Value::use_iterator UI = Source->use_begin(), UE = Source->use_end();
           UI != UE; ++UI)
        if (LoadInst *Cand = dyn_cast<LoadInst>(*UI)) {// Is a load of source?
          if (Cand->getParent()->getParent() == F &&   // In the same function?
              Cand != LI && !Cand->isVolatile())       // Not LI itself?
            CandidateLoads.push_back(Cand);     // Got one...
        } else if (StoreInst *Cand = dyn_cast<StoreInst>(*UI)) {
          if (Cand->getParent()->getParent() == F && !Cand->isVolatile() &&
              Cand->getOperand(1) == Source)  // It's a store THROUGH the ptr...
            CandidateStores.push_back(Cand);
        }
    }

    // Remove duplicates from the CandidateLoads list because alias analysis
    // processing may be somewhat expensive and we don't want to do more work
    // than necessary.
    //
    unsigned OldSize = CandidateLoads.size();
    std::sort(CandidateLoads.begin(), CandidateLoads.end());
    CandidateLoads.erase(std::unique(CandidateLoads.begin(),
                                     CandidateLoads.end()),
                         CandidateLoads.end());
    // FIXME: REMOVE THIS SORTING AND UNIQUING IF IT CAN'T HAPPEN
    assert(CandidateLoads.size() == OldSize && "Shrunk the candloads list?");

    // Get Alias Analysis...
    AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
    DominatorSet &DomSetInfo = getAnalysis<DominatorSet>();
    
    // Loop over all of the candidate loads.  If they are not invalidated by
    // stores or calls between execution of them and LI, then add them to
    // RetVals.
    for (unsigned i = 0, e = CandidateLoads.size(); i != e; ++i)
      if (haveEqualValueNumber(LI, CandidateLoads[i], AA, DomSetInfo))
        RetVals.push_back(CandidateLoads[i]);
    for (unsigned i = 0, e = CandidateStores.size(); i != e; ++i)
      if (haveEqualValueNumber(LI, CandidateStores[i], AA, DomSetInfo))
        RetVals.push_back(CandidateStores[i]->getOperand(0));
      
  } else {
    assert(&getAnalysis<ValueNumbering>() != (ValueNumbering*)this &&
           "getAnalysis() returned this!");

    // Not a load instruction?  Just chain to the base value numbering
    // implementation to satisfy the request...
    return getAnalysis<ValueNumbering>().getEqualNumberNodes(V, RetVals);
  }
}

// CheckForInvalidatingInst - Return true if BB or any of the predecessors of BB
// (until DestBB) contain an instruction that might invalidate Ptr.
//
static bool CheckForInvalidatingInst(BasicBlock *BB, BasicBlock *DestBB,
                                     Value *Ptr, unsigned Size,
                                     AliasAnalysis &AA,
                                     std::set<BasicBlock*> &VisitedSet) {
  // Found the termination point!
  if (BB == DestBB || VisitedSet.count(BB)) return false;

  // Avoid infinite recursion!
  VisitedSet.insert(BB);

  // Can this basic block modify Ptr?
  if (AA.canBasicBlockModify(*BB, Ptr, Size))
    return true;

  // Check all of our predecessor blocks...
  for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI)
    if (CheckForInvalidatingInst(*PI, DestBB, Ptr, Size, AA, VisitedSet))
      return true;

  // None of our predecessor blocks contain an invalidating instruction, and we
  // don't either!
  return false;
}


/// haveEqualValueNumber - Given two load instructions, determine if they both
/// produce the same value on every execution of the program, assuming that
/// their source operands always give the same value.  This uses the
/// AliasAnalysis implementation to invalidate loads when stores or function
/// calls occur that could modify the value produced by the load.
///
bool LoadVN::haveEqualValueNumber(LoadInst *L1, LoadInst *L2,
                                  AliasAnalysis &AA,
                                  DominatorSet &DomSetInfo) const {
  assert(L1 != L2 && "haveEqualValueNumber assumes differing loads!");
  assert(L1->getType() == L2->getType() &&
         "How could the same source pointer return different types?");
  Value *LoadAddress = L1->getOperand(0);

  // Find out how many bytes of memory are loaded by the load instruction...
  unsigned LoadSize = getAnalysis<TargetData>().getTypeSize(L1->getType());

  // If the two loads are in the same basic block, just do a local analysis.
  if (L1->getParent() == L2->getParent()) {
    // It can be _very_ expensive to determine which instruction occurs first in
    // the basic block if the block is large (see PR209).  For this reason,
    // instead of figuring out which block is first, then scanning all of the
    // instructions, we scan the instructions both ways from L1 until we find
    // L2.  Along the way if we find a potentially modifying instruction, we
    // kill the search.  This helps in cases where we have large blocks the have
    // potentially modifying instructions in them which stop the search.

    BasicBlock *BB = L1->getParent();
    BasicBlock::iterator UpIt = L1, DownIt = L1; ++DownIt;
    bool NoModifiesUp = true, NoModifiesDown = true;

    // Scan up and down looking for L2, a modifying instruction, or the end of a
    // basic block.
    while (UpIt != BB->begin() && DownIt != BB->end()) {
      // Scan up...
      --UpIt;
      if (&*UpIt == L2)
        return NoModifiesUp;  // No instructions invalidate the loads!
      if (NoModifiesUp)
        NoModifiesUp &=
          !(AA.getModRefInfo(UpIt, LoadAddress, LoadSize) & AliasAnalysis::Mod);

      if (&*DownIt == L2)
        return NoModifiesDown;
      if (NoModifiesDown)
        NoModifiesDown &=
          !(AA.getModRefInfo(DownIt, LoadAddress, LoadSize)
            & AliasAnalysis::Mod);
      ++DownIt;
    }

    // If we got here, we ran into one end of the basic block or the other.
    if (UpIt != BB->begin()) {
      // If we know that the upward scan found a modifier, return false.
      if (!NoModifiesUp) return false;

      // Otherwise, continue the scan looking for a modifier or L2.
      for (--UpIt; &*UpIt != L2; --UpIt)
        if (AA.getModRefInfo(UpIt, LoadAddress, LoadSize) & AliasAnalysis::Mod)
          return false;
      return true;
    } else {
      // If we know that the downward scan found a modifier, return false.
      assert(DownIt != BB->end() && "Didn't find instructions??");
      if (!NoModifiesDown) return false;
      
      // Otherwise, continue the scan looking for a modifier or L2.
      for (; &*DownIt != L2; ++DownIt) {
        if (AA.getModRefInfo(DownIt, LoadAddress, LoadSize) &AliasAnalysis::Mod)
          return false;
      }
      return true;
    }
  } else {
    // Figure out which load dominates the other one.  If neither dominates the
    // other we cannot eliminate them.
    //
    // FIXME: This could be enhanced greatly!
    //
    if (DomSetInfo.dominates(L2, L1)) 
      std::swap(L1, L2);   // Make L1 dominate L2
    else if (!DomSetInfo.dominates(L1, L2))
      return false;  // Neither instruction dominates the other one...

    BasicBlock *BB1 = L1->getParent(), *BB2 = L2->getParent();
    
    // L1 now dominates L2.  Check to see if the intervening instructions
    // between the two loads might modify the loaded location.

    // Make sure that there are no modifying instructions between L1 and the end
    // of its basic block.
    //
    if (AA.canInstructionRangeModify(*L1, *BB1->getTerminator(), LoadAddress,
                                     LoadSize))
      return false;   // Cannot eliminate load

    // Make sure that there are no modifying instructions between the start of
    // BB2 and the second load instruction.
    //
    if (AA.canInstructionRangeModify(BB2->front(), *L2, LoadAddress, LoadSize))
      return false;   // Cannot eliminate load

    // Do a depth first traversal of the inverse CFG starting at L2's block,
    // looking for L1's block.  The inverse CFG is made up of the predecessor
    // nodes of a block... so all of the edges in the graph are "backward".
    //
    std::set<BasicBlock*> VisitedSet;
    for (pred_iterator PI = pred_begin(BB2), PE = pred_end(BB2); PI != PE; ++PI)
      if (CheckForInvalidatingInst(*PI, BB1, LoadAddress, LoadSize, AA,
                                   VisitedSet))
        return false;
    
    // If we passed all of these checks then we are sure that the two loads
    // produce the same value.
    return true;
  }
}


/// haveEqualValueNumber - Given a load instruction and a store instruction,
/// determine if the stored value reaches the loaded value unambiguously on
/// every execution of the program.  This uses the AliasAnalysis implementation
/// to invalidate the stored value when stores or function calls occur that
/// could modify the value produced by the load.
///
bool LoadVN::haveEqualValueNumber(LoadInst *Load, StoreInst *Store,
                                  AliasAnalysis &AA,
                                  DominatorSet &DomSetInfo) const {
  // If the store does not dominate the load, we cannot do anything...
  if (!DomSetInfo.dominates(Store, Load)) 
    return false;

  BasicBlock *BB1 = Store->getParent(), *BB2 = Load->getParent();
  Value *LoadAddress = Load->getOperand(0);

  assert(LoadAddress->getType() == Store->getOperand(1)->getType() &&
         "How could the same source pointer return different types?");

  // Find out how many bytes of memory are loaded by the load instruction...
  unsigned LoadSize = getAnalysis<TargetData>().getTypeSize(Load->getType());

  // Compute a basic block iterator pointing to the instruction after the store.
  BasicBlock::iterator StoreIt = Store; ++StoreIt;

  // Check to see if the intervening instructions between the two store and load
  // include a store or call...
  //
  if (BB1 == BB2) {  // In same basic block?
    // In this degenerate case, no checking of global basic blocks has to occur
    // just check the instructions BETWEEN Store & Load...
    //
    if (AA.canInstructionRangeModify(*StoreIt, *Load, LoadAddress, LoadSize))
      return false;   // Cannot eliminate load

    // No instructions invalidate the stored value, they produce the same value!
    return true;
  } else {
    // Make sure that there are no store instructions between the Store and the
    // end of its basic block...
    //
    if (AA.canInstructionRangeModify(*StoreIt, *BB1->getTerminator(),
                                     LoadAddress, LoadSize))
      return false;   // Cannot eliminate load

    // Make sure that there are no store instructions between the start of BB2
    // and the second load instruction...
    //
    if (AA.canInstructionRangeModify(BB2->front(), *Load, LoadAddress,LoadSize))
      return false;   // Cannot eliminate load

    // Do a depth first traversal of the inverse CFG starting at L2's block,
    // looking for L1's block.  The inverse CFG is made up of the predecessor
    // nodes of a block... so all of the edges in the graph are "backward".
    //
    std::set<BasicBlock*> VisitedSet;
    for (pred_iterator PI = pred_begin(BB2), PE = pred_end(BB2); PI != PE; ++PI)
      if (CheckForInvalidatingInst(*PI, BB1, LoadAddress, LoadSize, AA,
                                   VisitedSet))
        return false;

    // If we passed all of these checks then we are sure that the two loads
    // produce the same value.
    return true;
  }
}

} // End llvm namespace
