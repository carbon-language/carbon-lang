//===- LoadValueNumbering.cpp - Load Value #'ing Implementation -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements a value numbering pass that value numbers load and call
// instructions.  To do this, it finds lexically identical load instructions,
// and uses alias analysis to determine which loads are guaranteed to produce
// the same value.  To value number call instructions, it looks for calls to
// functions that do not write to memory which do not have intervening
// instructions that clobber the memory that is read from.
//
// This pass builds off of another value numbering pass to implement value
// numbering for non-load and non-call instructions.  It uses Alias Analysis so
// that it can disambiguate the load instructions.  The more powerful these base
// analyses are, the more powerful the resultant value numbering will be.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoadValueNumbering.h"
#include "llvm/Function.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/Analysis/ValueNumbering.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/CFG.h"
#include "llvm/Target/TargetData.h"
#include <set>
using namespace llvm;

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

    /// getCallEqualNumberNodes - Given a call instruction, find other calls
    /// that have the same value number.
    void getCallEqualNumberNodes(CallInst *CI,
                                 std::vector<Value*> &RetVals) const;
  };

  // Register this pass...
  RegisterOpt<LoadVN> X("load-vn", "Load Value Numbering");

  // Declare that we implement the ValueNumbering interface
  RegisterAnalysisGroup<ValueNumbering, LoadVN> Y;
}

Pass *llvm::createLoadValueNumberingPass() { return new LoadVN(); }


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

static bool isPathTransparentTo(BasicBlock *CurBlock, BasicBlock *Dom,
                                Value *Ptr, unsigned Size, AliasAnalysis &AA,
                                std::set<BasicBlock*> &Visited,
                                std::map<BasicBlock*, bool> &TransparentBlocks){
  // If we have already checked out this path, or if we reached our destination,
  // stop searching, returning success.
  if (CurBlock == Dom || !Visited.insert(CurBlock).second)
    return true;
  
  // Check whether this block is known transparent or not.
  std::map<BasicBlock*, bool>::iterator TBI =
    TransparentBlocks.lower_bound(CurBlock);

  if (TBI == TransparentBlocks.end() || TBI->first != CurBlock) {
    // If this basic block can modify the memory location, then the path is not
    // transparent!
    if (AA.canBasicBlockModify(*CurBlock, Ptr, Size)) {
      TransparentBlocks.insert(TBI, std::make_pair(CurBlock, false));
      return false;
    }
      TransparentBlocks.insert(TBI, std::make_pair(CurBlock, true));
  } else if (!TBI->second)
    // This block is known non-transparent, so that path can't be either.
    return false;
  
  // The current block is known to be transparent.  The entire path is
  // transparent if all of the predecessors paths to the parent is also
  // transparent to the memory location.
  for (pred_iterator PI = pred_begin(CurBlock), E = pred_end(CurBlock);
       PI != E; ++PI)
    if (!isPathTransparentTo(*PI, Dom, Ptr, Size, AA, Visited,
                             TransparentBlocks))
      return false;
  return true;
}

/// getCallEqualNumberNodes - Given a call instruction, find other calls that
/// have the same value number.
void LoadVN::getCallEqualNumberNodes(CallInst *CI,
                                     std::vector<Value*> &RetVals) const {
  Function *CF = CI->getCalledFunction();
  if (CF == 0) return;  // Indirect call.
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  if (!AA.onlyReadsMemory(CF)) return;  // Nothing we can do.

  // Scan all of the arguments of the function, looking for one that is not
  // global.  In particular, we would prefer to have an argument or instruction
  // operand to chase the def-use chains of.
  Value *Op = CF;
  for (unsigned i = 1, e = CI->getNumOperands(); i != e; ++i)
    if (isa<Argument>(CI->getOperand(i)) ||
        isa<Instruction>(CI->getOperand(i))) {
      Op = CI->getOperand(i);
      break;
    }

  // Identify all lexically identical calls in this function.
  std::vector<CallInst*> IdenticalCalls;

  Function *CIFunc = CI->getParent()->getParent();
  for (Value::use_iterator UI = Op->use_begin(), E = Op->use_end(); UI != E;
       ++UI)
    if (CallInst *C = dyn_cast<CallInst>(*UI))
      if (C->getNumOperands() == CI->getNumOperands() &&
          C->getOperand(0) == CI->getOperand(0) &&
          C->getParent()->getParent() == CIFunc && C != CI) {
        bool AllOperandsEqual = true;
        for (unsigned i = 1, e = CI->getNumOperands(); i != e; ++i)
          if (C->getOperand(i) != CI->getOperand(i)) {
            AllOperandsEqual = false;
            break;
          }

        if (AllOperandsEqual)
          IdenticalCalls.push_back(C);
      }
  
  if (IdenticalCalls.empty()) return;

  // Eliminate duplicates, which could occur if we chose a value that is passed
  // into a call site multiple times.
  std::sort(IdenticalCalls.begin(), IdenticalCalls.end());
  IdenticalCalls.erase(std::unique(IdenticalCalls.begin(),IdenticalCalls.end()),
                       IdenticalCalls.end());

  // If the call reads memory, we must make sure that there are no stores
  // between the calls in question.
  //
  // FIXME: This should use mod/ref information.  What we really care about it
  // whether an intervening instruction could modify memory that is read, not
  // ANY memory.
  //
  if (!AA.doesNotAccessMemory(CF)) {
    DominatorSet &DomSetInfo = getAnalysis<DominatorSet>();
    BasicBlock *CIBB = CI->getParent();
    for (unsigned i = 0; i != IdenticalCalls.size(); ++i) {
      CallInst *C = IdenticalCalls[i];
      bool CantEqual = false;

      if (DomSetInfo.dominates(CIBB, C->getParent())) {
        // FIXME: we currently only handle the case where both calls are in the
        // same basic block.
        if (CIBB != C->getParent()) {
          CantEqual = true;
        } else {
          Instruction *First = CI, *Second = C;
          if (!DomSetInfo.dominates(CI, C))
            std::swap(First, Second);
          
          // Scan the instructions between the calls, checking for stores or
          // calls to dangerous functions.
          BasicBlock::iterator I = First;
          for (++First; I != BasicBlock::iterator(Second); ++I) {
            if (isa<StoreInst>(I)) {
              // FIXME: We could use mod/ref information to make this much
              // better!
              CantEqual = true;
              break;
            } else if (CallInst *CI = dyn_cast<CallInst>(I)) {
              if (CI->getCalledFunction() == 0 ||
                  !AA.onlyReadsMemory(CI->getCalledFunction())) {
                CantEqual = true;
                break;
              }
            } else if (I->mayWriteToMemory()) {
              CantEqual = true;
              break;
            }
          }
        }

      } else if (DomSetInfo.dominates(C->getParent(), CIBB)) {
        // FIXME: We could implement this, but we don't for now.
        CantEqual = true;
      } else {
        // FIXME: if one doesn't dominate the other, we can't tell yet.
        CantEqual = true;
      }


      if (CantEqual) {
        // This call does not produce the same value as the one in the query.
        std::swap(IdenticalCalls[i--], IdenticalCalls.back());
        IdenticalCalls.pop_back();
      }
    }
  }

  // Any calls that are identical and not destroyed will produce equal values!
  for (unsigned i = 0, e = IdenticalCalls.size(); i != e; ++i)
    RetVals.push_back(IdenticalCalls[i]);
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

  if (!isa<LoadInst>(V)) {
    if (CallInst *CI = dyn_cast<CallInst>(V))
      return getCallEqualNumberNodes(CI, RetVals);

    // Not a load instruction?  Just chain to the base value numbering
    // implementation to satisfy the request...
    assert(&getAnalysis<ValueNumbering>() != (ValueNumbering*)this &&
           "getAnalysis() returned this!");

    return getAnalysis<ValueNumbering>().getEqualNumberNodes(V, RetVals);
  }

  // Volatile loads cannot be replaced with the value of other loads.
  LoadInst *LI = cast<LoadInst>(V);
  if (LI->isVolatile())
    return getAnalysis<ValueNumbering>().getEqualNumberNodes(V, RetVals);
  
  // If we have a load instruction, find all of the load and store instructions
  // that use the same source operand.  We implement this recursively, because
  // there could be a load of a load of a load that are all identical.  We are
  // guaranteed that this cannot be an infinite recursion because load
  // instructions would have to pass through a PHI node in order for there to be
  // a cycle.  The PHI node would be handled by the else case here, breaking the
  // infinite recursion.
  //
  std::vector<Value*> PointerSources;
  getEqualNumberNodes(LI->getOperand(0), PointerSources);
  PointerSources.push_back(LI->getOperand(0));
  
  BasicBlock *LoadBB = LI->getParent();
  Function *F = LoadBB->getParent();
  
  // Now that we know the set of equivalent source pointers for the load
  // instruction, look to see if there are any load or store candidates that are
  // identical.
  //
  std::map<BasicBlock*, std::vector<LoadInst*> >  CandidateLoads;
  std::map<BasicBlock*, std::vector<StoreInst*> > CandidateStores;
  
  while (!PointerSources.empty()) {
    Value *Source = PointerSources.back();
    PointerSources.pop_back();                // Get a source pointer...
    
    for (Value::use_iterator UI = Source->use_begin(), UE = Source->use_end();
         UI != UE; ++UI)
      if (LoadInst *Cand = dyn_cast<LoadInst>(*UI)) {// Is a load of source?
        if (Cand->getParent()->getParent() == F &&   // In the same function?
            Cand != LI && !Cand->isVolatile())       // Not LI itself?
          CandidateLoads[Cand->getParent()].push_back(Cand);     // Got one...
      } else if (StoreInst *Cand = dyn_cast<StoreInst>(*UI)) {
        if (Cand->getParent()->getParent() == F && !Cand->isVolatile() &&
            Cand->getOperand(1) == Source)  // It's a store THROUGH the ptr...
          CandidateStores[Cand->getParent()].push_back(Cand);
      }
  }
  
  // Get alias analysis & dominators.
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  DominatorSet &DomSetInfo = getAnalysis<DominatorSet>();
  Value *LoadPtr = LI->getOperand(0);
  // Find out how many bytes of memory are loaded by the load instruction...
  unsigned LoadSize = getAnalysis<TargetData>().getTypeSize(LI->getType());

  // Find all of the candidate loads and stores that are in the same block as
  // the defining instruction.
  std::set<Instruction*> Instrs;
  Instrs.insert(CandidateLoads[LoadBB].begin(), CandidateLoads[LoadBB].end());
  CandidateLoads.erase(LoadBB);
  Instrs.insert(CandidateStores[LoadBB].begin(), CandidateStores[LoadBB].end());
  CandidateStores.erase(LoadBB);

  // Figure out if the load is invalidated from the entry of the block it is in
  // until the actual instruction.  This scans the block backwards from LI.  If
  // we see any candidate load or store instructions, then we know that the
  // candidates have the same value # as LI.
  bool LoadInvalidatedInBBBefore = false;
  for (BasicBlock::iterator I = LI; I != LoadBB->begin(); ) {
    --I;
    // If this instruction is a candidate load before LI, we know there are no
    // invalidating instructions between it and LI, so they have the same value
    // number.
    if (isa<LoadInst>(I) && Instrs.count(I)) {
      RetVals.push_back(I);
      Instrs.erase(I);
    }

    if (AA.getModRefInfo(I, LoadPtr, LoadSize) & AliasAnalysis::Mod) {
      // If the invalidating instruction is a store, and its in our candidate
      // set, then we can do store-load forwarding: the load has the same value
      // # as the stored value.
      if (isa<StoreInst>(I) && Instrs.count(I)) {
        Instrs.erase(I);
        RetVals.push_back(I->getOperand(0));
      }

      LoadInvalidatedInBBBefore = true;
      break;
    }
  }

  // Figure out if the load is invalidated between the load and the exit of the
  // block it is defined in.  While we are scanning the current basic block, if
  // we see any candidate loads, then we know they have the same value # as LI.
  //
  bool LoadInvalidatedInBBAfter = false;
  for (BasicBlock::iterator I = LI->getNext(); I != LoadBB->end(); ++I) {
    // If this instruction is a load, then this instruction returns the same
    // value as LI.
    if (isa<LoadInst>(I) && Instrs.count(I)) {
      RetVals.push_back(I);
      Instrs.erase(I);
    }

    if (AA.getModRefInfo(I, LoadPtr, LoadSize) & AliasAnalysis::Mod) {
      LoadInvalidatedInBBAfter = true;
      break;
    }
  }

  // If there is anything left in the Instrs set, it could not possibly equal
  // LI.
  Instrs.clear();

  // TransparentBlocks - For each basic block the load/store is alive across,
  // figure out if the pointer is invalidated or not.  If it is invalidated, the
  // boolean is set to false, if it's not it is set to true.  If we don't know
  // yet, the entry is not in the map.
  std::map<BasicBlock*, bool> TransparentBlocks;

  // Loop over all of the basic blocks that also load the value.  If the value
  // is live across the CFG from the source to destination blocks, and if the
  // value is not invalidated in either the source or destination blocks, add it
  // to the equivalence sets.
  for (std::map<BasicBlock*, std::vector<LoadInst*> >::iterator
         I = CandidateLoads.begin(), E = CandidateLoads.end(); I != E; ++I) {
    bool CantEqual = false;

    // Right now we only can handle cases where one load dominates the other.
    // FIXME: generalize this!
    BasicBlock *BB1 = I->first, *BB2 = LoadBB;
    if (DomSetInfo.dominates(BB1, BB2)) {
      // The other load dominates LI.  If the loaded value is killed entering
      // the LoadBB block, we know the load is not live.
      if (LoadInvalidatedInBBBefore)
        CantEqual = true;
    } else if (DomSetInfo.dominates(BB2, BB1)) {
      std::swap(BB1, BB2);          // Canonicalize
      // LI dominates the other load.  If the loaded value is killed exiting
      // the LoadBB block, we know the load is not live.
      if (LoadInvalidatedInBBAfter)
        CantEqual = true;
    } else {
      // None of these loads can VN the same.
      CantEqual = true;
    }

    if (!CantEqual) {
      // Ok, at this point, we know that BB1 dominates BB2, and that there is
      // nothing in the LI block that kills the loaded value.  Check to see if
      // the value is live across the CFG.
      std::set<BasicBlock*> Visited;
      for (pred_iterator PI = pred_begin(BB2), E = pred_end(BB2); PI!=E; ++PI)
        if (!isPathTransparentTo(*PI, BB1, LoadPtr, LoadSize, AA,
                                 Visited, TransparentBlocks)) {
          // None of these loads can VN the same.
          CantEqual = true;
          break;
        }
    }

    // If the loads can equal so far, scan the basic block that contains the
    // loads under consideration to see if they are invalidated in the block.
    // For any loads that are not invalidated, add them to the equivalence
    // set!
    if (!CantEqual) {
      Instrs.insert(I->second.begin(), I->second.end());
      if (BB1 == LoadBB) {
        // If LI dominates the block in question, check to see if any of the
        // loads in this block are invalidated before they are reached.
        for (BasicBlock::iterator BBI = I->first->begin(); ; ++BBI) {
          if (isa<LoadInst>(BBI) && Instrs.count(BBI)) {
            // The load is in the set!
            RetVals.push_back(BBI);
            Instrs.erase(BBI);
            if (Instrs.empty()) break;
          } else if (AA.getModRefInfo(BBI, LoadPtr, LoadSize)
                             & AliasAnalysis::Mod) {
            // If there is a modifying instruction, nothing below it will value
            // # the same.
            break;
          }
        }
      } else {
        // If the block dominates LI, make sure that the loads in the block are
        // not invalidated before the block ends.
        BasicBlock::iterator BBI = I->first->end();
        while (1) {
          --BBI;
          if (isa<LoadInst>(BBI) && Instrs.count(BBI)) {
            // The load is in the set!
            RetVals.push_back(BBI);
            Instrs.erase(BBI);
            if (Instrs.empty()) break;
          } else if (AA.getModRefInfo(BBI, LoadPtr, LoadSize)
                             & AliasAnalysis::Mod) {
            // If there is a modifying instruction, nothing above it will value
            // # the same.
            break;
          }
        }
      }

      Instrs.clear();
    }
  }

  // Handle candidate stores.  If the loaded location is clobbered on entrance
  // to the LoadBB, no store outside of the LoadBB can value number equal, so
  // quick exit.
  if (LoadInvalidatedInBBBefore)
    return;

  for (std::map<BasicBlock*, std::vector<StoreInst*> >::iterator
         I = CandidateStores.begin(), E = CandidateStores.end(); I != E; ++I)
    if (DomSetInfo.dominates(I->first, LoadBB)) {
      // Check to see if the path from the store to the load is transparent
      // w.r.t. the memory location.
      bool CantEqual = false;
      std::set<BasicBlock*> Visited;
      for (pred_iterator PI = pred_begin(LoadBB), E = pred_end(LoadBB);
           PI != E; ++PI)
        if (!isPathTransparentTo(*PI, I->first, LoadPtr, LoadSize, AA,
                                 Visited, TransparentBlocks)) {
          // None of these stores can VN the same.
          CantEqual = true;
          break;
        }
      Visited.clear();
      if (!CantEqual) {
        // Okay, the path from the store block to the load block is clear, and
        // we know that there are no invalidating instructions from the start
        // of the load block to the load itself.  Now we just scan the store
        // block.

        BasicBlock::iterator BBI = I->first->end();
        while (1) {
          --BBI;
          if (AA.getModRefInfo(BBI, LoadPtr, LoadSize)& AliasAnalysis::Mod){
            // If the invalidating instruction is one of the candidates,
            // then it provides the value the load loads.
            if (StoreInst *SI = dyn_cast<StoreInst>(BBI))
              if (std::find(I->second.begin(), I->second.end(), SI) !=
                  I->second.end())
                RetVals.push_back(SI->getOperand(0));
            break;
          }
        }
      }
    }
}
