//===- PromoteMemoryToRegister.cpp - Convert allocas to registers ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file promote memory references to be register references.  It promotes
// alloca instructions which only have loads and stores as uses.  An alloca is
// transformed by using dominator frontiers to place PHI nodes, then traversing
// the function in depth-first order to rewrite loads and stores as appropriate.
// This is just the standard SSA construction algorithm to construct "pruned"
// SSA form.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
using namespace llvm;

// Provide DenseMapKeyInfo for all pointers.
namespace llvm {
template<>
struct DenseMapKeyInfo<std::pair<BasicBlock*, unsigned> > {
  static inline std::pair<BasicBlock*, unsigned> getEmptyKey() {
    return std::make_pair((BasicBlock*)-1, ~0U);
  }
  static inline std::pair<BasicBlock*, unsigned> getTombstoneKey() {
    return std::make_pair((BasicBlock*)-2, 0U);
  }
  static unsigned getHashValue(const std::pair<BasicBlock*, unsigned> &Val) {
    return DenseMapKeyInfo<void*>::getHashValue(Val.first) + Val.second*2;
  }
  static bool isPod() { return true; }
};
}

/// isAllocaPromotable - Return true if this alloca is legal for promotion.
/// This is true if there are only loads and stores to the alloca.
///
bool llvm::isAllocaPromotable(const AllocaInst *AI) {
  // FIXME: If the memory unit is of pointer or integer type, we can permit
  // assignments to subsections of the memory unit.

  // Only allow direct loads and stores...
  for (Value::use_const_iterator UI = AI->use_begin(), UE = AI->use_end();
       UI != UE; ++UI)     // Loop over all of the uses of the alloca
    if (isa<LoadInst>(*UI)) {
      // noop
    } else if (const StoreInst *SI = dyn_cast<StoreInst>(*UI)) {
      if (SI->getOperand(0) == AI)
        return false;   // Don't allow a store OF the AI, only INTO the AI.
    } else {
      return false;   // Not a load or store.
    }

  return true;
}

namespace {

  // Data package used by RenamePass()
  class VISIBILITY_HIDDEN RenamePassData {
  public:
    RenamePassData(BasicBlock *B, BasicBlock *P,
                   const std::vector<Value *> &V) : BB(B), Pred(P), Values(V) {}
    BasicBlock *BB;
    BasicBlock *Pred;
    std::vector<Value *> Values;
  };

  struct VISIBILITY_HIDDEN PromoteMem2Reg {
    /// Allocas - The alloca instructions being promoted.
    ///
    std::vector<AllocaInst*> Allocas;
    SmallVector<AllocaInst*, 16> &RetryList;
    ETForest &ET;
    DominanceFrontier &DF;

    /// AST - An AliasSetTracker object to update.  If null, don't update it.
    ///
    AliasSetTracker *AST;

    /// AllocaLookup - Reverse mapping of Allocas.
    ///
    std::map<AllocaInst*, unsigned>  AllocaLookup;

    /// NewPhiNodes - The PhiNodes we're adding.
    ///
    DenseMap<std::pair<BasicBlock*, unsigned>, PHINode*> NewPhiNodes;
    
    /// PhiToAllocaMap - For each PHI node, keep track of which entry in Allocas
    /// it corresponds to.
    DenseMap<PHINode*, unsigned> PhiToAllocaMap;
    
    /// PointerAllocaValues - If we are updating an AliasSetTracker, then for
    /// each alloca that is of pointer type, we keep track of what to copyValue
    /// to the inserted PHI nodes here.
    ///
    std::vector<Value*> PointerAllocaValues;

    /// Visited - The set of basic blocks the renamer has already visited.
    ///
    SmallPtrSet<BasicBlock*, 16> Visited;

    /// BBNumbers - Contains a stable numbering of basic blocks to avoid
    /// non-determinstic behavior.
    DenseMap<BasicBlock*, unsigned> BBNumbers;

    /// RenamePassWorkList - Worklist used by RenamePass()
    std::vector<RenamePassData> RenamePassWorkList;

  public:
    PromoteMem2Reg(const std::vector<AllocaInst*> &A,
                   SmallVector<AllocaInst*, 16> &Retry, ETForest &et,
                   DominanceFrontier &df, AliasSetTracker *ast)
      : Allocas(A), RetryList(Retry), ET(et), DF(df), AST(ast) {}

    void run();

    /// properlyDominates - Return true if I1 properly dominates I2.
    ///
    bool properlyDominates(Instruction *I1, Instruction *I2) const {
      if (InvokeInst *II = dyn_cast<InvokeInst>(I1))
        I1 = II->getNormalDest()->begin();
      return ET.properlyDominates(I1->getParent(), I2->getParent());
    }
    
    /// dominates - Return true if BB1 dominates BB2 using the ETForest.
    ///
    bool dominates(BasicBlock *BB1, BasicBlock *BB2) const {
      return ET.dominates(BB1, BB2);
    }

  private:
    void MarkDominatingPHILive(BasicBlock *BB, unsigned AllocaNum,
                               SmallPtrSet<PHINode*, 16> &DeadPHINodes);
    bool PromoteLocallyUsedAlloca(BasicBlock *BB, AllocaInst *AI);
    void PromoteLocallyUsedAllocas(BasicBlock *BB,
                                   const std::vector<AllocaInst*> &AIs);

    void RenamePass(BasicBlock *BB, BasicBlock *Pred,
                    std::vector<Value*> &IncVals);
    bool QueuePhiNode(BasicBlock *BB, unsigned AllocaIdx, unsigned &Version,
                      SmallPtrSet<PHINode*, 16> &InsertedPHINodes);
  };

}  // end of anonymous namespace

void PromoteMem2Reg::run() {
  Function &F = *DF.getRoot()->getParent();

  // LocallyUsedAllocas - Keep track of all of the alloca instructions which are
  // only used in a single basic block.  These instructions can be efficiently
  // promoted by performing a single linear scan over that one block.  Since
  // individual basic blocks are sometimes large, we group together all allocas
  // that are live in a single basic block by the basic block they are live in.
  std::map<BasicBlock*, std::vector<AllocaInst*> > LocallyUsedAllocas;

  if (AST) PointerAllocaValues.resize(Allocas.size());

  for (unsigned AllocaNum = 0; AllocaNum != Allocas.size(); ++AllocaNum) {
    AllocaInst *AI = Allocas[AllocaNum];

    assert(isAllocaPromotable(AI) &&
           "Cannot promote non-promotable alloca!");
    assert(AI->getParent()->getParent() == &F &&
           "All allocas should be in the same function, which is same as DF!");

    if (AI->use_empty()) {
      // If there are no uses of the alloca, just delete it now.
      if (AST) AST->deleteValue(AI);
      AI->eraseFromParent();

      // Remove the alloca from the Allocas list, since it has been processed
      Allocas[AllocaNum] = Allocas.back();
      Allocas.pop_back();
      --AllocaNum;
      continue;
    }

    // Calculate the set of read and write-locations for each alloca.  This is
    // analogous to finding the 'uses' and 'definitions' of each variable.
    std::vector<BasicBlock*> DefiningBlocks;
    std::vector<BasicBlock*> UsingBlocks;

    StoreInst  *OnlyStore = 0;
    BasicBlock *OnlyBlock = 0;
    bool OnlyUsedInOneBlock = true;

    // As we scan the uses of the alloca instruction, keep track of stores, and
    // decide whether all of the loads and stores to the alloca are within the
    // same basic block.
    Value *AllocaPointerVal = 0;
    for (Value::use_iterator U =AI->use_begin(), E = AI->use_end(); U != E;++U){
      Instruction *User = cast<Instruction>(*U);
      if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
        // Remember the basic blocks which define new values for the alloca
        DefiningBlocks.push_back(SI->getParent());
        AllocaPointerVal = SI->getOperand(0);
        OnlyStore = SI;
      } else {
        LoadInst *LI = cast<LoadInst>(User);
        // Otherwise it must be a load instruction, keep track of variable reads
        UsingBlocks.push_back(LI->getParent());
        AllocaPointerVal = LI;
      }

      if (OnlyUsedInOneBlock) {
        if (OnlyBlock == 0)
          OnlyBlock = User->getParent();
        else if (OnlyBlock != User->getParent())
          OnlyUsedInOneBlock = false;
      }
    }

    // If the alloca is only read and written in one basic block, just perform a
    // linear sweep over the block to eliminate it.
    if (OnlyUsedInOneBlock) {
      LocallyUsedAllocas[OnlyBlock].push_back(AI);

      // Remove the alloca from the Allocas list, since it will be processed.
      Allocas[AllocaNum] = Allocas.back();
      Allocas.pop_back();
      --AllocaNum;
      continue;
    }

    // If there is only a single store to this value, replace any loads of
    // it that are directly dominated by the definition with the value stored.
    if (DefiningBlocks.size() == 1) {
      // Be aware of loads before the store.
      std::set<BasicBlock*> ProcessedBlocks;
      for (unsigned i = 0, e = UsingBlocks.size(); i != e; ++i)
        // If the store dominates the block and if we haven't processed it yet,
        // do so now.
        if (dominates(OnlyStore->getParent(), UsingBlocks[i]))
          if (ProcessedBlocks.insert(UsingBlocks[i]).second) {
            BasicBlock *UseBlock = UsingBlocks[i];
            
            // If the use and store are in the same block, do a quick scan to
            // verify that there are no uses before the store.
            if (UseBlock == OnlyStore->getParent()) {
              BasicBlock::iterator I = UseBlock->begin();
              for (; &*I != OnlyStore; ++I) { // scan block for store.
                if (isa<LoadInst>(I) && I->getOperand(0) == AI)
                  break;
              }
              if (&*I != OnlyStore) break;  // Do not handle this case.
            }
        
            // Otherwise, if this is a different block or if all uses happen
            // after the store, do a simple linear scan to replace loads with
            // the stored value.
            for (BasicBlock::iterator I = UseBlock->begin(),E = UseBlock->end();
                 I != E; ) {
              if (LoadInst *LI = dyn_cast<LoadInst>(I++)) {
                if (LI->getOperand(0) == AI) {
                  LI->replaceAllUsesWith(OnlyStore->getOperand(0));
                  if (AST && isa<PointerType>(LI->getType()))
                    AST->deleteValue(LI);
                  LI->eraseFromParent();
                }
              }
            }
            
            // Finally, remove this block from the UsingBlock set.
            UsingBlocks[i] = UsingBlocks.back();
            --i; --e;
          }

      // Finally, after the scan, check to see if the store is all that is left.
      if (UsingBlocks.empty()) {
        // The alloca has been processed, move on.
        Allocas[AllocaNum] = Allocas.back();
        Allocas.pop_back();
        --AllocaNum;
        continue;
      }
    }
    
    
    if (AST)
      PointerAllocaValues[AllocaNum] = AllocaPointerVal;

    // If we haven't computed a numbering for the BB's in the function, do so
    // now.
    if (BBNumbers.empty()) {
      unsigned ID = 0;
      for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
        BBNumbers[I] = ID++;
    }

    // Compute the locations where PhiNodes need to be inserted.  Look at the
    // dominance frontier of EACH basic-block we have a write in.
    //
    unsigned CurrentVersion = 0;
    SmallPtrSet<PHINode*, 16> InsertedPHINodes;
    std::vector<std::pair<unsigned, BasicBlock*> > DFBlocks;
    while (!DefiningBlocks.empty()) {
      BasicBlock *BB = DefiningBlocks.back();
      DefiningBlocks.pop_back();

      // Look up the DF for this write, add it to PhiNodes
      DominanceFrontier::const_iterator it = DF.find(BB);
      if (it != DF.end()) {
        const DominanceFrontier::DomSetType &S = it->second;

        // In theory we don't need the indirection through the DFBlocks vector.
        // In practice, the order of calling QueuePhiNode would depend on the
        // (unspecified) ordering of basic blocks in the dominance frontier,
        // which would give PHI nodes non-determinstic subscripts.  Fix this by
        // processing blocks in order of the occurance in the function.
        for (DominanceFrontier::DomSetType::const_iterator P = S.begin(),
             PE = S.end(); P != PE; ++P)
          DFBlocks.push_back(std::make_pair(BBNumbers[*P], *P));

        // Sort by which the block ordering in the function.
        std::sort(DFBlocks.begin(), DFBlocks.end());

        for (unsigned i = 0, e = DFBlocks.size(); i != e; ++i) {
          BasicBlock *BB = DFBlocks[i].second;
          if (QueuePhiNode(BB, AllocaNum, CurrentVersion, InsertedPHINodes))
            DefiningBlocks.push_back(BB);
        }
        DFBlocks.clear();
      }
    }

    // Now that we have inserted PHI nodes along the Iterated Dominance Frontier
    // of the writes to the variable, scan through the reads of the variable,
    // marking PHI nodes which are actually necessary as alive (by removing them
    // from the InsertedPHINodes set).  This is not perfect: there may PHI
    // marked alive because of loads which are dominated by stores, but there
    // will be no unmarked PHI nodes which are actually used.
    //
    for (unsigned i = 0, e = UsingBlocks.size(); i != e; ++i)
      MarkDominatingPHILive(UsingBlocks[i], AllocaNum, InsertedPHINodes);
    UsingBlocks.clear();

    // If there are any PHI nodes which are now known to be dead, remove them!
    for (SmallPtrSet<PHINode*, 16>::iterator I = InsertedPHINodes.begin(),
           E = InsertedPHINodes.end(); I != E; ++I) {
      PHINode *PN = *I;
      bool Erased=NewPhiNodes.erase(std::make_pair(PN->getParent(), AllocaNum));
      Erased=Erased;
      assert(Erased && "PHI already removed?");
      
      if (AST && isa<PointerType>(PN->getType()))
        AST->deleteValue(PN);
      PN->eraseFromParent();
      PhiToAllocaMap.erase(PN);
    }

    // Keep the reverse mapping of the 'Allocas' array.
    AllocaLookup[Allocas[AllocaNum]] = AllocaNum;
  }

  // Process all allocas which are only used in a single basic block.
  for (std::map<BasicBlock*, std::vector<AllocaInst*> >::iterator I =
         LocallyUsedAllocas.begin(), E = LocallyUsedAllocas.end(); I != E; ++I){
    const std::vector<AllocaInst*> &LocAllocas = I->second;
    assert(!LocAllocas.empty() && "empty alloca list??");

    // It's common for there to only be one alloca in the list.  Handle it
    // efficiently.
    if (LocAllocas.size() == 1) {
      // If we can do the quick promotion pass, do so now.
      if (PromoteLocallyUsedAlloca(I->first, LocAllocas[0]))
        RetryList.push_back(LocAllocas[0]);  // Failed, retry later.
    } else {
      // Locally promote anything possible.  Note that if this is unable to
      // promote a particular alloca, it puts the alloca onto the Allocas vector
      // for global processing.
      PromoteLocallyUsedAllocas(I->first, LocAllocas);
    }
  }

  if (Allocas.empty())
    return; // All of the allocas must have been trivial!

  // Set the incoming values for the basic block to be null values for all of
  // the alloca's.  We do this in case there is a load of a value that has not
  // been stored yet.  In this case, it will get this null value.
  //
  std::vector<Value *> Values(Allocas.size());
  for (unsigned i = 0, e = Allocas.size(); i != e; ++i)
    Values[i] = UndefValue::get(Allocas[i]->getAllocatedType());

  // Walks all basic blocks in the function performing the SSA rename algorithm
  // and inserting the phi nodes we marked as necessary
  //
  RenamePassWorkList.clear();
  RenamePassWorkList.push_back(RenamePassData(F.begin(), 0, Values));
  while(!RenamePassWorkList.empty()) {
    RenamePassData RPD = RenamePassWorkList.back(); 
    RenamePassWorkList.pop_back();
    // RenamePass may add new worklist entries.
    RenamePass(RPD.BB, RPD.Pred, RPD.Values);
  }
  
  // The renamer uses the Visited set to avoid infinite loops.  Clear it now.
  Visited.clear();

  // Remove the allocas themselves from the function.
  for (unsigned i = 0, e = Allocas.size(); i != e; ++i) {
    Instruction *A = Allocas[i];

    // If there are any uses of the alloca instructions left, they must be in
    // sections of dead code that were not processed on the dominance frontier.
    // Just delete the users now.
    //
    if (!A->use_empty())
      A->replaceAllUsesWith(UndefValue::get(A->getType()));
    if (AST) AST->deleteValue(A);
    A->eraseFromParent();
  }

  
  // Loop over all of the PHI nodes and see if there are any that we can get
  // rid of because they merge all of the same incoming values.  This can
  // happen due to undef values coming into the PHI nodes.  This process is
  // iterative, because eliminating one PHI node can cause others to be removed.
  bool EliminatedAPHI = true;
  while (EliminatedAPHI) {
    EliminatedAPHI = false;
    
    for (DenseMap<std::pair<BasicBlock*, unsigned>, PHINode*>::iterator I =
           NewPhiNodes.begin(), E = NewPhiNodes.end(); I != E;) {
      PHINode *PN = I->second;
      
      // If this PHI node merges one value and/or undefs, get the value.
      if (Value *V = PN->hasConstantValue(true)) {
        if (!isa<Instruction>(V) ||
            properlyDominates(cast<Instruction>(V), PN)) {
          if (AST && isa<PointerType>(PN->getType()))
            AST->deleteValue(PN);
          PN->replaceAllUsesWith(V);
          PN->eraseFromParent();
          NewPhiNodes.erase(I++);
          EliminatedAPHI = true;
          continue;
        }
      }
      ++I;
    }
  }
  
  // At this point, the renamer has added entries to PHI nodes for all reachable
  // code.  Unfortunately, there may be unreachable blocks which the renamer
  // hasn't traversed.  If this is the case, the PHI nodes may not
  // have incoming values for all predecessors.  Loop over all PHI nodes we have
  // created, inserting undef values if they are missing any incoming values.
  //
  for (DenseMap<std::pair<BasicBlock*, unsigned>, PHINode*>::iterator I =
         NewPhiNodes.begin(), E = NewPhiNodes.end(); I != E; ++I) {
    // We want to do this once per basic block.  As such, only process a block
    // when we find the PHI that is the first entry in the block.
    PHINode *SomePHI = I->second;
    BasicBlock *BB = SomePHI->getParent();
    if (&BB->front() != SomePHI)
      continue;

    // Count the number of preds for BB.
    SmallVector<BasicBlock*, 16> Preds(pred_begin(BB), pred_end(BB));

    // Only do work here if there the PHI nodes are missing incoming values.  We
    // know that all PHI nodes that were inserted in a block will have the same
    // number of incoming values, so we can just check any of them.
    if (SomePHI->getNumIncomingValues() == Preds.size())
      continue;
    
    // Ok, now we know that all of the PHI nodes are missing entries for some
    // basic blocks.  Start by sorting the incoming predecessors for efficient
    // access.
    std::sort(Preds.begin(), Preds.end());
    
    // Now we loop through all BB's which have entries in SomePHI and remove
    // them from the Preds list.
    for (unsigned i = 0, e = SomePHI->getNumIncomingValues(); i != e; ++i) {
      // Do a log(n) search of the Preds list for the entry we want.
      SmallVector<BasicBlock*, 16>::iterator EntIt =
        std::lower_bound(Preds.begin(), Preds.end(),
                         SomePHI->getIncomingBlock(i));
      assert(EntIt != Preds.end() && *EntIt == SomePHI->getIncomingBlock(i)&&
             "PHI node has entry for a block which is not a predecessor!");

      // Remove the entry
      Preds.erase(EntIt);
    }

    // At this point, the blocks left in the preds list must have dummy
    // entries inserted into every PHI nodes for the block.  Update all the phi
    // nodes in this block that we are inserting (there could be phis before
    // mem2reg runs).
    unsigned NumBadPreds = SomePHI->getNumIncomingValues();
    BasicBlock::iterator BBI = BB->begin();
    while ((SomePHI = dyn_cast<PHINode>(BBI++)) &&
           SomePHI->getNumIncomingValues() == NumBadPreds) {
      Value *UndefVal = UndefValue::get(SomePHI->getType());
      for (unsigned pred = 0, e = Preds.size(); pred != e; ++pred)
        SomePHI->addIncoming(UndefVal, Preds[pred]);
    }
  }
        
  NewPhiNodes.clear();
}

// MarkDominatingPHILive - Mem2Reg wants to construct "pruned" SSA form, not
// "minimal" SSA form.  To do this, it inserts all of the PHI nodes on the IDF
// as usual (inserting the PHI nodes in the DeadPHINodes set), then processes
// each read of the variable.  For each block that reads the variable, this
// function is called, which removes used PHI nodes from the DeadPHINodes set.
// After all of the reads have been processed, any PHI nodes left in the
// DeadPHINodes set are removed.
//
void PromoteMem2Reg::MarkDominatingPHILive(BasicBlock *BB, unsigned AllocaNum,
                                      SmallPtrSet<PHINode*, 16> &DeadPHINodes) {
  // Scan the immediate dominators of this block looking for a block which has a
  // PHI node for Alloca num.  If we find it, mark the PHI node as being alive!
  for (BasicBlock* DomBB = BB; DomBB; DomBB = ET.getIDom(DomBB)) {
    DenseMap<std::pair<BasicBlock*, unsigned>, PHINode*>::iterator
      I = NewPhiNodes.find(std::make_pair(DomBB, AllocaNum));
    if (I != NewPhiNodes.end()) {
      // Ok, we found an inserted PHI node which dominates this value.
      PHINode *DominatingPHI = I->second;

      // Find out if we previously thought it was dead.  If so, mark it as being
      // live by removing it from the DeadPHINodes set.
      if (DeadPHINodes.erase(DominatingPHI)) {
        // Now that we have marked the PHI node alive, also mark any PHI nodes
        // which it might use as being alive as well.
        for (pred_iterator PI = pred_begin(DomBB), PE = pred_end(DomBB);
             PI != PE; ++PI)
          MarkDominatingPHILive(*PI, AllocaNum, DeadPHINodes);
      }
    }
  }
}

/// PromoteLocallyUsedAlloca - Many allocas are only used within a single basic
/// block.  If this is the case, avoid traversing the CFG and inserting a lot of
/// potentially useless PHI nodes by just performing a single linear pass over
/// the basic block using the Alloca.
///
/// If we cannot promote this alloca (because it is read before it is written),
/// return true.  This is necessary in cases where, due to control flow, the
/// alloca is potentially undefined on some control flow paths.  e.g. code like
/// this is potentially correct:
///
///   for (...) { if (c) { A = undef; undef = B; } }
///
/// ... so long as A is not used before undef is set.
///
bool PromoteMem2Reg::PromoteLocallyUsedAlloca(BasicBlock *BB, AllocaInst *AI) {
  assert(!AI->use_empty() && "There are no uses of the alloca!");

  // Handle degenerate cases quickly.
  if (AI->hasOneUse()) {
    Instruction *U = cast<Instruction>(AI->use_back());
    if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
      // Must be a load of uninitialized value.
      LI->replaceAllUsesWith(UndefValue::get(AI->getAllocatedType()));
      if (AST && isa<PointerType>(LI->getType()))
        AST->deleteValue(LI);
    } else {
      // Otherwise it must be a store which is never read.
      assert(isa<StoreInst>(U));
    }
    BB->getInstList().erase(U);
  } else {
    // Uses of the uninitialized memory location shall get undef.
    Value *CurVal = 0;

    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
      Instruction *Inst = I++;
      if (LoadInst *LI = dyn_cast<LoadInst>(Inst)) {
        if (LI->getOperand(0) == AI) {
          if (!CurVal) return true;  // Could not locally promote!

          // Loads just returns the "current value"...
          LI->replaceAllUsesWith(CurVal);
          if (AST && isa<PointerType>(LI->getType()))
            AST->deleteValue(LI);
          BB->getInstList().erase(LI);
        }
      } else if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
        if (SI->getOperand(1) == AI) {
          // Store updates the "current value"...
          CurVal = SI->getOperand(0);
          BB->getInstList().erase(SI);
        }
      }
    }
  }

  // After traversing the basic block, there should be no more uses of the
  // alloca, remove it now.
  assert(AI->use_empty() && "Uses of alloca from more than one BB??");
  if (AST) AST->deleteValue(AI);
  AI->getParent()->getInstList().erase(AI);
  return false;
}

/// PromoteLocallyUsedAllocas - This method is just like
/// PromoteLocallyUsedAlloca, except that it processes multiple alloca
/// instructions in parallel.  This is important in cases where we have large
/// basic blocks, as we don't want to rescan the entire basic block for each
/// alloca which is locally used in it (which might be a lot).
void PromoteMem2Reg::
PromoteLocallyUsedAllocas(BasicBlock *BB, const std::vector<AllocaInst*> &AIs) {
  std::map<AllocaInst*, Value*> CurValues;
  for (unsigned i = 0, e = AIs.size(); i != e; ++i)
    CurValues[AIs[i]] = 0; // Insert with null value

  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
    Instruction *Inst = I++;
    if (LoadInst *LI = dyn_cast<LoadInst>(Inst)) {
      // Is this a load of an alloca we are tracking?
      if (AllocaInst *AI = dyn_cast<AllocaInst>(LI->getOperand(0))) {
        std::map<AllocaInst*, Value*>::iterator AIt = CurValues.find(AI);
        if (AIt != CurValues.end()) {
          // If loading an uninitialized value, allow the inter-block case to
          // handle it.  Due to control flow, this might actually be ok.
          if (AIt->second == 0) {  // Use of locally uninitialized value??
            RetryList.push_back(AI);   // Retry elsewhere.
            CurValues.erase(AIt);   // Stop tracking this here.
            if (CurValues.empty()) return;
          } else {
            // Loads just returns the "current value"...
            LI->replaceAllUsesWith(AIt->second);
            if (AST && isa<PointerType>(LI->getType()))
              AST->deleteValue(LI);
            BB->getInstList().erase(LI);
          }
        }
      }
    } else if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
      if (AllocaInst *AI = dyn_cast<AllocaInst>(SI->getOperand(1))) {
        std::map<AllocaInst*, Value*>::iterator AIt = CurValues.find(AI);
        if (AIt != CurValues.end()) {
          // Store updates the "current value"...
          AIt->second = SI->getOperand(0);
          BB->getInstList().erase(SI);
        }
      }
    }
  }
}



// QueuePhiNode - queues a phi-node to be added to a basic-block for a specific
// Alloca returns true if there wasn't already a phi-node for that variable
//
bool PromoteMem2Reg::QueuePhiNode(BasicBlock *BB, unsigned AllocaNo,
                                  unsigned &Version,
                                  SmallPtrSet<PHINode*, 16> &InsertedPHINodes) {
  // Look up the basic-block in question.
  PHINode *&PN = NewPhiNodes[std::make_pair(BB, AllocaNo)];

  // If the BB already has a phi node added for the i'th alloca then we're done!
  if (PN) return false;

  // Create a PhiNode using the dereferenced type... and add the phi-node to the
  // BasicBlock.
  PN = new PHINode(Allocas[AllocaNo]->getAllocatedType(),
                   Allocas[AllocaNo]->getName() + "." +
                   utostr(Version++), BB->begin());
  PhiToAllocaMap[PN] = AllocaNo;
  
  InsertedPHINodes.insert(PN);

  if (AST && isa<PointerType>(PN->getType()))
    AST->copyValue(PointerAllocaValues[AllocaNo], PN);

  return true;
}


// RenamePass - Recursively traverse the CFG of the function, renaming loads and
// stores to the allocas which we are promoting.  IncomingVals indicates what
// value each Alloca contains on exit from the predecessor block Pred.
//
void PromoteMem2Reg::RenamePass(BasicBlock *BB, BasicBlock *Pred,
                                std::vector<Value*> &IncomingVals) {
  // If we are inserting any phi nodes into this BB, they will already be in the
  // block.
  if (PHINode *APN = dyn_cast<PHINode>(BB->begin())) {
    // Pred may have multiple edges to BB.  If so, we want to add N incoming
    // values to each PHI we are inserting on the first time we see the edge.
    // Check to see if APN already has incoming values from Pred.  This also
    // prevents us from modifying PHI nodes that are not currently being
    // inserted.
    bool HasPredEntries = false;
    for (unsigned i = 0, e = APN->getNumIncomingValues(); i != e; ++i) {
      if (APN->getIncomingBlock(i) == Pred) {
        HasPredEntries = true;
        break;
      }
    }
    
    // If we have PHI nodes to update, compute the number of edges from Pred to
    // BB.
    if (!HasPredEntries) {
      TerminatorInst *PredTerm = Pred->getTerminator();
      unsigned NumEdges = 0;
      for (unsigned i = 0, e = PredTerm->getNumSuccessors(); i != e; ++i) {
        if (PredTerm->getSuccessor(i) == BB)
          ++NumEdges;
      }
      assert(NumEdges && "Must be at least one edge from Pred to BB!");
      
      // Add entries for all the phis.
      BasicBlock::iterator PNI = BB->begin();
      do {
        unsigned AllocaNo = PhiToAllocaMap[APN];
        
        // Add N incoming values to the PHI node.
        for (unsigned i = 0; i != NumEdges; ++i)
          APN->addIncoming(IncomingVals[AllocaNo], Pred);
        
        // The currently active variable for this block is now the PHI.
        IncomingVals[AllocaNo] = APN;
        
        // Get the next phi node.
        ++PNI;
        APN = dyn_cast<PHINode>(PNI);
        if (APN == 0) break;
        
        // Verify it doesn't already have entries for Pred.  If it does, it is
        // not being inserted by this mem2reg invocation.
        HasPredEntries = false;
        for (unsigned i = 0, e = APN->getNumIncomingValues(); i != e; ++i) {
          if (APN->getIncomingBlock(i) == Pred) {
            HasPredEntries = true;
            break;
          }
        }
      } while (!HasPredEntries);
    }
  }
  
  // Don't revisit blocks.
  if (!Visited.insert(BB)) return;

  for (BasicBlock::iterator II = BB->begin(); !isa<TerminatorInst>(II); ) {
    Instruction *I = II++; // get the instruction, increment iterator

    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      if (AllocaInst *Src = dyn_cast<AllocaInst>(LI->getPointerOperand())) {
        std::map<AllocaInst*, unsigned>::iterator AI = AllocaLookup.find(Src);
        if (AI != AllocaLookup.end()) {
          Value *V = IncomingVals[AI->second];

          // walk the use list of this load and replace all uses with r
          LI->replaceAllUsesWith(V);
          if (AST && isa<PointerType>(LI->getType()))
            AST->deleteValue(LI);
          BB->getInstList().erase(LI);
        }
      }
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      // Delete this instruction and mark the name as the current holder of the
      // value
      if (AllocaInst *Dest = dyn_cast<AllocaInst>(SI->getPointerOperand())) {
        std::map<AllocaInst *, unsigned>::iterator ai = AllocaLookup.find(Dest);
        if (ai != AllocaLookup.end()) {
          // what value were we writing?
          IncomingVals[ai->second] = SI->getOperand(0);
          BB->getInstList().erase(SI);
        }
      }
    }
  }

  // Recurse to our successors.
  TerminatorInst *TI = BB->getTerminator();
  for (unsigned i = 0; i != TI->getNumSuccessors(); i++)
    RenamePassWorkList.push_back(RenamePassData(TI->getSuccessor(i), BB, IncomingVals));
}

/// PromoteMemToReg - Promote the specified list of alloca instructions into
/// scalar registers, inserting PHI nodes as appropriate.  This function makes
/// use of DominanceFrontier information.  This function does not modify the CFG
/// of the function at all.  All allocas must be from the same function.
///
/// If AST is specified, the specified tracker is updated to reflect changes
/// made to the IR.
///
void llvm::PromoteMemToReg(const std::vector<AllocaInst*> &Allocas,
                           ETForest &ET, DominanceFrontier &DF,
                           AliasSetTracker *AST) {
  // If there is nothing to do, bail out...
  if (Allocas.empty()) return;

  SmallVector<AllocaInst*, 16> RetryList;
  PromoteMem2Reg(Allocas, RetryList, ET, DF, AST).run();

  // PromoteMem2Reg may not have been able to promote all of the allocas in one
  // pass, run it again if needed.
  std::vector<AllocaInst*> NewAllocas;
  while (!RetryList.empty()) {
    // If we need to retry some allocas, this is due to there being no store
    // before a read in a local block.  To counteract this, insert a store of
    // undef into the alloca right after the alloca itself.
    for (unsigned i = 0, e = RetryList.size(); i != e; ++i) {
      BasicBlock::iterator BBI = RetryList[i];

      new StoreInst(UndefValue::get(RetryList[i]->getAllocatedType()),
                    RetryList[i], ++BBI);
    }

    NewAllocas.assign(RetryList.begin(), RetryList.end());
    RetryList.clear();
    PromoteMem2Reg(NewAllocas, RetryList, ET, DF, AST).run();
    NewAllocas.clear();
  }
}
