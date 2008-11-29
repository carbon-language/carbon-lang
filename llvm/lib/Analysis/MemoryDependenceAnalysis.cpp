//===- MemoryDependenceAnalysis.cpp - Mem Deps Implementation  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an analysis that determines, for a given memory
// operation, what preceding memory operations it depends on.  It builds on 
// alias analysis information, and tries to provide a lazy, caching interface to
// a common kind of alias information query.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "memdep"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetData.h"
using namespace llvm;

STATISTIC(NumCacheNonLocal, "Number of cached non-local responses");
STATISTIC(NumUncacheNonLocal, "Number of uncached non-local responses");

char MemoryDependenceAnalysis::ID = 0;
  
// Register this pass...
static RegisterPass<MemoryDependenceAnalysis> X("memdep",
                                     "Memory Dependence Analysis", false, true);

/// getAnalysisUsage - Does not modify anything.  It uses Alias Analysis.
///
void MemoryDependenceAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<AliasAnalysis>();
  AU.addRequiredTransitive<TargetData>();
}

/// getCallSiteDependency - Private helper for finding the local dependencies
/// of a call site.
MemDepResult MemoryDependenceAnalysis::
getCallSiteDependency(CallSite C, BasicBlock::iterator ScanIt,
                      BasicBlock *BB) {
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  TargetData &TD = getAnalysis<TargetData>();
  
  // Walk backwards through the block, looking for dependencies
  while (ScanIt != BB->begin()) {
    Instruction *Inst = --ScanIt;
    
    // If this inst is a memory op, get the pointer it accessed
    Value *Pointer = 0;
    uint64_t PointerSize = 0;
    if (StoreInst *S = dyn_cast<StoreInst>(Inst)) {
      Pointer = S->getPointerOperand();
      PointerSize = TD.getTypeStoreSize(S->getOperand(0)->getType());
    } else if (AllocationInst *AI = dyn_cast<AllocationInst>(Inst)) {
      Pointer = AI;
      if (ConstantInt *C = dyn_cast<ConstantInt>(AI->getArraySize()))
        // Use ABI size (size between elements), not store size (size of one
        // element without padding).
        PointerSize = C->getZExtValue() *
                      TD.getABITypeSize(AI->getAllocatedType());
      else
        PointerSize = ~0UL;
    } else if (VAArgInst *V = dyn_cast<VAArgInst>(Inst)) {
      Pointer = V->getOperand(0);
      PointerSize = TD.getTypeStoreSize(V->getType());
    } else if (FreeInst *F = dyn_cast<FreeInst>(Inst)) {
      Pointer = F->getPointerOperand();
      
      // FreeInsts erase the entire structure
      PointerSize = ~0UL;
    } else if (isa<CallInst>(Inst) || isa<InvokeInst>(Inst)) {
      if (AA.getModRefBehavior(CallSite::get(Inst)) ==
            AliasAnalysis::DoesNotAccessMemory)
        continue;
      return MemDepResult::get(Inst);
    } else
      continue;
    
    if (AA.getModRefInfo(C, Pointer, PointerSize) != AliasAnalysis::NoModRef)
      return MemDepResult::get(Inst);
  }
  
  // No dependence found.
  return MemDepResult::getNonLocal();
}

/// getNonLocalDependency - Perform a full dependency query for the
/// specified instruction, returning the set of blocks that the value is
/// potentially live across.  The returned set of results will include a
/// "NonLocal" result for all blocks where the value is live across.
///
/// This method assumes the instruction returns a "nonlocal" dependency
/// within its own block.
///
void MemoryDependenceAnalysis::
getNonLocalDependency(Instruction *QueryInst,
                      SmallVectorImpl<std::pair<BasicBlock*, 
                                                      MemDepResult> > &Result) {
  assert(getDependency(QueryInst).isNonLocal() &&
     "getNonLocalDependency should only be used on insts with non-local deps!");
  DenseMap<BasicBlock*, DepResultTy> &Cache = NonLocalDeps[QueryInst];

  /// DirtyBlocks - This is the set of blocks that need to be recomputed.  In
  /// the cached case, this can happen due to instructions being deleted etc. In
  /// the uncached case, this starts out as the set of predecessors we care
  /// about.
  SmallVector<BasicBlock*, 32> DirtyBlocks;
  
  if (!Cache.empty()) {
    // If we already have a partially computed set of results, scan them to
    // determine what is dirty, seeding our initial DirtyBlocks worklist.
    // FIXME: In the "don't need to be updated" case, this is expensive, why not
    // have a per-"cache" flag saying it is undirty?
    for (DenseMap<BasicBlock*, DepResultTy>::iterator I = Cache.begin(),
         E = Cache.end(); I != E; ++I)
      if (I->second.getInt() == Dirty)
        DirtyBlocks.push_back(I->first);
    
    NumCacheNonLocal++;
    
    //cerr << "CACHED CASE: " << DirtyBlocks.size() << " dirty: "
    //     << Cache.size() << " cached: " << *QueryInst;
  } else {
    // Seed DirtyBlocks with each of the preds of QueryInst's block.
    BasicBlock *QueryBB = QueryInst->getParent();
    DirtyBlocks.append(pred_begin(QueryBB), pred_end(QueryBB));
    NumUncacheNonLocal++;
  }
  
  
  // Iterate while we still have blocks to update.
  while (!DirtyBlocks.empty()) {
    BasicBlock *DirtyBB = DirtyBlocks.back();
    DirtyBlocks.pop_back();
    
    // Get the entry for this block.  Note that this relies on DepResultTy
    // default initializing to Dirty.
    DepResultTy &DirtyBBEntry = Cache[DirtyBB];
    
    // If DirtyBBEntry isn't dirty, it ended up on the worklist multiple times.
    if (DirtyBBEntry.getInt() != Dirty) continue;

    // Find out if this block has a local dependency for QueryInst.
    // FIXME: If the dirty entry has an instruction pointer, scan from it!
    // FIXME: Don't convert back and forth for MemDepResult <-> DepResultTy.
    
    // If the dirty entry has a pointer, start scanning from it so we don't have
    // to rescan the entire block.
    BasicBlock::iterator ScanPos = DirtyBB->end();
    if (Instruction *Inst = DirtyBBEntry.getPointer())
      ScanPos = Inst;
    
    DirtyBBEntry = ConvFromResult(getDependencyFrom(QueryInst, ScanPos,
                                                    DirtyBB));
           
    // If the block has a dependency (i.e. it isn't completely transparent to
    // the value), remember it!
    if (DirtyBBEntry.getInt() != NonLocal) {
      // Keep the ReverseNonLocalDeps map up to date so we can efficiently
      // update this when we remove instructions.
      if (Instruction *Inst = DirtyBBEntry.getPointer())
        ReverseNonLocalDeps[Inst].insert(QueryInst);
      continue;
    }
    
    // If the block *is* completely transparent to the load, we need to check
    // the predecessors of this block.  Add them to our worklist.
    DirtyBlocks.append(pred_begin(DirtyBB), pred_end(DirtyBB));
  }
  
  
  // Copy the result into the output set.
  for (DenseMap<BasicBlock*, DepResultTy>::iterator I = Cache.begin(),
       E = Cache.end(); I != E; ++I)
    Result.push_back(std::make_pair(I->first, ConvToResult(I->second)));
}

/// getDependency - Return the instruction on which a memory operation
/// depends.  The local parameter indicates if the query should only
/// evaluate dependencies within the same basic block.
MemDepResult MemoryDependenceAnalysis::
getDependencyFrom(Instruction *QueryInst, BasicBlock::iterator ScanIt, 
                  BasicBlock *BB) {
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  TargetData &TD = getAnalysis<TargetData>();
  
  // Get the pointer value for which dependence will be determined
  Value *MemPtr = 0;
  uint64_t MemSize = 0;
  bool MemVolatile = false;
  
  if (StoreInst* S = dyn_cast<StoreInst>(QueryInst)) {
    MemPtr = S->getPointerOperand();
    MemSize = TD.getTypeStoreSize(S->getOperand(0)->getType());
    MemVolatile = S->isVolatile();
  } else if (LoadInst* L = dyn_cast<LoadInst>(QueryInst)) {
    MemPtr = L->getPointerOperand();
    MemSize = TD.getTypeStoreSize(L->getType());
    MemVolatile = L->isVolatile();
  } else if (VAArgInst* V = dyn_cast<VAArgInst>(QueryInst)) {
    MemPtr = V->getOperand(0);
    MemSize = TD.getTypeStoreSize(V->getType());
  } else if (FreeInst* F = dyn_cast<FreeInst>(QueryInst)) {
    MemPtr = F->getPointerOperand();
    // FreeInsts erase the entire structure, not just a field.
    MemSize = ~0UL;
  } else if (isa<CallInst>(QueryInst) || isa<InvokeInst>(QueryInst))
    return getCallSiteDependency(CallSite::get(QueryInst), ScanIt, BB);
  else  // Non-memory instructions depend on nothing.
    return MemDepResult::getNone();
  
  // Walk backwards through the basic block, looking for dependencies
  while (ScanIt != BB->begin()) {
    Instruction *Inst = --ScanIt;

    // If the access is volatile and this is a volatile load/store, return a
    // dependence.
    if (MemVolatile &&
        ((isa<LoadInst>(Inst) && cast<LoadInst>(Inst)->isVolatile()) ||
         (isa<StoreInst>(Inst) && cast<StoreInst>(Inst)->isVolatile())))
      return MemDepResult::get(Inst);

    // MemDep is broken w.r.t. loads: it says that two loads of the same pointer
    // depend on each other.  :(
    // FIXME: ELIMINATE THIS!
    if (LoadInst *L = dyn_cast<LoadInst>(Inst)) {
      Value *Pointer = L->getPointerOperand();
      uint64_t PointerSize = TD.getTypeStoreSize(L->getType());
      
      // If we found a pointer, check if it could be the same as our pointer
      AliasAnalysis::AliasResult R =
        AA.alias(Pointer, PointerSize, MemPtr, MemSize);
      
      if (R == AliasAnalysis::NoAlias)
        continue;
      
      // May-alias loads don't depend on each other without a dependence.
      if (isa<LoadInst>(QueryInst) && R == AliasAnalysis::MayAlias)
        continue;
      return MemDepResult::get(Inst);
    }
    
    // FIXME: This claims that an access depends on the allocation.  This may
    // make sense, but is dubious at best.  It would be better to fix GVN to
    // handle a 'None' Query.
    if (AllocationInst *AI = dyn_cast<AllocationInst>(Inst)) {
      Value *Pointer = AI;
      uint64_t PointerSize;
      if (ConstantInt *C = dyn_cast<ConstantInt>(AI->getArraySize()))
        // Use ABI size (size between elements), not store size (size of one
        // element without padding).
        PointerSize = C->getZExtValue() * 
          TD.getABITypeSize(AI->getAllocatedType());
      else
        PointerSize = ~0UL;
      
      AliasAnalysis::AliasResult R =
        AA.alias(Pointer, PointerSize, MemPtr, MemSize);
      
      if (R == AliasAnalysis::NoAlias)
        continue;
      return MemDepResult::get(Inst);
    }
      
    
    // See if this instruction mod/ref's the pointer.
    AliasAnalysis::ModRefResult MRR = AA.getModRefInfo(Inst, MemPtr, MemSize);

    if (MRR == AliasAnalysis::NoModRef)
      continue;
    
    // Loads don't depend on read-only instructions.
    if (isa<LoadInst>(QueryInst) && MRR == AliasAnalysis::Ref)
      continue;
    
    // Otherwise, there is a dependence.
    return MemDepResult::get(Inst);
  }
  
  // If we found nothing, return the non-local flag.
  return MemDepResult::getNonLocal();
}

/// getDependency - Return the instruction on which a memory operation
/// depends.
MemDepResult MemoryDependenceAnalysis::getDependency(Instruction *QueryInst) {
  Instruction *ScanPos = QueryInst;
  
  // Check for a cached result
  DepResultTy &LocalCache = LocalDeps[QueryInst];
  
  // If the cached entry is non-dirty, just return it.  Note that this depends
  // on DepResultTy's default constructing to 'dirty'.
  if (LocalCache.getInt() != Dirty)
    return ConvToResult(LocalCache);
    
  // Otherwise, if we have a dirty entry, we know we can start the scan at that
  // instruction, which may save us some work.
  if (Instruction *Inst = LocalCache.getPointer())
    ScanPos = Inst;
  
  // Do the scan.
  MemDepResult Res = 
    getDependencyFrom(QueryInst, ScanPos, QueryInst->getParent());  
  
  // Remember the result!
  // FIXME: Don't convert back and forth!  Make a shared helper function.
  LocalCache = ConvFromResult(Res);
  if (Instruction *I = Res.getInst())
    ReverseLocalDeps[I].insert(QueryInst);
  
  return Res;
}

/// removeInstruction - Remove an instruction from the dependence analysis,
/// updating the dependence of instructions that previously depended on it.
/// This method attempts to keep the cache coherent using the reverse map.
void MemoryDependenceAnalysis::removeInstruction(Instruction *RemInst) {
  // Walk through the Non-local dependencies, removing this one as the value
  // for any cached queries.
  for (DenseMap<BasicBlock*, DepResultTy>::iterator DI =
       NonLocalDeps[RemInst].begin(), DE = NonLocalDeps[RemInst].end();
       DI != DE; ++DI)
    if (Instruction *Inst = DI->second.getPointer())
      ReverseNonLocalDeps[Inst].erase(RemInst);

  // Shortly after this, we will look for things that depend on RemInst.  In
  // order to update these, we'll need a new dependency to base them on.  We
  // could completely delete any entries that depend on this, but it is better
  // to make a more accurate approximation where possible.  Compute that better
  // approximation if we can.
  DepResultTy NewDependency;
  
  // If we have a cached local dependence query for this instruction, remove it.
  //
  LocalDepMapType::iterator LocalDepEntry = LocalDeps.find(RemInst);
  if (LocalDepEntry != LocalDeps.end()) {
    DepResultTy LocalDep = LocalDepEntry->second;
    
    // Remove this local dependency info.
    LocalDeps.erase(LocalDepEntry);
    
    // Remove us from DepInst's reverse set now that the local dep info is gone.
    if (Instruction *Inst = LocalDep.getPointer())
      ReverseLocalDeps[Inst].erase(RemInst);

    // If we have unconfirmed info, don't trust it.
    if (LocalDep.getInt() != Dirty) {
      // If we have a confirmed non-local flag, use it.
      if (LocalDep.getInt() == NonLocal || LocalDep.getInt() == None) {
        // The only time this dependency is confirmed is if it is non-local.
        NewDependency = LocalDep;
      } else {
        // If we have dep info for RemInst, set them to it.
        Instruction *NDI = next(BasicBlock::iterator(LocalDep.getPointer()));
        if (NDI != RemInst) // Don't use RemInst for the new dependency!
          NewDependency = DepResultTy(NDI, Dirty);
      }
    }
  }
  
  // If we don't already have a local dependency answer for this instruction,
  // use the immediate successor of RemInst.  We use the successor because
  // getDependence starts by checking the immediate predecessor of what is in
  // the cache.
  if (NewDependency == DepResultTy(0, Dirty))
    NewDependency = DepResultTy(next(BasicBlock::iterator(RemInst)), Dirty);
  
  // Loop over all of the things that depend on the instruction we're removing.
  // 
  SmallVector<std::pair<Instruction*, Instruction*>, 8> ReverseDepsToAdd;
  
  ReverseDepMapType::iterator ReverseDepIt = ReverseLocalDeps.find(RemInst);
  if (ReverseDepIt != ReverseLocalDeps.end()) {
    SmallPtrSet<Instruction*, 4> &ReverseDeps = ReverseDepIt->second;
    for (SmallPtrSet<Instruction*, 4>::iterator I = ReverseDeps.begin(),
         E = ReverseDeps.end(); I != E; ++I) {
      Instruction *InstDependingOnRemInst = *I;
      
      // If we thought the instruction depended on itself (possible for
      // unconfirmed dependencies) ignore the update.
      if (InstDependingOnRemInst == RemInst) continue;
      
      // Insert the new dependencies.
      // FIXME: DEPENDENCIES ARE NOT TRANSITIVE!
      //cerr << "FOO:\n";
      //RemInst->dump();
      //InstDependingOnRemInst->dump();
      LocalDeps[InstDependingOnRemInst] = NewDependency;
      
      // If our NewDependency is an instruction, make sure to remember that new
      // things depend on it.
      if (Instruction *Inst = NewDependency.getPointer()) {
        assert(Inst != RemInst);
        ReverseDepsToAdd.push_back(std::make_pair(Inst, 
                                                  InstDependingOnRemInst));
      }
    }
    
    ReverseLocalDeps.erase(ReverseDepIt);

    // Add new reverse deps after scanning the set, to avoid invalidating the
    // 'ReverseDeps' reference.
    while (!ReverseDepsToAdd.empty()) {
      ReverseLocalDeps[ReverseDepsToAdd.back().first]
        .insert(ReverseDepsToAdd.back().second);
      ReverseDepsToAdd.pop_back();
    }
  }
  
  ReverseDepIt = ReverseNonLocalDeps.find(RemInst);
  if (ReverseDepIt != ReverseNonLocalDeps.end()) {
    SmallPtrSet<Instruction*, 4>& set = ReverseDepIt->second;
    for (SmallPtrSet<Instruction*, 4>::iterator I = set.begin(), E = set.end();
         I != E; ++I)
      for (DenseMap<BasicBlock*, DepResultTy>::iterator
           DI = NonLocalDeps[*I].begin(), DE = NonLocalDeps[*I].end();
           DI != DE; ++DI)
        if (DI->second.getPointer() == RemInst) {
          // Convert to a dirty entry for the subsequent instruction.
          DI->second.setInt(Dirty);
          if (RemInst->isTerminator())
            DI->second.setPointer(0);
          else {
            Instruction *NextI = next(BasicBlock::iterator(RemInst));
            DI->second.setPointer(NextI);
            assert(NextI != RemInst);
            ReverseDepsToAdd.push_back(std::make_pair(NextI, *I));
          }
        }

    ReverseNonLocalDeps.erase(ReverseDepIt);

    // Add new reverse deps after scanning the set, to avoid invalidating 'Set'
    while (!ReverseDepsToAdd.empty()) {
      ReverseNonLocalDeps[ReverseDepsToAdd.back().first]
        .insert(ReverseDepsToAdd.back().second);
      ReverseDepsToAdd.pop_back();
    }
  }
  
  NonLocalDeps.erase(RemInst);
  getAnalysis<AliasAnalysis>().deleteValue(RemInst);
  DEBUG(verifyRemoved(RemInst));
}

/// verifyRemoved - Verify that the specified instruction does not occur
/// in our internal data structures.
void MemoryDependenceAnalysis::verifyRemoved(Instruction *D) const {
  for (LocalDepMapType::const_iterator I = LocalDeps.begin(),
       E = LocalDeps.end(); I != E; ++I) {
    assert(I->first != D && "Inst occurs in data structures");
    assert(I->second.getPointer() != D &&
           "Inst occurs in data structures");
  }
  
  for (NonLocalDepMapType::const_iterator I = NonLocalDeps.begin(),
       E = NonLocalDeps.end(); I != E; ++I) {
    assert(I->first != D && "Inst occurs in data structures");
    for (DenseMap<BasicBlock*, DepResultTy>::iterator II = I->second.begin(),
         EE = I->second.end(); II  != EE; ++II)
      assert(II->second.getPointer() != D && "Inst occurs in data structures");
  }
  
  for (ReverseDepMapType::const_iterator I = ReverseLocalDeps.begin(),
       E = ReverseLocalDeps.end(); I != E; ++I)
    for (SmallPtrSet<Instruction*, 4>::const_iterator II = I->second.begin(),
         EE = I->second.end(); II != EE; ++II)
      assert(*II != D && "Inst occurs in data structures");
  
  for (ReverseDepMapType::const_iterator I = ReverseNonLocalDeps.begin(),
       E = ReverseNonLocalDeps.end();
       I != E; ++I)
    for (SmallPtrSet<Instruction*, 4>::const_iterator II = I->second.begin(),
         EE = I->second.end(); II != EE; ++II)
      assert(*II != D && "Inst occurs in data structures");
}
