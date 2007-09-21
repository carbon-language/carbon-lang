//===- MemoryDependenceAnalysis.cpp - Mem Deps Implementation  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an analysis that determines, for a given memory
// operation, what preceding memory operations it depends on.  It builds on 
// alias analysis information, and tries to provide a lazy, caching interface to
// a common kind of alias information query.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Support/CFG.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "memdep"

using namespace llvm;

STATISTIC(NumCacheNonlocal, "Number of cached non-local responses");
STATISTIC(NumUncacheNonlocal, "Number of uncached non-local responses");

char MemoryDependenceAnalysis::ID = 0;
  
Instruction* const MemoryDependenceAnalysis::NonLocal = (Instruction*)-3;
Instruction* const MemoryDependenceAnalysis::None = (Instruction*)-4;
Instruction* const MemoryDependenceAnalysis::Dirty = (Instruction*)-5;
  
// Register this pass...
static RegisterPass<MemoryDependenceAnalysis> X("memdep",
                                                "Memory Dependence Analysis");

/// getAnalysisUsage - Does not modify anything.  It uses Alias Analysis.
///
void MemoryDependenceAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<AliasAnalysis>();
  AU.addRequiredTransitive<TargetData>();
}

/// getCallSiteDependency - Private helper for finding the local dependencies
/// of a call site.
Instruction* MemoryDependenceAnalysis::getCallSiteDependency(CallSite C,
                                                           Instruction* start,
                                                            BasicBlock* block) {
  
  AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
  TargetData& TD = getAnalysis<TargetData>();
  BasicBlock::iterator blockBegin = C.getInstruction()->getParent()->begin();
  BasicBlock::iterator QI = C.getInstruction();
  
  // If the starting point was specifiy, use it
  if (start) {
    QI = start;
    blockBegin = start->getParent()->end();
  // If the starting point wasn't specified, but the block was, use it
  } else if (!start && block) {
    QI = block->end();
    blockBegin = block->end();
  }
  
  // Walk backwards through the block, looking for dependencies
  while (QI != blockBegin) {
    --QI;
    
    // If this inst is a memory op, get the pointer it accessed
    Value* pointer = 0;
    uint64_t pointerSize = 0;
    if (StoreInst* S = dyn_cast<StoreInst>(QI)) {
      pointer = S->getPointerOperand();
      pointerSize = TD.getTypeSize(S->getOperand(0)->getType());
    } else if (LoadInst* L = dyn_cast<LoadInst>(QI)) {
      pointer = L->getPointerOperand();
      pointerSize = TD.getTypeSize(L->getType());
    } else if (AllocationInst* AI = dyn_cast<AllocationInst>(QI)) {
      pointer = AI;
      if (ConstantInt* C = dyn_cast<ConstantInt>(AI->getArraySize()))
        pointerSize = C->getZExtValue() * \
                      TD.getTypeSize(AI->getAllocatedType());
      else
        pointerSize = ~0UL;
    } else if (VAArgInst* V = dyn_cast<VAArgInst>(QI)) {
      pointer = V->getOperand(0);
      pointerSize = TD.getTypeSize(V->getType());
    } else if (FreeInst* F = dyn_cast<FreeInst>(QI)) {
      pointer = F->getPointerOperand();
      
      // FreeInsts erase the entire structure
      pointerSize = ~0UL;
    } else if (CallSite::get(QI).getInstruction() != 0) {
      if (AA.getModRefInfo(C, CallSite::get(QI)) != AliasAnalysis::NoModRef) {
        if (!start && !block) {
          depGraphLocal.insert(std::make_pair(C.getInstruction(),
                                              std::make_pair(QI, true)));
          reverseDep[QI].insert(C.getInstruction());
        }
        return QI;
      } else {
        continue;
      }
    } else
      continue;
    
    if (AA.getModRefInfo(C, pointer, pointerSize) != AliasAnalysis::NoModRef) {
      if (!start && !block) {
        depGraphLocal.insert(std::make_pair(C.getInstruction(),
                                            std::make_pair(QI, true)));
        reverseDep[QI].insert(C.getInstruction());
      }
      return QI;
    }
  }
  
  // No dependence found
  depGraphLocal.insert(std::make_pair(C.getInstruction(),
                                      std::make_pair(NonLocal, true)));
  reverseDep[NonLocal].insert(C.getInstruction());
  return NonLocal;
}

/// nonLocalHelper - Private helper used to calculate non-local dependencies
/// by doing DFS on the predecessors of a block to find its dependencies
void MemoryDependenceAnalysis::nonLocalHelper(Instruction* query,
                                              BasicBlock* block,
                                         DenseMap<BasicBlock*, Value*>& resp) {
  // Set of blocks that we've already visited in our DFS
  SmallPtrSet<BasicBlock*, 4> visited;
  // If we're updating a dirtied cache entry, we don't need to reprocess
  // already computed entries.
  for (DenseMap<BasicBlock*, Value*>::iterator I = resp.begin(), 
       E = resp.end(); I != E; ++I)
    if (I->second != Dirty)
      visited.insert(I->first);
  
  // Current stack of the DFS
  SmallVector<BasicBlock*, 4> stack;
  stack.push_back(block);
  
  // Do a basic DFS
  while (!stack.empty()) {
    BasicBlock* BB = stack.back();
    
    // If we've already visited this block, no need to revist
    if (visited.count(BB)) {
      stack.pop_back();
      continue;
    }
    
    // If we find a new block with a local dependency for query,
    // then we insert the new dependency and backtrack.
    if (BB != block) {
      visited.insert(BB);
      
      Instruction* localDep = getDependency(query, 0, BB);
      if (localDep != NonLocal) {
        resp.insert(std::make_pair(BB, localDep));
        stack.pop_back();
        
        continue;
      }
    // If we re-encounter the starting block, we still need to search it
    // because there might be a dependency in the starting block AFTER
    // the position of the query.  This is necessary to get loops right.
    } else if (BB == block && stack.size() > 1) {
      visited.insert(BB);
      
      Instruction* localDep = getDependency(query, 0, BB);
      if (localDep != query)
        resp.insert(std::make_pair(BB, localDep));
      
      stack.pop_back();
      
      continue;
    }
    
    // If we didn't find anything, recurse on the precessors of this block
    bool predOnStack = false;
    bool inserted = false;
    for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB);
         PI != PE; ++PI)
      if (!visited.count(*PI)) {
        stack.push_back(*PI);
        inserted = true;
      } else
        predOnStack = true;
    
    // If we inserted a new predecessor, then we'll come back to this block
    if (inserted)
      continue;
    // If we didn't insert because we have no predecessors, then this
    // query has no dependency at all.
    else if (!inserted && !predOnStack) {
      resp.insert(std::make_pair(BB, None));
    // If we didn't insert because our predecessors are already on the stack,
    // then we might still have a dependency, but it will be discovered during
    // backtracking.
    } else if (!inserted && predOnStack){
      resp.insert(std::make_pair(BB, NonLocal));
    }
    
    stack.pop_back();
  }
}

/// getNonLocalDependency - Fills the passed-in map with the non-local 
/// dependencies of the queries.  The map will contain NonLocal for
/// blocks between the query and its dependencies.
void MemoryDependenceAnalysis::getNonLocalDependency(Instruction* query,
                                         DenseMap<BasicBlock*, Value*>& resp) {
  if (depGraphNonLocal.count(query)) {
    DenseMap<BasicBlock*, Value*>& cached = depGraphNonLocal[query];
    NumCacheNonlocal++;
    
    SmallVector<BasicBlock*, 4> dirtied;
    for (DenseMap<BasicBlock*, Value*>::iterator I = cached.begin(),
         E = cached.end(); I != E; ++I)
      if (I->second == Dirty)
        dirtied.push_back(I->first);
    
    for (SmallVector<BasicBlock*, 4>::iterator I = dirtied.begin(),
         E = dirtied.end(); I != E; ++I) {
      Instruction* localDep = getDependency(query, 0, *I);
      if (localDep != NonLocal)
        cached[*I] = localDep;
      else {
        cached.erase(*I);
        nonLocalHelper(query, *I, cached);
      }
    }
    
    resp = cached;
    
    return;
  } else
    NumUncacheNonlocal++;
  
  // If not, go ahead and search for non-local deps.
  nonLocalHelper(query, query->getParent(), resp);
  
  // Update the non-local dependency cache
  for (DenseMap<BasicBlock*, Value*>::iterator I = resp.begin(), E = resp.end();
       I != E; ++I) {
    depGraphNonLocal[query].insert(*I);
    reverseDepNonLocal[I->second].insert(query);
  }
}

/// getDependency - Return the instruction on which a memory operation
/// depends.  The local paramter indicates if the query should only
/// evaluate dependencies within the same basic block.
Instruction* MemoryDependenceAnalysis::getDependency(Instruction* query,
                                                     Instruction* start,
                                                     BasicBlock* block) {
  // Start looking for dependencies with the queried inst
  BasicBlock::iterator QI = query;
  
  // Check for a cached result
  std::pair<Instruction*, bool> cachedResult = depGraphLocal[query];
  // If we have a _confirmed_ cached entry, return it
  if (cachedResult.second)
    return cachedResult.first;
  else if (cachedResult.first && cachedResult.first != NonLocal)
  // If we have an unconfirmed cached entry, we can start our search from there
    QI = cachedResult.first;
  
  if (start)
    QI = start;
  else if (!start && block)
    QI = block->end();
  
  AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
  TargetData& TD = getAnalysis<TargetData>();
  
  // Get the pointer value for which dependence will be determined
  Value* dependee = 0;
  uint64_t dependeeSize = 0;
  bool queryIsVolatile = false;
  if (StoreInst* S = dyn_cast<StoreInst>(query)) {
    dependee = S->getPointerOperand();
    dependeeSize = TD.getTypeSize(S->getOperand(0)->getType());
    queryIsVolatile = S->isVolatile();
  } else if (LoadInst* L = dyn_cast<LoadInst>(query)) {
    dependee = L->getPointerOperand();
    dependeeSize = TD.getTypeSize(L->getType());
    queryIsVolatile = L->isVolatile();
  } else if (VAArgInst* V = dyn_cast<VAArgInst>(query)) {
    dependee = V->getOperand(0);
    dependeeSize = TD.getTypeSize(V->getType());
  } else if (FreeInst* F = dyn_cast<FreeInst>(query)) {
    dependee = F->getPointerOperand();
    
    // FreeInsts erase the entire structure, not just a field
    dependeeSize = ~0UL;
  } else if (CallSite::get(query).getInstruction() != 0)
    return getCallSiteDependency(CallSite::get(query), start, block);
  else if (isa<AllocationInst>(query))
    return None;
  else
    return None;
  
  BasicBlock::iterator blockBegin = block ? block->begin()
                                          : query->getParent()->begin();
  
  // Walk backwards through the basic block, looking for dependencies
  while (QI != blockBegin) {
    --QI;
    
    // If this inst is a memory op, get the pointer it accessed
    Value* pointer = 0;
    uint64_t pointerSize = 0;
    if (StoreInst* S = dyn_cast<StoreInst>(QI)) {
      // All volatile loads/stores depend on each other
      if (queryIsVolatile && S->isVolatile()) {
        if (!start && !block) {
          depGraphLocal.insert(std::make_pair(query, std::make_pair(S, true)));
          reverseDep[S].insert(query);
        }
        
        return S;
      }
      
      pointer = S->getPointerOperand();
      pointerSize = TD.getTypeSize(S->getOperand(0)->getType());
    } else if (LoadInst* L = dyn_cast<LoadInst>(QI)) {
      // All volatile loads/stores depend on each other
      if (queryIsVolatile && L->isVolatile()) {
        if (!start && !block) {
          depGraphLocal.insert(std::make_pair(query, std::make_pair(L, true)));
          reverseDep[L].insert(query);
        }
        
        return L;
      }
      
      pointer = L->getPointerOperand();
      pointerSize = TD.getTypeSize(L->getType());
    } else if (AllocationInst* AI = dyn_cast<AllocationInst>(QI)) {
      pointer = AI;
      if (ConstantInt* C = dyn_cast<ConstantInt>(AI->getArraySize()))
        pointerSize = C->getZExtValue() * \
                      TD.getTypeSize(AI->getAllocatedType());
      else
        pointerSize = ~0UL;
    } else if (VAArgInst* V = dyn_cast<VAArgInst>(QI)) {
      pointer = V->getOperand(0);
      pointerSize = TD.getTypeSize(V->getType());
    } else if (FreeInst* F = dyn_cast<FreeInst>(QI)) {
      pointer = F->getPointerOperand();
      
      // FreeInsts erase the entire structure
      pointerSize = ~0UL;
    } else if (CallSite::get(QI).getInstruction() != 0) {
      // Call insts need special handling. Check if they can modify our pointer
      AliasAnalysis::ModRefResult MR = AA.getModRefInfo(CallSite::get(QI),
                                                      dependee, dependeeSize);
      
      if (MR != AliasAnalysis::NoModRef) {
        // Loads don't depend on read-only calls
        if (isa<LoadInst>(query) && MR == AliasAnalysis::Ref)
          continue;
        
        if (!start && !block) {
          depGraphLocal.insert(std::make_pair(query,
                                              std::make_pair(QI, true)));
          reverseDep[QI].insert(query);
        }
        
        return QI;
      } else {
        continue;
      }
    }
    
    // If we found a pointer, check if it could be the same as our pointer
    if (pointer) {
      AliasAnalysis::AliasResult R = AA.alias(pointer, pointerSize,
                                              dependee, dependeeSize);
      
      if (R != AliasAnalysis::NoAlias) {
        // May-alias loads don't depend on each other
        if (isa<LoadInst>(query) && isa<LoadInst>(QI) &&
            R == AliasAnalysis::MayAlias)
          continue;
        
        if (!start && !block) {
          depGraphLocal.insert(std::make_pair(query,
                                              std::make_pair(QI, true)));
          reverseDep[QI].insert(query);
        }
        
        return QI;
      }
    }
  }
  
  // If we found nothing, return the non-local flag
  if (!start && !block) {
    depGraphLocal.insert(std::make_pair(query,
                                        std::make_pair(NonLocal, true)));
    reverseDep[NonLocal].insert(query);
  }
  
  return NonLocal;
}

/// removeInstruction - Remove an instruction from the dependence analysis,
/// updating the dependence of instructions that previously depended on it.
/// This method attempts to keep the cache coherent using the reverse map.
void MemoryDependenceAnalysis::removeInstruction(Instruction* rem) {
  // Figure out the new dep for things that currently depend on rem
  Instruction* newDep = NonLocal;

  depMapType::iterator depGraphEntry = depGraphLocal.find(rem);

  if (depGraphEntry != depGraphLocal.end()) {
    if (depGraphEntry->second.first != NonLocal &&
        depGraphEntry->second.second) {
      // If we have dep info for rem, set them to it
      BasicBlock::iterator RI = depGraphEntry->second.first;
      RI++;
      newDep = RI;
    } else if (depGraphEntry->second.first == NonLocal &&
               depGraphEntry->second.second ) {
      // If we have a confirmed non-local flag, use it
      newDep = NonLocal;
    } else {
      // Otherwise, use the immediate successor of rem
      // NOTE: This is because, when getDependence is called, it will first
      // check the immediate predecessor of what is in the cache.
      BasicBlock::iterator RI = rem;
      RI++;
      newDep = RI;
    }
    
    SmallPtrSet<Instruction*, 4>& set = reverseDep[rem];
    for (SmallPtrSet<Instruction*, 4>::iterator I = set.begin(), E = set.end();
         I != E; ++I) {
      // Insert the new dependencies
      // Mark it as unconfirmed as long as it is not the non-local flag
      depGraphLocal[*I] = std::make_pair(newDep, !newDep);
    }
    
    reverseDep.erase(rem);
  }
  
  if (reverseDepNonLocal.count(rem)) {
    SmallPtrSet<Instruction*, 4>& set = reverseDepNonLocal[rem];
    for (SmallPtrSet<Instruction*, 4>::iterator I = set.begin(), E = set.end();
         I != E; ++I)
      for (DenseMap<BasicBlock*, Value*>::iterator DI =
           depGraphNonLocal[*I].begin(), DE = depGraphNonLocal[*I].end();
           DI != DE; ++DI)
        if (DI->second == rem)
          DI->second = Dirty;
    
    reverseDepNonLocal.erase(rem);
  }

  getAnalysis<AliasAnalysis>().deleteValue(rem);
}
