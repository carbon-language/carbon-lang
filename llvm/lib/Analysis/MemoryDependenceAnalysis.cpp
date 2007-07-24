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

using namespace llvm;

char MemoryDependenceAnalysis::ID = 0;
  
Instruction* MemoryDependenceAnalysis::NonLocal = (Instruction*)0;
Instruction* MemoryDependenceAnalysis::None = (Instruction*)(~0 - 1);
  
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

// Find the dependency of a CallSite
Instruction* MemoryDependenceAnalysis::getCallSiteDependency(CallSite C, Instruction* start,
                                                             bool local) {
  assert(local && "Non-local memory dependence analysis not yet implemented");
  
  AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
  TargetData& TD = getAnalysis<TargetData>();
  BasicBlock::iterator blockBegin = C.getInstruction()->getParent()->begin();
  BasicBlock::iterator QI = C.getInstruction();
  
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
        pointerSize = C->getZExtValue() * TD.getTypeSize(AI->getAllocatedType());
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
        depGraphLocal.insert(std::make_pair(C.getInstruction(), std::make_pair(QI, true)));
        reverseDep.insert(std::make_pair(QI, C.getInstruction()));
        return QI;
      } else {
        continue;
      }
    } else
      continue;
    
    if (AA.getModRefInfo(C, pointer, pointerSize) != AliasAnalysis::NoModRef) {
      depGraphLocal.insert(std::make_pair(C.getInstruction(), std::make_pair(QI, true)));
      reverseDep.insert(std::make_pair(QI, C.getInstruction()));
      return QI;
    }
  }
  
  // No dependence found
  depGraphLocal.insert(std::make_pair(C.getInstruction(), std::make_pair(NonLocal, true)));
  reverseDep.insert(std::make_pair(NonLocal, C.getInstruction()));
  return NonLocal;
}

SmallPtrSet<Instruction*, 4> MemoryDependenceAnalysis::nonLocalHelper(Instruction* query,
                                                                      BasicBlock* block) {
  SmallPtrSet<Instruction*, 4> ret;
  
  Instruction* localDep = getDependency(query, block->end(), block);
  if (localDep != NonLocal) {
    ret.insert(localDep);
    return ret;
  }
  
  for (pred_iterator PI = pred_begin(block), PE = pred_end(block);
       PI != PE; ++PI) {
    SmallPtrSet<Instruction*, 4> pred_deps = nonLocalHelper(query, *PI);
    for (SmallPtrSet<Instruction*, 4>::iterator I = pred_deps.begin(),
         E = pred_deps.end(); I != E; ++I)
      ret.insert(*I);
  }
  
  if (ret.empty())
    ret.insert(None);
  
  return ret;
}

SmallPtrSet<Instruction*, 4> MemoryDependenceAnalysis::getNonLocalDependency(Instruction* query) {
  SmallPtrSet<Instruction*, 4> ret;
  
  Instruction* localDep = getDependency(query);
  if (localDep != NonLocal) {
    ret.insert(localDep);
    return ret;
  }
  
  BasicBlock* parent = query->getParent();
  for (pred_iterator PI = pred_begin(parent), PE = pred_end(parent);
       PI != PE; ++PI) {
    SmallPtrSet<Instruction*, 4> pred_deps = nonLocalHelper(query, *PI);
    for (SmallPtrSet<Instruction*, 4>::iterator I = pred_deps.begin(),
         E = pred_deps.end(); I != E; ++I)
      ret.insert(*I);
  }
  
  if (ret.empty())
    ret.insert(None);
  
  return ret;
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
  else if (cachedResult.first != NonLocal)
  // If we have an unconfirmed cached entry, we can start our search from there
    QI = cachedResult.first;
  
  if (start)
    QI = start;
  
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
    return getCallSiteDependency(CallSite::get(query), start);
  else if (isa<AllocationInst>(query))
    return None;
  else
    return None;
  
  BasicBlock::iterator blockBegin = block ? block->begin()
                                          : query->getParent()->begin();
  
  while (QI != blockBegin) {
    --QI;
    
    // If this inst is a memory op, get the pointer it accessed
    Value* pointer = 0;
    uint64_t pointerSize = 0;
    if (StoreInst* S = dyn_cast<StoreInst>(QI)) {
      // All volatile loads/stores depend on each other
      if (queryIsVolatile && S->isVolatile()) {
        if (!start) {
          depGraphLocal.insert(std::make_pair(query, std::make_pair(S, true)));
          reverseDep.insert(std::make_pair(S, query));
        }
        
        return S;
      }
      
      pointer = S->getPointerOperand();
      pointerSize = TD.getTypeSize(S->getOperand(0)->getType());
    } else if (LoadInst* L = dyn_cast<LoadInst>(QI)) {
      // All volatile loads/stores depend on each other
      if (queryIsVolatile && L->isVolatile()) {
        if (!start) {
          depGraphLocal.insert(std::make_pair(query, std::make_pair(L, true)));
          reverseDep.insert(std::make_pair(L, query));
        }
        
        return L;
      }
      
      pointer = L->getPointerOperand();
      pointerSize = TD.getTypeSize(L->getType());
    } else if (AllocationInst* AI = dyn_cast<AllocationInst>(QI)) {
      pointer = AI;
      if (ConstantInt* C = dyn_cast<ConstantInt>(AI->getArraySize()))
        pointerSize = C->getZExtValue() * TD.getTypeSize(AI->getAllocatedType());
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
      // Call insts need special handling.  Check is they can modify our pointer
      if (AA.getModRefInfo(CallSite::get(QI), dependee, dependeeSize) !=
          AliasAnalysis::NoModRef) {
        if (!start) {
          depGraphLocal.insert(std::make_pair(query, std::make_pair(QI, true)));
          reverseDep.insert(std::make_pair(QI, query));
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
        if (!start) {
          depGraphLocal.insert(std::make_pair(query, std::make_pair(QI, true)));
          reverseDep.insert(std::make_pair(QI, query));
        }
        
        return QI;
      }
    }
  }
  
  // If we found nothing, return the non-local flag
  if (!start) {
    depGraphLocal.insert(std::make_pair(query,
                                        std::make_pair(NonLocal, true)));
    reverseDep.insert(std::make_pair(NonLocal, query));
  }
  
  return NonLocal;
}

/// removeInstruction - Remove an instruction from the dependence analysis,
/// updating the dependence of instructions that previously depended on it.
void MemoryDependenceAnalysis::removeInstruction(Instruction* rem) {
  // Figure out the new dep for things that currently depend on rem
  Instruction* newDep = NonLocal;
  if (depGraphLocal[rem].first != NonLocal &&
      depGraphLocal[rem].second) {
    // If we have dep info for rem, set them to it
    BasicBlock::iterator RI = depGraphLocal[rem].first;
    RI++;
    newDep = RI;
  } else if (depGraphLocal[rem].first == NonLocal &&
             depGraphLocal[rem].second ) {
    // If we have a confirmed non-local flag, use it
    newDep = NonLocal;
  } else {
    // Otherwise, use the immediate successor of rem
    // NOTE: This is because, when getDependence is called, it will first check
    // the immediate predecessor of what is in the cache.
    BasicBlock::iterator RI = rem;
    RI++;
    newDep = RI;
  }

  std::multimap<Instruction*, Instruction*>::iterator I = reverseDep.find(rem);
  while (I->first == rem) {
    // Insert the new dependencies
    // Mark it as unconfirmed as long as it is not the non-local flag
    depGraphLocal[I->second] = std::make_pair(newDep, !newDep);
    reverseDep.erase(I);
    I = reverseDep.find(rem);
  }
  
  getAnalysis<AliasAnalysis>().deleteValue(rem);
}
