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
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Target/TargetData.h"

using namespace llvm;

char MemoryDependenceAnalysis::ID = 0;
  
Instruction* MemoryDependenceAnalysis::NonLocal = (Instruction*)0;
Instruction* MemoryDependenceAnalysis::None = (Instruction*)~0;
  
// Register this pass...
RegisterPass<MemoryDependenceAnalysis> X("memdep",
                                           "Memory Dependence Analysis");

/// getAnalysisUsage - Does not modify anything.  It uses Alias Analysis.
///
void MemoryDependenceAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<AliasAnalysis>();
  AU.addRequiredTransitive<TargetData>();
}

/// getDependency - Return the instruction on which a memory operation
/// depends.  NOTE: A return value of NULL indicates that no dependency
/// was found in the parent block.
Instruction* MemoryDependenceAnalysis::getDependency(Instruction* query,
                                                     bool local) {
  if (!local)
    assert(0 && "Non-local memory dependence is not yet supported.");
  
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
  
  AliasAnalysis& AA = getAnalysis<AliasAnalysis>();
  
  BasicBlock::iterator blockBegin = query->getParent()->begin();
  
  // Get the pointer value for which dependence will be determined
  Value* dependee = 0;
  if (StoreInst* S = dyn_cast<StoreInst>(QI))
    dependee = S->getPointerOperand();
  else if (LoadInst* L = dyn_cast<LoadInst>(QI))
    dependee = L->getPointerOperand();
  else if (FreeInst* F = dyn_cast<FreeInst>(QI))
    dependee = F->getPointerOperand();
  else if (isa<AllocationInst>(query)) {
    // Allocations don't depend on anything
    depGraphLocal.insert(std::make_pair(query, std::make_pair(None,
                                                              true)));
    reverseDep.insert(std::make_pair(None, query));
    return None;
  } else {
    // Non-memory operations depend on their immediate predecessor
    --QI;
    depGraphLocal.insert(std::make_pair(query, std::make_pair(QI, true)));
    reverseDep.insert(std::make_pair(QI, query));
    return QI;
  }
  
  // Start with the predecessor of the queried inst
  --QI;
  
  TargetData& TD = getAnalysis<TargetData>();
  
  while (QI != blockBegin) {
    // If this inst is a memory op, get the pointer it accessed
    Value* pointer = 0;
    if (StoreInst* S = dyn_cast<StoreInst>(QI))
      pointer = S->getPointerOperand();
    else if (LoadInst* L = dyn_cast<LoadInst>(QI))
      pointer = L->getPointerOperand();
    else if (isa<AllocationInst>(QI))
      pointer = QI;
    else if (FreeInst* F = dyn_cast<FreeInst>(QI))
      pointer = F->getPointerOperand();
    else if (CallInst* C = dyn_cast<CallInst>(QI)) {
      // Call insts need special handling.  Check is they can modify our pointer
      if (AA.getModRefInfo(C, dependee, TD.getTypeSize(dependee->getType())) !=
          AliasAnalysis::NoModRef) {
        depGraphLocal.insert(std::make_pair(query, std::make_pair(C, true)));
        reverseDep.insert(std::make_pair(C, query));
        return C;
      } else {
        continue;
      }
    }
    
    // If we found a pointer, check if it could be the same as our pointer
    if (pointer) {
      AliasAnalysis::AliasResult R = AA.alias(
                                 pointer, TD.getTypeSize(pointer->getType()),
                                 dependee, TD.getTypeSize(dependee->getType()));
      
      if (R != AliasAnalysis::NoAlias) {
        depGraphLocal.insert(std::make_pair(query, std::make_pair(QI, true)));
        reverseDep.insert(std::make_pair(QI, query));
        return QI;
      }
    }
    
    QI--;
  }
  
  // If we found nothing, return the non-local flag
  depGraphLocal.insert(std::make_pair(query,
                                      std::make_pair(NonLocal, true)));
  reverseDep.insert(std::make_pair(NonLocal, query));
  
  return NonLocal;
}

/// removeInstruction - Remove an instruction from the dependence analysis,
/// updating the dependence of instructions that previously depended on it.
void MemoryDependenceAnalysis::removeInstruction(Instruction* rem) {
  // Figure out the new dep for things that currently depend on rem
  Instruction* newDep = NonLocal;
  if (depGraphLocal[rem].first != NonLocal) {
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
}
