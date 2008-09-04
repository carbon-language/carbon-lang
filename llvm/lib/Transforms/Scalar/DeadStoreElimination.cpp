//===- DeadStoreElimination.cpp - Fast Dead Store Elimination -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a trivial dead store elimination that only considers
// basic-block local redundant stores.
//
// FIXME: This should eventually be extended to be a post-dominator tree
// traversal.  Doing so would be pretty trivial.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dse"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

STATISTIC(NumFastStores, "Number of stores deleted");
STATISTIC(NumFastOther , "Number of other instrs removed");

namespace {
  struct VISIBILITY_HIDDEN DSE : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    DSE() : FunctionPass(&ID) {}

    virtual bool runOnFunction(Function &F) {
      bool Changed = false;
      for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
        Changed |= runOnBasicBlock(*I);
      return Changed;
    }

    bool runOnBasicBlock(BasicBlock &BB);
    bool handleFreeWithNonTrivialDependency(FreeInst* F,
                                            Instruction* dependency,
                                        SetVector<Instruction*>& possiblyDead);
    bool handleEndBlock(BasicBlock& BB, SetVector<Instruction*>& possiblyDead);
    bool RemoveUndeadPointers(Value* pointer, uint64_t killPointerSize,
                              BasicBlock::iterator& BBI,
                              SmallPtrSet<Value*, 64>& deadPointers, 
                              SetVector<Instruction*>& possiblyDead);
    void DeleteDeadInstructionChains(Instruction *I,
                                     SetVector<Instruction*> &DeadInsts);
    
    /// Find the base pointer that a pointer came from
    /// Because this is used to find pointers that originate
    /// from allocas, it is safe to ignore GEP indices, since
    /// either the store will be in the alloca, and thus dead,
    /// or beyond the end of the alloca, and thus undefined.
    void TranslatePointerBitCasts(Value*& v, bool zeroGepsOnly = false) {
      assert(isa<PointerType>(v->getType()) &&
             "Translating a non-pointer type?");
      while (true) {
        if (BitCastInst* C = dyn_cast<BitCastInst>(v))
          v = C->getOperand(0);
        else if (GetElementPtrInst* G = dyn_cast<GetElementPtrInst>(v))
          if (!zeroGepsOnly || G->hasAllZeroIndices()) {
            v = G->getOperand(0);
          } else {
            break;
          }
        else
          break;
      }
    }

    // getAnalysisUsage - We require post dominance frontiers (aka Control
    // Dependence Graph)
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTree>();
      AU.addRequired<TargetData>();
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<MemoryDependenceAnalysis>();
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<AliasAnalysis>();
      AU.addPreserved<MemoryDependenceAnalysis>();
    }
  };
}

char DSE::ID = 0;
static RegisterPass<DSE> X("dse", "Dead Store Elimination");

FunctionPass *llvm::createDeadStoreEliminationPass() { return new DSE(); }

bool DSE::runOnBasicBlock(BasicBlock &BB) {
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
  TargetData &TD = getAnalysis<TargetData>();  

  // Record the last-seen store to this pointer
  DenseMap<Value*, StoreInst*> lastStore;
  // Record instructions possibly made dead by deleting a store
  SetVector<Instruction*> possiblyDead;
  
  bool MadeChange = false;
  
  // Do a top-down walk on the BB
  for (BasicBlock::iterator BBI = BB.begin(), BBE = BB.end();
       BBI != BBE; ++BBI) {
    // If we find a store or a free...
    if (!isa<StoreInst>(BBI) && !isa<FreeInst>(BBI))
      continue;
      
    Value* pointer = 0;
    if (StoreInst* S = dyn_cast<StoreInst>(BBI)) {
      if (!S->isVolatile())
        pointer = S->getPointerOperand();
      else
        continue;
    } else
      pointer = cast<FreeInst>(BBI)->getPointerOperand();
      
    TranslatePointerBitCasts(pointer, true);
    StoreInst*& last = lastStore[pointer];
    bool deletedStore = false;
      
    // ... to a pointer that has been stored to before...
    if (last) {
      Instruction* dep = MD.getDependency(BBI);
        
      // ... and no other memory dependencies are between them....
      while (dep != MemoryDependenceAnalysis::None &&
             dep != MemoryDependenceAnalysis::NonLocal &&
             isa<StoreInst>(dep)) {
        if (dep != last ||
             TD.getTypeStoreSize(last->getOperand(0)->getType()) >
             TD.getTypeStoreSize(BBI->getOperand(0)->getType())) {
          dep = MD.getDependency(BBI, dep);
          continue;
        }
        
        // Remove it!
        MD.removeInstruction(last);
          
        // DCE instructions only used to calculate that store
        if (Instruction* D = dyn_cast<Instruction>(last->getOperand(0)))
          possiblyDead.insert(D);
        if (Instruction* D = dyn_cast<Instruction>(last->getOperand(1)))
          possiblyDead.insert(D);
        
        last->eraseFromParent();
        NumFastStores++;
        deletedStore = true;
        MadeChange = true;
          
        break;
      }
    }
    
    // Handle frees whose dependencies are non-trivial.
    if (FreeInst* F = dyn_cast<FreeInst>(BBI)) {
      if (!deletedStore)
        MadeChange |= handleFreeWithNonTrivialDependency(F,
                                                         MD.getDependency(F),
                                                         possiblyDead);
      // No known stores after the free
      last = 0;
    } else {
      StoreInst* S = cast<StoreInst>(BBI);
      
      // If we're storing the same value back to a pointer that we just
      // loaded from, then the store can be removed;
      if (LoadInst* L = dyn_cast<LoadInst>(S->getOperand(0))) {
        Instruction* dep = MD.getDependency(S);
        DominatorTree& DT = getAnalysis<DominatorTree>();
        
        if (!S->isVolatile() && S->getParent() == L->getParent() &&
            S->getPointerOperand() == L->getPointerOperand() &&
            ( dep == MemoryDependenceAnalysis::None ||
              dep == MemoryDependenceAnalysis::NonLocal ||
              DT.dominates(dep, L))) {
          if (Instruction* D = dyn_cast<Instruction>(S->getOperand(0)))
            possiblyDead.insert(D);
          if (Instruction* D = dyn_cast<Instruction>(S->getOperand(1)))
            possiblyDead.insert(D);
          
          // Avoid iterator invalidation.
          BBI--;
          
          MD.removeInstruction(S);
          S->eraseFromParent();
          NumFastStores++;
          MadeChange = true;
        } else
          // Update our most-recent-store map.
          last = S;
      } else
        // Update our most-recent-store map.
        last = S;
    }
  }
  
  // If this block ends in a return, unwind, unreachable, and eventually
  // tailcall, then all allocas are dead at its end.
  if (BB.getTerminator()->getNumSuccessors() == 0)
    MadeChange |= handleEndBlock(BB, possiblyDead);
  
  // Do a trivial DCE
  while (!possiblyDead.empty()) {
    Instruction *I = possiblyDead.back();
    possiblyDead.pop_back();
    DeleteDeadInstructionChains(I, possiblyDead);
  }
  
  return MadeChange;
}

/// handleFreeWithNonTrivialDependency - Handle frees of entire structures whose
/// dependency is a store to a field of that structure
bool DSE::handleFreeWithNonTrivialDependency(FreeInst* F, Instruction* dep,
                                       SetVector<Instruction*>& possiblyDead) {
  TargetData &TD = getAnalysis<TargetData>();
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
  
  if (dep == MemoryDependenceAnalysis::None ||
      dep == MemoryDependenceAnalysis::NonLocal)
    return false;
  
  StoreInst* dependency = dyn_cast<StoreInst>(dep);
  if (!dependency)
    return false;
  else if (dependency->isVolatile())
    return false;
  
  Value* depPointer = dependency->getPointerOperand();
  const Type* depType = dependency->getOperand(0)->getType();
  unsigned depPointerSize = TD.getTypeStoreSize(depType);

  // Check for aliasing
  AliasAnalysis::AliasResult A = AA.alias(F->getPointerOperand(), ~0U,
                                          depPointer, depPointerSize);

  if (A == AliasAnalysis::MustAlias) {
    // Remove it!
    MD.removeInstruction(dependency);

    // DCE instructions only used to calculate that store
    if (Instruction* D = dyn_cast<Instruction>(dependency->getOperand(0)))
      possiblyDead.insert(D);
    if (Instruction* D = dyn_cast<Instruction>(dependency->getOperand(1)))
      possiblyDead.insert(D);

    dependency->eraseFromParent();
    NumFastStores++;
    return true;
  }
  
  return false;
}

/// handleEndBlock - Remove dead stores to stack-allocated locations in the
/// function end block.  Ex:
/// %A = alloca i32
/// ...
/// store i32 1, i32* %A
/// ret void
bool DSE::handleEndBlock(BasicBlock& BB,
                         SetVector<Instruction*>& possiblyDead) {
  TargetData &TD = getAnalysis<TargetData>();
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
  
  bool MadeChange = false;
  
  // Pointers alloca'd in this function are dead in the end block
  SmallPtrSet<Value*, 64> deadPointers;
  
  // Find all of the alloca'd pointers in the entry block
  BasicBlock *Entry = BB.getParent()->begin();
  for (BasicBlock::iterator I = Entry->begin(), E = Entry->end(); I != E; ++I)
    if (AllocaInst *AI = dyn_cast<AllocaInst>(I))
      deadPointers.insert(AI);
  for (Function::arg_iterator AI = BB.getParent()->arg_begin(),
       AE = BB.getParent()->arg_end(); AI != AE; ++AI)
    if (AI->hasByValAttr())
      deadPointers.insert(AI);
  
  // Scan the basic block backwards
  for (BasicBlock::iterator BBI = BB.end(); BBI != BB.begin(); ){
    --BBI;
    
    // If we find a store whose pointer is dead...
    if (StoreInst* S = dyn_cast<StoreInst>(BBI)) {
      if (!S->isVolatile()) {
        Value* pointerOperand = S->getPointerOperand();
        // See through pointer-to-pointer bitcasts
        TranslatePointerBitCasts(pointerOperand);
      
        // Alloca'd pointers or byval arguments (which are functionally like
        // alloca's) are valid candidates for removal.
        if (deadPointers.count(pointerOperand)) {
          // Remove it!
          MD.removeInstruction(S);
        
          // DCE instructions only used to calculate that store
          if (Instruction* D = dyn_cast<Instruction>(S->getOperand(0)))
            possiblyDead.insert(D);
          if (Instruction* D = dyn_cast<Instruction>(S->getOperand(1)))
            possiblyDead.insert(D);
        
          BBI++;
          MD.removeInstruction(S);
          S->eraseFromParent();
          NumFastStores++;
          MadeChange = true;
        }
      }
      
      continue;
    
    // We can also remove memcpy's to local variables at the end of a function
    } else if (MemCpyInst* M = dyn_cast<MemCpyInst>(BBI)) {
      Value* dest = M->getDest();
      TranslatePointerBitCasts(dest);
      
      if (deadPointers.count(dest)) {
        MD.removeInstruction(M);
        
        // DCE instructions only used to calculate that memcpy
        if (Instruction* D = dyn_cast<Instruction>(M->getRawSource()))
          possiblyDead.insert(D);
        if (Instruction* D = dyn_cast<Instruction>(M->getLength()))
          possiblyDead.insert(D);
        if (Instruction* D = dyn_cast<Instruction>(M->getRawDest()))
          possiblyDead.insert(D);
        
        BBI++;
        M->eraseFromParent();
        NumFastOther++;
        MadeChange = true;
        
        continue;
      }
      
      // Because a memcpy is also a load, we can't skip it if we didn't remove it
    }
    
    Value* killPointer = 0;
    uint64_t killPointerSize = ~0UL;
    
    // If we encounter a use of the pointer, it is no longer considered dead
    if (LoadInst* L = dyn_cast<LoadInst>(BBI)) {
      // However, if this load is unused and not volatile, we can go ahead and
      // remove it, and not have to worry about it making our pointer undead!
      if (L->use_empty() && !L->isVolatile()) {
        MD.removeInstruction(L);
        
        // DCE instructions only used to calculate that load
        if (Instruction* D = dyn_cast<Instruction>(L->getPointerOperand()))
          possiblyDead.insert(D);
        
        BBI++;
        L->eraseFromParent();
        NumFastOther++;
        MadeChange = true;
        possiblyDead.remove(L);
        
        continue;
      }
      
      killPointer = L->getPointerOperand();
    } else if (VAArgInst* V = dyn_cast<VAArgInst>(BBI)) {
      killPointer = V->getOperand(0);
    } else if (isa<MemCpyInst>(BBI) &&
               isa<ConstantInt>(cast<MemCpyInst>(BBI)->getLength())) {
      killPointer = cast<MemCpyInst>(BBI)->getSource();
      killPointerSize = cast<ConstantInt>(
                            cast<MemCpyInst>(BBI)->getLength())->getZExtValue();
    } else if (AllocaInst* A = dyn_cast<AllocaInst>(BBI)) {
      deadPointers.erase(A);
      
      // Dead alloca's can be DCE'd when we reach them
      if (A->use_empty()) {
        MD.removeInstruction(A);
        
        // DCE instructions only used to calculate that load
        if (Instruction* D = dyn_cast<Instruction>(A->getArraySize()))
          possiblyDead.insert(D);
        
        BBI++;
        A->eraseFromParent();
        NumFastOther++;
        MadeChange = true;
        possiblyDead.remove(A);
      }
      
      continue;
    } else if (CallSite::get(BBI).getInstruction() != 0) {
      // If this call does not access memory, it can't
      // be undeadifying any of our pointers.
      CallSite CS = CallSite::get(BBI);
      if (AA.doesNotAccessMemory(CS))
        continue;
      
      unsigned modRef = 0;
      unsigned other = 0;
      
      // Remove any pointers made undead by the call from the dead set
      std::vector<Value*> dead;
      for (SmallPtrSet<Value*, 64>::iterator I = deadPointers.begin(),
           E = deadPointers.end(); I != E; ++I) {
        // HACK: if we detect that our AA is imprecise, it's not
        // worth it to scan the rest of the deadPointers set.  Just
        // assume that the AA will return ModRef for everything, and
        // go ahead and bail.
        if (modRef >= 16 && other == 0) {
          deadPointers.clear();
          return MadeChange;
        }

        // Get size information for the alloca
        unsigned pointerSize = ~0U;
        if (AllocaInst* A = dyn_cast<AllocaInst>(*I)) {
          if (ConstantInt* C = dyn_cast<ConstantInt>(A->getArraySize()))
            pointerSize = C->getZExtValue() * \
                          TD.getABITypeSize(A->getAllocatedType());
        } else {
          const PointerType* PT = cast<PointerType>(
                                                 cast<Argument>(*I)->getType());
          pointerSize = TD.getABITypeSize(PT->getElementType());
        }

        // See if the call site touches it
        AliasAnalysis::ModRefResult A = AA.getModRefInfo(CS, *I, pointerSize);
        
        if (A == AliasAnalysis::ModRef)
          modRef++;
        else
          other++;
        
        if (A == AliasAnalysis::ModRef || A == AliasAnalysis::Ref)
          dead.push_back(*I);
      }

      for (std::vector<Value*>::iterator I = dead.begin(), E = dead.end();
           I != E; ++I)
        deadPointers.erase(*I);
      
      continue;
    } else {
      // For any non-memory-affecting non-terminators, DCE them as we reach them
      Instruction *CI = BBI;
      if (!CI->isTerminator() && CI->use_empty() && !isa<FreeInst>(CI)) {
        
        // DCE instructions only used to calculate that load
        for (Instruction::op_iterator OI = CI->op_begin(), OE = CI->op_end();
             OI != OE; ++OI)
          if (Instruction* D = dyn_cast<Instruction>(OI))
            possiblyDead.insert(D);
        
        BBI++;
        CI->eraseFromParent();
        NumFastOther++;
        MadeChange = true;
        possiblyDead.remove(CI);
        
        continue;
      }
    }
    
    if (!killPointer)
      continue;
    
    TranslatePointerBitCasts(killPointer);
    
    // Deal with undead pointers
    MadeChange |= RemoveUndeadPointers(killPointer, killPointerSize, BBI,
                                       deadPointers, possiblyDead);
  }
  
  return MadeChange;
}

/// RemoveUndeadPointers - check for uses of a pointer that make it
/// undead when scanning for dead stores to alloca's.
bool DSE::RemoveUndeadPointers(Value* killPointer, uint64_t killPointerSize,
                                BasicBlock::iterator& BBI,
                                SmallPtrSet<Value*, 64>& deadPointers, 
                                SetVector<Instruction*>& possiblyDead) {
  TargetData &TD = getAnalysis<TargetData>();
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
                                  
  // If the kill pointer can be easily reduced to an alloca,
  // don't bother doing extraneous AA queries
  if (deadPointers.count(killPointer)) {
    deadPointers.erase(killPointer);
    return false;
  } else if (isa<GlobalValue>(killPointer)) {
    // A global can't be in the dead pointer set
    return false;
  }
  
  bool MadeChange = false;
  
  std::vector<Value*> undead;
    
  for (SmallPtrSet<Value*, 64>::iterator I = deadPointers.begin(),
      E = deadPointers.end(); I != E; ++I) {
    // Get size information for the alloca
    unsigned pointerSize = ~0U;
    if (AllocaInst* A = dyn_cast<AllocaInst>(*I)) {
      if (ConstantInt* C = dyn_cast<ConstantInt>(A->getArraySize()))
        pointerSize = C->getZExtValue() * \
                      TD.getABITypeSize(A->getAllocatedType());
    } else {
      const PointerType* PT = cast<PointerType>(
                                                cast<Argument>(*I)->getType());
      pointerSize = TD.getABITypeSize(PT->getElementType());
    }

    // See if this pointer could alias it
    AliasAnalysis::AliasResult A = AA.alias(*I, pointerSize,
                                            killPointer, killPointerSize);

    // If it must-alias and a store, we can delete it
    if (isa<StoreInst>(BBI) && A == AliasAnalysis::MustAlias) {
      StoreInst* S = cast<StoreInst>(BBI);

      // Remove it!
      MD.removeInstruction(S);

      // DCE instructions only used to calculate that store
      if (Instruction* D = dyn_cast<Instruction>(S->getOperand(0)))
        possiblyDead.insert(D);
      if (Instruction* D = dyn_cast<Instruction>(S->getOperand(1)))
        possiblyDead.insert(D);

      BBI++;
      S->eraseFromParent();
      NumFastStores++;
      MadeChange = true;

      continue;

      // Otherwise, it is undead
      } else if (A != AliasAnalysis::NoAlias)
        undead.push_back(*I);
  }

  for (std::vector<Value*>::iterator I = undead.begin(), E = undead.end();
       I != E; ++I)
      deadPointers.erase(*I);
  
  return MadeChange;
}

/// DeleteDeadInstructionChains - takes an instruction and a setvector of
/// dead instructions.  If I is dead, it is erased, and its operands are
/// checked for deadness.  If they are dead, they are added to the dead
/// setvector.
void DSE::DeleteDeadInstructionChains(Instruction *I,
                                      SetVector<Instruction*> &DeadInsts) {
  // Instruction must be dead.
  if (!I->use_empty() || !isInstructionTriviallyDead(I)) return;

  // Let the memory dependence know
  getAnalysis<MemoryDependenceAnalysis>().removeInstruction(I);

  // See if this made any operands dead.  We do it this way in case the
  // instruction uses the same operand twice.  We don't want to delete a
  // value then reference it.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    if (I->getOperand(i)->hasOneUse())
      if (Instruction* Op = dyn_cast<Instruction>(I->getOperand(i)))
        DeadInsts.insert(Op);      // Attempt to nuke it later.
    
    I->setOperand(i, 0);         // Drop from the operand list.
  }

  I->eraseFromParent();
  ++NumFastOther;
}
