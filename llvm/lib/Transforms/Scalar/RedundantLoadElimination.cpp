//===- FastDLE.cpp - Fast Dead Load Elimination ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a trivial dead load elimination that only considers
// basic-block local redundant load.
//
// FIXME: This should eventually be extended to be a post-dominator tree
// traversal.  Doing so would be pretty trivial.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "rle"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

STATISTIC(NumFastLoads, "Number of loads deleted");

namespace {
  struct VISIBILITY_HIDDEN RLE : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    RLE() : FunctionPass((intptr_t)&ID) {}

    virtual bool runOnFunction(Function &F) {
      bool Changed = false;
      for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
        Changed |= runOnBasicBlock(*I);
      return Changed;
    }

    bool runOnBasicBlock(BasicBlock &BB);

    // getAnalysisUsage - We require post dominance frontiers (aka Control
    // Dependence Graph)
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<MemoryDependenceAnalysis>();
      AU.addPreserved<MemoryDependenceAnalysis>();
    }
  };
  char RLE::ID = 0;
  RegisterPass<RLE> X("rle", "Redundant Load Elimination");
}

FunctionPass *llvm::createRedundantLoadEliminationPass() { return new RLE(); }

bool RLE::runOnBasicBlock(BasicBlock &BB) {
  MemoryDependenceAnalysis& MD = getAnalysis<MemoryDependenceAnalysis>();
  
  // Record the last-seen load from this pointer
  DenseMap<Value*, LoadInst*> lastLoad;
  
  bool MadeChange = false;
  
  // Do a top-down walk on the BB
  for (BasicBlock::iterator BBI = BB.begin(), BBE = BB.end(); BBI != BBE; ++BBI) {
    // If we find a store or a free...
    if (LoadInst* L = dyn_cast<LoadInst>(BBI)) {
      // We can't delete volatile loads
      if (L->isVolatile()) {
        lastLoad[L->getPointerOperand()] = L;
        continue;
      }
      
      Value* pointer = L->getPointerOperand();
      LoadInst*& last = lastLoad[pointer];
      
      // ... to a pointer that has been loaded from before...
      Instruction* dep = MD.getDependency(BBI);
      bool deletedLoad = false;
      
      while (dep != MemoryDependenceAnalysis::None &&
             dep != MemoryDependenceAnalysis::NonLocal &&
             (isa<LoadInst>(dep) || isa<StoreInst>(dep))) {
        // ... that depends on a store ...
        if (StoreInst* S = dyn_cast<StoreInst>(dep)) {
          if (S->getPointerOperand() == pointer) {
            // Remove it!
            MD.removeInstruction(BBI);
            
            BBI--;
            L->replaceAllUsesWith(S->getOperand(0));
            L->eraseFromParent();
            NumFastLoads++;
            deletedLoad = true;
            MadeChange = true;
          }
          
          // Whether we removed it or not, we can't
          // go any further
          break;
        } else if (!last) {
          // If we don't depend on a store, and we haven't
          // been loaded before, bail.
          break;
        } else if (dep == last) {
          // Remove it!
          MD.removeInstruction(BBI);
          
          BBI--;
          L->replaceAllUsesWith(last);
          L->eraseFromParent();
          deletedLoad = true;
          NumFastLoads++;
          MadeChange = true;
            
          break;
        } else {
          dep = MD.getDependency(BBI, dep);
        }
      }
      
      if (!deletedLoad)
        last = L;
    }
  }
  
  return MadeChange;
}


