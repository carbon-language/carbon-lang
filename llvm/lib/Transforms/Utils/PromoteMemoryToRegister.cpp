//===- PromoteMemoryToRegister.cpp - Convert memory refs to regs ----------===//
//
// This pass is used to promote memory references to be register references.  A
// simple example of the transformation performed by this pass is:
//
//        FROM CODE                           TO CODE
//   %X = alloca int, uint 1                 ret int 42
//   store int 42, int *%X
//   %Y = load int* %X
//   ret int %Y
//
// To do this transformation, a simple analysis is done to ensure it is safe.
// Currently this just loops over all alloca instructions, looking for
// instructions that are only used in simple load and stores.
//
// After this, the code is transformed by...
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/PromoteMemoryToRegister.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/iMemory.h"
#include "llvm/Pass.h"
#include "llvm/Method.h"
#include "llvm/Assembly/Writer.h"  // For debugging
using cfg::DominanceFrontier;

// PromotePass - This class is implements the PromoteMemoryToRegister pass
//
class PromotePass : public MethodPass {
public:
  // runOnMethod - To run this pass, first we calculate the alloca instructions
  // that are safe for promotion, then we promote each one.
  //
  virtual bool runOnMethod(Method *M) {
    std::vector<AllocaInst*> Allocas;
    findSafeAllocas(M, Allocas);      // Calculate safe allocas

    // Get dominance frontier information...
    DominanceFrontier &DF = getAnalysis<DominanceFrontier>();

    // Transform each alloca in turn...
    for (std::vector<AllocaInst*>::iterator I = Allocas.begin(),
           E = Allocas.end(); I != E; ++I)
      promoteAlloca(*I, DF);

    return !Allocas.empty();
  }


  // getAnalysisUsageInfo - We need dominance frontiers
  //
  virtual void getAnalysisUsageInfo(Pass::AnalysisSet &Requires,
                                    Pass::AnalysisSet &Destroyed,
                                    Pass::AnalysisSet &Provided) {
    Requires.push_back(DominanceFrontier::ID);
  }

private:
  // findSafeAllocas - Find allocas that are safe to promote
  //
  void findSafeAllocas(Method *M, std::vector<AllocaInst*> &Allocas) const;

  // promoteAlloca - Convert the use chain of an alloca instruction into
  // register references.
  //
  void promoteAlloca(AllocaInst *AI, DominanceFrontier &DF);
};


// findSafeAllocas - Find allocas that are safe to promote
//
void PromotePass::findSafeAllocas(Method *M,
                                  std::vector<AllocaInst*> &Allocas) const {
  BasicBlock *BB = M->front();  // Get the entry node for the method

  // Look at all instructions in the entry node
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    if (AllocaInst *AI = dyn_cast<AllocaInst>(*I))       // Is it an alloca?
      if (!AI->isArrayAllocation()) {
        bool isSafe = true;
        for (Value::use_iterator UI = AI->use_begin(), UE = AI->use_end();
             UI != UE; ++UI) {   // Loop over all of the uses of the alloca
          // Only allow nonindexed memory access instructions...
          if (MemAccessInst *MAI = dyn_cast<MemAccessInst>(*UI)) {
            if (MAI->hasIndices()) { isSafe = false; break; } // indexed?
          } else {
            isSafe = false; break;   // Not a load or store?
          }
        }

        if (isSafe)              // If all checks pass, add alloca to safe list
          Allocas.push_back(AI);
      }

}



// promoteAlloca - Convert the use chain of an alloca instruction into
// register references.
//
void PromotePass::promoteAlloca(AllocaInst *AI, DominanceFrontier &DFInfo) {
  cerr << "TODO: Should process: " << AI;
}


// newPromoteMemoryToRegister - Provide an entry point to create this pass.
//
Pass *newPromoteMemoryToRegister() {
  return new PromotePass();
}
