//===-- PoolAllocate.cpp - Pool Allocation Pass ---------------------------===//
//
// This transform changes programs so that disjoint data structures are
// allocated out of different pools of memory, increasing locality and shrinking
// pointer size.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/PoolAllocate.h"
#include "llvm/Analysis/DataStructure.h"
#include "llvm/Pass.h"


namespace {
  struct PoolAllocate : public Pass {
    bool run(Module *M) {
      DataStructure &DS = getAnalysis<DataStructure>();
      return false;
    }

    // getAnalysisUsageInfo - This function works on the call graph of a module.
    // It is capable of updating the call graph to reflect the new state of the
    // module.
    //
    virtual void getAnalysisUsageInfo(Pass::AnalysisSet &Required,
                                      Pass::AnalysisSet &Destroyed,
                                      Pass::AnalysisSet &Provided) {
      Required.push_back(DataStructure::ID);
    }
  };
}

Pass *createPoolAllocatePass() { return new PoolAllocate(); }
