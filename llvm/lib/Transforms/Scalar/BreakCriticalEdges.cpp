//===- BreakCriticalEdges.cpp - Critical Edge Elimination Pass ------------===//
//
// BreakCriticalEdges pass - Break all of the critical edges in the CFG by
// inserting a dummy basic block.  This pass may be "required" by passes that
// cannot deal with critical edges.  For this usage, the structure type is
// forward declared.  This pass obviously invalidates the CFG, but can update
// forward dominator (set, immediate dominators, and tree) information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Function.h"
#include "llvm/InstrTypes.h"
#include "Support/StatisticReporter.h"

static Statistic<> NumBroken("break-crit-edges\t- Number of blocks inserted");

class BreakCriticalEdges : public FunctionPass {
public:
  virtual bool runOnFunction(Function &F);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addPreserved<DominatorSet>();
    AU.addPreserved<ImmediateDominators>();
    AU.addPreserved<DominatorTree>();
  }
};

static RegisterOpt<BreakCriticalEdges> X("break-crit-edges",
                                         "Break critical edges in CFG");

Pass *createBreakCriticalEdgesPass() { return new BreakCriticalEdges(); }

// runOnFunction - Loop over all of the edges in the CFG, breaking critical
// edges as they are found.
//
bool BreakCriticalEdges::runOnFunction(Function &F) {
  bool Changed = false;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    TerminatorInst *TI = I->getTerminator();
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
      if (isCriticalEdge(TI, i)) {
        SplitCriticalEdge(TI, i, this);
        ++NumBroken;
        Changed = true;
      }
  }

  return Changed;
}
