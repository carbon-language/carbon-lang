//===- CallGraphSCCPass.cpp - Pass that operates BU on call graph ---------===//
//
// This file implements the CallGraphSCCPass class, which is used for passes
// which are implemented as bottom-up traversals on the call graph.  Because
// there may be cycles in the call graph, passes of this type operate on the
// call-graph in SCC order: that is, they process function bottom-up, except for
// recursive functions, which they process all at once.
//
//===----------------------------------------------------------------------===//

#include "llvm/CallGraphSCCPass.h"
#include "llvm/Analysis/CallGraph.h"
#include "Support/TarjanSCCIterator.h"

/// getAnalysisUsage - For this class, we declare that we require and preserve
/// the call graph.  If the derived class implements this method, it should
/// always explicitly call the implementation here.
void CallGraphSCCPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<CallGraph>();
  AU.addPreserved<CallGraph>();
}

bool CallGraphSCCPass::run(Module &M) {
  CallGraph &CG = getAnalysis<CallGraph>();
  bool Changed = false;
  for (TarjanSCC_iterator<CallGraph*> I = tarj_begin(&CG), E = tarj_end(&CG);
       I != E; ++I)
    Changed = runOnSCC(**I);
  return Changed;
}
