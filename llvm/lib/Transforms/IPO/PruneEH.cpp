//===- PruneEH.cpp - Pass which deletes unused exception handlers ---------===//
//
// This file implements a simple interprocedural pass which walks the
// call-graph, turning invoke instructions into calls, iff the callee cannot
// throw an exception.  It implements this as a bottom-up traversal of the
// call-graph.
//
//===----------------------------------------------------------------------===//

#include "llvm/CallGraphSCCPass.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/Analysis/CallGraph.h"
#include "Support/Statistic.h"
#include <set>

namespace {
  Statistic<> NumRemoved("prune-eh", "Number of invokes removed");

  struct PruneEH : public CallGraphSCCPass {
    /// DoesNotThrow - This set contains all of the functions which we have
    /// determined cannot throw exceptions.
    std::set<CallGraphNode*> DoesNotThrow;

    // runOnSCC - Analyze the SCC, performing the transformation if possible.
    bool runOnSCC(const std::vector<CallGraphNode *> &SCC);
  };
  RegisterOpt<PruneEH> X("prune-eh", "Remove unused exception handling info");
}

Pass *createPruneEHPass() { return new PruneEH(); }


bool PruneEH::runOnSCC(const std::vector<CallGraphNode *> &SCC) {
  CallGraph &CG = getAnalysis<CallGraph>();

  // First, check to see if any callees might throw or if there are any external
  // functions in this SCC: if so, we cannot prune any functions in this SCC.
  // If this SCC includes the unwind instruction, we KNOW it throws, so
  // obviously the SCC might throw.
  //
  bool SCCMightThrow = false;
  for (unsigned i = 0, e = SCC.size(); i != e; ++i)
    if (!DoesNotThrow.count(SCC[i]) &&          // Calls maybe throwing fn
        // Make sure this is not one of the fn's in the SCC.
        std::find(SCC.begin(), SCC.end(), SCC[i]) == SCC.end()) {
      SCCMightThrow = true; break;
    } else if (Function *F = SCC[i]->getFunction())
      if (F->isExternal()) {
        SCCMightThrow = true; break;
      } else {
        for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I)
          if (isa<UnwindInst>(I->getTerminator())) {  // Uses unwind!
            SCCMightThrow = true; break;
          }
      }

  bool MadeChange = false;

  for (unsigned i = 0, e = SCC.size(); i != e; ++i) {
    // If the SCC can't throw, remember this for callers...
    if (!SCCMightThrow)
      DoesNotThrow.insert(SCC[i]);

    // Convert any invoke instructions to non-throwing functions in this node
    // into call instructions with a branch.  This makes the exception blocks
    // dead.
    if (Function *F = SCC[i]->getFunction())
      for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I)
        if (InvokeInst *II = dyn_cast<InvokeInst>(I->getTerminator()))
          if (Function *F = II->getCalledFunction())
            if (DoesNotThrow.count(CG[F])) {
              // Insert a call instruction before the invoke...
              std::string Name = II->getName();  II->setName("");
              Value *Call = new CallInst(II->getCalledValue(),
                                         std::vector<Value*>(II->op_begin()+3,
                                                             II->op_end()),
                                         Name, II);
              
              // Anything that used the value produced by the invoke instruction
              // now uses the value produced by the call instruction.
              II->replaceAllUsesWith(Call);
          
              // Insert a branch to the normal destination right before the
              // invoke.
              new BranchInst(II->getNormalDest(), II);
              
              // Finally, delete the invoke instruction!
              I->getInstList().pop_back();
              
              ++NumRemoved;
              MadeChange = true;
            }
  }

  return MadeChange; 
}
