//===-- GlobalDCE.cpp - DCE unreachable internal functions ----------------===//
//
// This transform is designed to eliminate unreachable internal globals
// FIXME: GlobalDCE should update the callgraph, not destroy it!
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Analysis/CallGraph.h"
#include "Support/DepthFirstIterator.h"
#include "Support/StatisticReporter.h"

static Statistic<> NumFunctions("globaldce\t- Number of functions removed");
static Statistic<> NumVariables("globaldce\t- Number of global variables removed");

static bool RemoveUnreachableFunctions(Module &M, CallGraph &CallGraph) {
  // Calculate which functions are reachable from the external functions in the
  // call graph.
  //
  std::set<CallGraphNode*> ReachableNodes(df_begin(&CallGraph),
                                          df_end(&CallGraph));

  // Loop over the functions in the module twice.  The first time is used to
  // drop references that functions have to each other before they are deleted.
  // The second pass removes the functions that need to be removed.
  //
  std::vector<CallGraphNode*> FunctionsToDelete;   // Track unused functions
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    CallGraphNode *N = CallGraph[I];

    if (!ReachableNodes.count(N)) {              // Not reachable??
      I->dropAllReferences();
      N->removeAllCalledFunctions();
      FunctionsToDelete.push_back(N);
      ++NumFunctions;
    }
  }

  // Nothing to do if no unreachable functions have been found...
  if (FunctionsToDelete.empty()) return false;

  // Unreachables functions have been found and should have no references to
  // them, delete them now.
  //
  for (std::vector<CallGraphNode*>::iterator I = FunctionsToDelete.begin(),
	 E = FunctionsToDelete.end(); I != E; ++I)
    delete CallGraph.removeFunctionFromModule(*I);

  return true;
}

static bool RemoveUnreachableGlobalVariables(Module &M) {
  bool Changed = false;
  // Eliminate all global variables that are unused, and that are internal, or
  // do not have an initializer.
  //
  for (Module::giterator I = M.gbegin(); I != M.gend(); )
    if (!I->use_empty() || (I->hasExternalLinkage() && I->hasInitializer()))
      ++I;                     // Cannot eliminate global variable
    else {
      I = M.getGlobalList().erase(I);
      ++NumVariables;
      Changed = true;
    }
  return Changed;
}

namespace {
  struct GlobalDCE : public Pass {
    const char *getPassName() const { return "Dead Global Elimination"; }

    // run - Do the GlobalDCE pass on the specified module, optionally updating
    // the specified callgraph to reflect the changes.
    //
    bool run(Module &M) {
      return RemoveUnreachableFunctions(M, getAnalysis<CallGraph>()) |
             RemoveUnreachableGlobalVariables(M);
    }

    // getAnalysisUsage - This function works on the call graph of a module.
    // It is capable of updating the call graph to reflect the new state of the
    // module.
    //
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired(CallGraph::ID);
    }
  };
}

Pass *createGlobalDCEPass() { return new GlobalDCE(); }
