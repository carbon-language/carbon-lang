//===-- GlobalDCE.cpp - DCE unreachable internal functions ----------------===//
//
// This transform is designed to eliminate unreachable internal globals
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "Support/DepthFirstIterator.h"
#include <set>

static bool RemoveUnreachableFunctions(Module *M, CallGraph &CallGraph) {
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
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I) {
    CallGraphNode *N = CallGraph[*I];
    if (!ReachableNodes.count(N)) {              // Not reachable??
      (*I)->dropAllReferences();
      N->removeAllCalledMethods();
      FunctionsToDelete.push_back(N);
    }
  }

  // Nothing to do if no unreachable functions have been found...
  if (FunctionsToDelete.empty()) return false;

  // Unreachables functions have been found and should have no references to
  // them, delete them now.
  //
  for (std::vector<CallGraphNode*>::iterator I = FunctionsToDelete.begin(),
	 E = FunctionsToDelete.end(); I != E; ++I)
    delete CallGraph.removeMethodFromModule(*I);

  return true;
}

namespace {
  struct GlobalDCE : public Pass {
    // run - Do the GlobalDCE pass on the specified module, optionally updating
    // the specified callgraph to reflect the changes.
    //
    bool run(Module *M) {
      return RemoveUnreachableFunctions(M, getAnalysis<CallGraph>());
    }

    // getAnalysisUsageInfo - This function works on the call graph of a module.
    // It is capable of updating the call graph to reflect the new state of the
    // module.
    //
    virtual void getAnalysisUsageInfo(Pass::AnalysisSet &Required,
                                      Pass::AnalysisSet &Destroyed,
                                      Pass::AnalysisSet &Provided) {
      Required.push_back(CallGraph::ID);
      // FIXME: This should update the callgraph, not destroy it!
      Destroyed.push_back(CallGraph::ID);
    }
  };
}

Pass *createGlobalDCEPass() { return new GlobalDCE(); }
