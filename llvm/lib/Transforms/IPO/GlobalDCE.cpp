//===-- GlobalDCE.cpp - DCE unreachable internal methods ---------*- C++ -*--=//
//
// This transform is designed to eliminate unreachable internal globals
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "Support/DepthFirstIterator.h"
#include <set>

static bool RemoveUnreachableMethods(Module *M, cfg::CallGraph &CallGraph) {
  // Calculate which methods are reachable from the external methods in the call
  // graph.
  //
  std::set<cfg::CallGraphNode*> ReachableNodes(df_begin(&CallGraph),
                                               df_end(&CallGraph));

  // Loop over the methods in the module twice.  The first time is used to drop
  // references that methods have to each other before they are deleted.  The
  // second pass removes the methods that need to be removed.
  //
  std::vector<cfg::CallGraphNode*> MethodsToDelete;   // Track unused methods
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I) {
    cfg::CallGraphNode *N = CallGraph[*I];
    if (!ReachableNodes.count(N)) {              // Not reachable??
      (*I)->dropAllReferences();
      N->removeAllCalledMethods();
      MethodsToDelete.push_back(N);
    }
  }

  // Nothing to do if no unreachable methods have been found...
  if (MethodsToDelete.empty()) return false;

  // Unreachables methods have been found and should have no references to them,
  // delete them now.
  //
  for (std::vector<cfg::CallGraphNode*>::iterator I = MethodsToDelete.begin(),
	 E = MethodsToDelete.end(); I != E; ++I)
    delete CallGraph.removeMethodFromModule(*I);

  return true;
}

bool GlobalDCE::run(Module *M) {
  // TODO: FIXME: GET THE CALL GRAPH FROM THE PASS!
  // Create a call graph if one is not already available...
  cfg::CallGraph CallGraph(M);
  return RemoveUnreachableMethods(M, CallGraph);
}
