//===- TopDownClosure.cpp - Compute the top-down interprocedure closure ---===//
//
// This file implements the TDDataStructures class, which represents the
// Top-down Interprocedural closure of the data structure graph over the
// program.  This is useful (but not strictly necessary?) for applications
// like pointer analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "Support/Statistic.h"
#include "DSCallSiteIterator.h"

namespace {
  RegisterAnalysis<TDDataStructures>   // Register the pass
  Y("tddatastructure", "Top-down Data Structure Analysis");

  Statistic<> NumTDInlines("tddatastructures", "Number of graphs inlined");
}

/// FunctionHasCompleteArguments - This function returns true if it is safe not
/// to mark arguments to the function complete.
///
/// FIXME: Need to check if all callers have been found, or rather if a
/// funcpointer escapes!
///
static bool FunctionHasCompleteArguments(Function &F) {
  return F.hasInternalLinkage();
}

// run - Calculate the top down data structure graphs for each function in the
// program.
//
bool TDDataStructures::run(Module &M) {
  BUDataStructures &BU = getAnalysis<BUDataStructures>();
  GlobalsGraph = new DSGraph(BU.getGlobalsGraph());

  // Figure out which functions must not mark their arguments complete because
  // they are accessible outside this compilation unit.
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!FunctionHasCompleteArguments(*I))
      ArgsRemainIncomplete.insert(I);

  // We want to traverse the call graph in reverse post-order.  To do this, we
  // calculate a post-order traversal, then reverse it.
  hash_set<DSGraph*> VisitedGraph;
  std::vector<DSGraph*> PostOrder;
  const BUDataStructures::ActualCalleesTy &ActualCallees = 
    getAnalysis<BUDataStructures>().getActualCallees();

  // Calculate top-down from main...
  if (Function *F = M.getMainFunction())
    ComputePostOrder(*F, VisitedGraph, PostOrder, ActualCallees);                     

  // Next calculate the graphs for each function unreachable function...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    ComputePostOrder(*I, VisitedGraph, PostOrder, ActualCallees);

  VisitedGraph.clear();   // Release memory!

  // Visit each of the graphs in reverse post-order now!
  while (!PostOrder.empty()) {
    inlineGraphIntoCallees(*PostOrder.back());
    PostOrder.pop_back();
  }

  ArgsRemainIncomplete.clear();
  return false;
}


DSGraph &TDDataStructures::getOrCreateDSGraph(Function &F) {
  DSGraph *&G = DSInfo[&F];
  if (G == 0) { // Not created yet?  Clone BU graph...
    G = new DSGraph(getAnalysis<BUDataStructures>().getDSGraph(F));
    G->getAuxFunctionCalls().clear();
    G->setPrintAuxCalls();
    G->setGlobalsGraph(GlobalsGraph);
  }
  return *G;
}


void TDDataStructures::ComputePostOrder(Function &F,hash_set<DSGraph*> &Visited,
                                        std::vector<DSGraph*> &PostOrder,
                      const BUDataStructures::ActualCalleesTy &ActualCallees) {
  if (F.isExternal()) return;
  DSGraph &G = getOrCreateDSGraph(F);
  if (Visited.count(&G)) return;
  Visited.insert(&G);
  
  // Recursively traverse all of the callee graphs.
  const std::vector<DSCallSite> &FunctionCalls = G.getFunctionCalls();

  for (unsigned i = 0, e = FunctionCalls.size(); i != e; ++i) {
    std::pair<BUDataStructures::ActualCalleesTy::const_iterator,
      BUDataStructures::ActualCalleesTy::const_iterator>
         IP = ActualCallees.equal_range(&FunctionCalls[i].getCallInst());

    for (BUDataStructures::ActualCalleesTy::const_iterator I = IP.first;
         I != IP.second; ++I)
      ComputePostOrder(*I->second, Visited, PostOrder, ActualCallees);
  }

  PostOrder.push_back(&G);
}





// releaseMemory - If the pass pipeline is done with this pass, we can release
// our memory... here...
//
// FIXME: This should be releaseMemory and will work fine, except that LoadVN
// has no way to extend the lifetime of the pass, which screws up ds-aa.
//
void TDDataStructures::releaseMyMemory() {
  for (hash_map<Function*, DSGraph*>::iterator I = DSInfo.begin(),
         E = DSInfo.end(); I != E; ++I) {
    I->second->getReturnNodes().erase(I->first);
    if (I->second->getReturnNodes().empty())
      delete I->second;
  }

  // Empty map so next time memory is released, data structures are not
  // re-deleted.
  DSInfo.clear();
  delete GlobalsGraph;
  GlobalsGraph = 0;
}

void TDDataStructures::inlineGraphIntoCallees(DSGraph &Graph) {
  // Recompute the Incomplete markers and eliminate unreachable nodes.
  Graph.removeTriviallyDeadNodes();
  Graph.maskIncompleteMarkers();

  // If any of the functions has incomplete incoming arguments, don't mark any
  // of them as complete.
  bool HasIncompleteArgs = false;
  const DSGraph::ReturnNodesTy &GraphReturnNodes = Graph.getReturnNodes();
  for (DSGraph::ReturnNodesTy::const_iterator I = GraphReturnNodes.begin(),
         E = GraphReturnNodes.end(); I != E; ++I)
    if (ArgsRemainIncomplete.count(I->first)) {
      HasIncompleteArgs = true;
      break;
    }
  
  unsigned Flags
    = HasIncompleteArgs ? DSGraph::MarkFormalArgs : DSGraph::IgnoreFormalArgs;
  Graph.markIncompleteNodes(Flags | DSGraph::IgnoreGlobals);
  Graph.removeDeadNodes(DSGraph::RemoveUnreachableGlobals);

  const std::vector<DSCallSite> &FunctionCalls = Graph.getFunctionCalls();
  if (FunctionCalls.empty()) {
    DEBUG(std::cerr << "  [TD] No callees for: " << Graph.getFunctionNames()
                    << "\n");
    return;
  }

  // Now that we have information about all of the callees, propagate the
  // current graph into the callees.
  //
  DEBUG(std::cerr << "  [TD] Inlining '" << Graph.getFunctionNames() <<"' into "
                  << FunctionCalls.size() << " call nodes.\n");

  const BUDataStructures::ActualCalleesTy &ActualCallees =
    getAnalysis<BUDataStructures>().getActualCallees();

  // Only inline this function into each real callee once.  After that, just
  // merge information into arguments...
  hash_map<DSGraph*, DSGraph::NodeMapTy> InlinedSites;

  // Loop over all the callees... cloning this graph into each one exactly once,
  // keeping track of the node mapping information...
  for (unsigned i = 0, e = FunctionCalls.size(); i != e; ++i) {
    // Inline this graph into each function in the invoked function list.
    std::pair<BUDataStructures::ActualCalleesTy::const_iterator,
      BUDataStructures::ActualCalleesTy::const_iterator>
          IP = ActualCallees.equal_range(&FunctionCalls[i].getCallInst());

    int NumArgs = 0;
    if (IP.first != IP.second) {
      NumArgs = IP.first->second->getFunctionType()->getNumParams();
      for (BUDataStructures::ActualCalleesTy::const_iterator I = IP.first;
           I != IP.second; ++I)
        if (NumArgs != (int)I->second->getFunctionType()->getNumParams()) {
          NumArgs = -1;
          break;
        }
    }
    
    if (NumArgs == -1) {
      std::cerr << "ERROR: NONSAME NUMBER OF ARGUMENTS TO CALLEES\n";
    }
 
    for (BUDataStructures::ActualCalleesTy::const_iterator I = IP.first;
         I != IP.second; ++I) {
      DSGraph &CG = getDSGraph(*I->second);
      assert(&CG != &Graph && "TD need not inline graph into self!");

      if (!InlinedSites.count(&CG)) {  // If we haven't already inlined into CG
        DEBUG(std::cerr << "     [TD] Inlining graph into callee graph '"
              << CG.getFunctionNames() << "': " << I->second->getFunctionType()->getNumParams() << " args\n");
        DSGraph::ScalarMapTy OldScalarMap;
        DSGraph::ReturnNodesTy ReturnNodes;
        CG.cloneInto(Graph, OldScalarMap, ReturnNodes, InlinedSites[&CG],
                     DSGraph::StripModRefBits | DSGraph::KeepAllocaBit |
                     DSGraph::DontCloneCallNodes |
                     DSGraph::DontCloneAuxCallNodes);
        ++NumTDInlines;
      }
    }
  }

  // Loop over all the callees...
  for (unsigned i = 0, e = FunctionCalls.size(); i != e; ++i) {
    // Inline this graph into each function in the invoked function list.
    std::pair<BUDataStructures::ActualCalleesTy::const_iterator,
      BUDataStructures::ActualCalleesTy::const_iterator>
          IP = ActualCallees.equal_range(&FunctionCalls[i].getCallInst());
    for (BUDataStructures::ActualCalleesTy::const_iterator I = IP.first;
         I != IP.second; ++I) {
      DSGraph &CG = getDSGraph(*I->second);
      DEBUG(std::cerr << "     [TD] Resolving arguments for callee graph '"
                      << CG.getFunctionNames() << "'\n");

      // Transform our call site information into the cloned version for CG
      DSCallSite CS(FunctionCalls[i], InlinedSites[&CG]);

      // Get the arguments bindings for the called function in CG... and merge
      // them with the cloned graph.
      CG.getCallSiteForArguments(*I->second).mergeWith(CS);
    }
  }

  DEBUG(std::cerr << "  [TD] Done inlining into callees for: "
        << Graph.getFunctionNames() << " [" << Graph.getGraphSize() << "+"
        << Graph.getFunctionCalls().size() << "]\n");
}

