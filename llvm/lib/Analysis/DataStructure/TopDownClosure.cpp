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
}

// run - Calculate the top down data structure graphs for each function in the
// program.
//
bool TDDataStructures::run(Module &M) {
  BUDataStructures &BU = getAnalysis<BUDataStructures>();
  GlobalsGraph = new DSGraph(BU.getGlobalsGraph());

  // Calculate top-down from main...
  if (Function *F = M.getMainFunction())
    calculateGraphFrom(*F);

  // Next calculate the graphs for each function unreachable function...
  for (Module::reverse_iterator I = M.rbegin(), E = M.rend(); I != E; ++I)
    if (!I->isExternal() && !DSInfo.count(&*I))
      calculateGraphFrom(*I);

  return false;
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


/// FunctionHasCompleteArguments - This function returns true if it is safe not
/// to mark arguments to the function complete.
///
/// FIXME: Need to check if all callers have been found, or rather if a
/// funcpointer escapes!
///
static bool FunctionHasCompleteArguments(Function &F) {
  return F.hasInternalLinkage();
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



void TDDataStructures::calculateGraphFrom(Function &F) {
  // We want to traverse the call graph in reverse post-order.  To do this, we
  // calculate a post-order traversal, then reverse it.
  hash_set<DSGraph*> VisitedGraph;
  std::vector<DSGraph*> PostOrder;
  ComputePostOrder(F, VisitedGraph, PostOrder,
                   getAnalysis<BUDataStructures>().getActualCallees());
  VisitedGraph.clear();   // Release memory!

  // Visit each of the graphs in reverse post-order now!
  while (!PostOrder.empty()) {
    inlineGraphIntoCallees(*PostOrder.back());
    PostOrder.pop_back();
  }
}


void TDDataStructures::inlineGraphIntoCallees(DSGraph &Graph) {
  // Recompute the Incomplete markers and eliminate unreachable nodes.
  Graph.maskIncompleteMarkers();
  unsigned Flags = true /* FIXME!! FunctionHasCompleteArguments(F)*/ ?
                            DSGraph::IgnoreFormalArgs : DSGraph::MarkFormalArgs;
  Graph.markIncompleteNodes(Flags | DSGraph::IgnoreGlobals);
  Graph.removeDeadNodes(DSGraph::RemoveUnreachableGlobals);

  DSCallSiteIterator CalleeI = DSCallSiteIterator::begin_std(Graph);
  DSCallSiteIterator CalleeE = DSCallSiteIterator::end_std(Graph);

  if (CalleeI == CalleeE) {
    DEBUG(std::cerr << "  [TD] No callees for: " << Graph.getFunctionNames()
                    << "\n");
    return;
  }

  // Loop over all of the call sites, building a multi-map from Callees to
  // DSCallSite*'s.  With this map we can then loop over each callee, cloning
  // this graph once into it, then resolving arguments.
  //
  std::multimap<std::pair<DSGraph*,Function*>, const DSCallSite*> CalleeSites;
  for (; CalleeI != CalleeE; ++CalleeI)
    if (!(*CalleeI)->isExternal()) {
      // We should have already created the graph here...
      if (!DSInfo.count(*CalleeI))
        std::cerr << "WARNING: TD pass, did not know about callee: '"
                  << (*CalleeI)->getName() << "'\n";

      DSGraph &IG = getOrCreateDSGraph(**CalleeI);
      if (&IG != &Graph)
        CalleeSites.insert(std::make_pair(std::make_pair(&IG, *CalleeI),
                                          &CalleeI.getCallSite()));
    }

  // Now that we have information about all of the callees, propagate the
  // current graph into the callees.
  //
  DEBUG(std::cerr << "  [TD] Inlining '" << Graph.getFunctionNames() <<"' into "
                  << CalleeSites.size() << " callees.\n");

  // Loop over all the callees...
  for (std::multimap<std::pair<DSGraph*, Function*>,
         const DSCallSite*>::iterator I = CalleeSites.begin(),
         E = CalleeSites.end(); I != E; ) {
    DSGraph &CG = *I->first.first;

    DEBUG(std::cerr << "     [TD] Inlining graph into callee graph '"
                    << CG.getFunctionNames() << "'\n");
    
    // Clone our current graph into the callee...
    DSGraph::ScalarMapTy OldValMap;
    DSGraph::NodeMapTy OldNodeMap;
    DSGraph::ReturnNodesTy ReturnNodes;
    CG.cloneInto(Graph, OldValMap, ReturnNodes, OldNodeMap,
                 DSGraph::StripModRefBits |
                 DSGraph::KeepAllocaBit | DSGraph::DontCloneCallNodes |
                 DSGraph::DontCloneAuxCallNodes);
    OldValMap.clear();  // We don't care about the ValMap
    ReturnNodes.clear();  // We don't care about return values either
    
    // Loop over all of the invocation sites of the callee, resolving
    // arguments to our graph.  This loop may iterate multiple times if the
    // current function calls this callee multiple times with different
    // signatures.
    //
    for (; I != E && I->first.first == &CG; ++I) {
      Function &Callee = *I->first.second;
      DEBUG(std::cerr << "\t   [TD] Merging args for callee '"
                      << Callee.getName() << "'\n");

      // Map call site into callee graph
      DSCallSite NewCS(*I->second, OldNodeMap);
        
      // Resolve the return values...
      NewCS.getRetVal().mergeWith(CG.getReturnNodeFor(Callee));
        
      // Resolve all of the arguments...
      Function::aiterator AI = Callee.abegin();
      for (unsigned i = 0, e = NewCS.getNumPtrArgs();
           i != e && AI != Callee.aend(); ++i, ++AI) {
        // Advance the argument iterator to the first pointer argument...
        while (AI != Callee.aend() && !DS::isPointerType(AI->getType()))
          ++AI;
        if (AI == Callee.aend()) break;

        // Add the link from the argument scalar to the provided value
        DSNodeHandle &NH = CG.getNodeForValue(AI);
        assert(NH.getNode() && "Pointer argument without scalarmap entry?");
        NH.mergeWith(NewCS.getPtrArg(i));
      }
    }

    // Done with the nodemap...
    OldNodeMap.clear();

    // Recompute the Incomplete markers and eliminate unreachable nodes.
    CG.removeTriviallyDeadNodes();
    //CG.maskIncompleteMarkers();
    //CG.markIncompleteNodes(DSGraph::MarkFormalArgs | DSGraph::IgnoreGlobals);
    //CG.removeDeadNodes(DSGraph::RemoveUnreachableGlobals);
  }

  DEBUG(std::cerr << "  [TD] Done inlining into callees for: "
        << Graph.getFunctionNames() << " [" << Graph.getGraphSize() << "+"
        << Graph.getFunctionCalls().size() << "]\n");

#if 0
  // Loop over all the callees... making sure they are all resolved now...
  Function *LastFunc = 0;
  for (std::multimap<Function*, const DSCallSite*>::iterator
         I = CalleeSites.begin(), E = CalleeSites.end(); I != E; ++I)
    if (I->first != LastFunc) {  // Only visit each callee once...
      LastFunc = I->first;
      calculateGraph(*I->first);
    }
#endif
}

