//===- TopDownClosure.cpp - Compute the top-down interprocedure closure ---===//
//
// This file implements the TDDataStructures class, which represents the
// Top-down Interprocedural closure of the data structure graph over the
// program.  This is useful (but not strictly necessary?) for applications
// like pointer analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DSGraph.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "Support/Statistic.h"

static RegisterAnalysis<TDDataStructures>
Y("tddatastructure", "Top-down Data Structure Analysis Closure");

// releaseMemory - If the pass pipeline is done with this pass, we can release
// our memory... here...
//
void TDDataStructures::releaseMemory() {
  BUMaps.clear();
  for (std::map<const Function*, DSGraph*>::iterator I = DSInfo.begin(),
         E = DSInfo.end(); I != E; ++I)
    delete I->second;

  // Empty map so next time memory is released, data structures are not
  // re-deleted.
  DSInfo.clear();
}

// run - Calculate the top down data structure graphs for each function in the
// program.
//
bool TDDataStructures::run(Module &M) {
  // Simply calculate the graphs for each function...
  for (Module::reverse_iterator I = M.rbegin(), E = M.rend(); I != E; ++I)
    if (!I->isExternal())
      calculateGraph(*I);
  return false;
}

/// ResolveCallSite - This method is used to link the actual arguments together
/// with the formal arguments for a function call in the top-down closure.  This
/// method assumes that the call site arguments have been mapped into nodes
/// local to the specified graph.
///
void TDDataStructures::ResolveCallSite(DSGraph &Graph,
                                       const DSCallSite &CallSite) {
  // Resolve all of the function formal arguments...
  Function &F = Graph.getFunction();
  Function::aiterator AI = F.abegin();

  for (unsigned i = 0, e = CallSite.getNumPtrArgs(); i != e; ++i, ++AI) {
    // Advance the argument iterator to the first pointer argument...
    while (!DataStructureAnalysis::isPointerType(AI->getType())) ++AI;
    
    // TD ...Merge the formal arg scalar with the actual arg node
    DSNodeHandle &NodeForFormal = Graph.getNodeForValue(AI);
    assert(NodeForFormal.getNode() && "Pointer argument has no dest node!");
    NodeForFormal.mergeWith(CallSite.getPtrArg(i));
  }
  
  // Merge returned node in the caller with the "return" node in callee
  if (CallSite.getRetVal().getNode() && Graph.getRetNode().getNode())
    Graph.getRetNode().mergeWith(CallSite.getRetVal());
}


DSGraph &TDDataStructures::calculateGraph(Function &F) {
  // Make sure this graph has not already been calculated, or that we don't get
  // into an infinite loop with mutually recursive functions.
  //
  DSGraph *&Graph = DSInfo[&F];
  if (Graph) return *Graph;

  BUDataStructures &BU = getAnalysis<BUDataStructures>();
  DSGraph &BUGraph = BU.getDSGraph(F);
  
  // Copy the BU graph, keeping a mapping from the BUGraph to the current Graph
  std::map<const DSNode*, DSNode*> BUNodeMap;
  Graph = new DSGraph(BUGraph, BUNodeMap);

  // Convert the mapping from a node-to-node map into a node-to-nodehandle map
  BUMaps[&F].insert(BUNodeMap.begin(), BUNodeMap.end());
  BUNodeMap.clear();   // We are done with the temporary map.

  const std::vector<DSCallSite> *CallSitesP = BU.getCallSites(F);
  if (CallSitesP == 0) {
    DEBUG(std::cerr << "  [TD] No callers for: " << F.getName() << "\n");
    return *Graph;  // If no call sites, the graph is the same as the BU graph!
  }

  // Loop over all call sites of this function, merging each one into this
  // graph.
  //
  DEBUG(std::cerr << "  [TD] Inlining callers for: " << F.getName() << "\n");
  const std::vector<DSCallSite> &CallSites = *CallSitesP;
  for (unsigned c = 0, ce = CallSites.size(); c != ce; ++c) {
    const DSCallSite &CallSite = CallSites[c];
    Function &Caller = *CallSite.getResolvingCaller();
    assert(&Caller && !Caller.isExternal() &&
           "Externals function cannot 'call'!");
    
    DEBUG(std::cerr << "\t [TD] Inlining caller #" << c << " '"
          << Caller.getName() << "' into callee: " << F.getName() << "\n");
    
    if (&Caller == &F) {
      // Self-recursive call: this can happen after a cycle of calls is inlined.
      ResolveCallSite(*Graph, CallSite);
    } else {

      // Recursively compute the graph for the Caller.  It should be fully
      // resolved except if there is mutual recursion...
      //
      DSGraph &CG = calculateGraph(Caller);  // Graph to inline
      
      DEBUG(std::cerr << "\t\t[TD] Got graph for " << Caller.getName()
                      << " in: " << F.getName() << "\n");

      // These two maps keep track of where scalars in the old graph _used_
      // to point to, and of new nodes matching nodes of the old graph.
      std::map<Value*, DSNodeHandle> OldValMap;
      std::map<const DSNode*, DSNode*> OldNodeMap;

      // Translate call site from having links into the BU graph
      DSCallSite CallSiteInCG(CallSite, BUMaps[&Caller]);

      // Clone the Caller's graph into the current graph, keeping
      // track of where scalars in the old graph _used_ to point...
      // Do this here because it only needs to happens once for each Caller!
      // Strip scalars but not allocas since they are alive in callee.
      // 
      DSNodeHandle RetVal = Graph->cloneInto(CG, OldValMap, OldNodeMap,
                                             /*StripScalars*/ true,
                                             /*StripAllocas*/ false);
      ResolveCallSite(*Graph, DSCallSite(CallSiteInCG, OldNodeMap));
    }
  }

  // Recompute the Incomplete markers and eliminate unreachable nodes.
#if 0
  Graph->maskIncompleteMarkers();
  Graph->markIncompleteNodes(/*markFormals*/ !F.hasInternalLinkage()
                             /*&& FIXME: NEED TO CHECK IF ALL CALLERS FOUND!*/);
  Graph->removeDeadNodes(/*KeepAllGlobals*/ false, /*KeepCalls*/ false);
#endif
  DEBUG(std::cerr << "  [TD] Done inlining callers for: " << F.getName() << " ["
        << Graph->getGraphSize() << "+" << Graph->getFunctionCalls().size()
        << "]\n");

  return *Graph;
}
