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
using std::map;
using std::vector;

static RegisterAnalysis<TDDataStructures>
Y("tddatastructure", "Top-down Data Structure Analysis Closure");

// releaseMemory - If the pass pipeline is done with this pass, we can release
// our memory... here...
//
void TDDataStructures::releaseMemory() {
  for (map<const Function*, DSGraph*>::iterator I = DSInfo.begin(),
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

#if 0

// MergeGlobalNodes - Merge all existing global nodes with globals
// inlined from the callee or with globals from the GlobalsGraph.
//
static void MergeGlobalNodes(DSGraph &Graph,
                             map<Value*, DSNodeHandle> &OldValMap) {
  map<Value*, DSNodeHandle> &ValMap = Graph.getValueMap();
  for (map<Value*, DSNodeHandle>::iterator I = ValMap.begin(), E = ValMap.end();
       I != E; ++I)
    if (GlobalValue* GV = dyn_cast<GlobalValue>(I->first)) {
      map<Value*, DSNodeHandle>:: iterator NHI = OldValMap.find(GV);
      if (NHI != OldValMap.end())       // was it inlined from the callee?
        I->second->mergeWith(NHI->second);
      else                              // get it from the GlobalsGraph
        I->second->mergeWith(Graph.cloneGlobalInto(GV));
    }

  // Add unused inlined global nodes into the value map
  for (map<Value*, DSNodeHandle>::iterator I = OldValMap.begin(),
         E = OldValMap.end(); I != E; ++I)
    if (isa<GlobalValue>(I->first)) {
      DSNodeHandle &NH = ValMap[I->first];  // If global is not in ValMap...
      if (NH == 0)
        NH = I->second;                     // Add the one just inlined.
    }
}

#endif

/// ResolveCallSite - This method is used to link the actual arguments together
/// with the formal arguments for a function call in the top-down closure.  This
/// method assumes that the call site arguments have been mapped into nodes
/// local to the specified graph.
///
void TDDataStructures::ResolveCallSite(DSGraph &Graph,
                                   const BUDataStructures::CallSite &CallSite) {
  // Resolve all of the function formal arguments...
  Function &F = Graph.getFunction();
  Function::aiterator AI = F.abegin();

  for (unsigned i = 2, e = CallSite.Context.size(); i != e; ++i, ++AI) {
    // Advance the argument iterator to the first pointer argument...
    while (!DataStructureAnalysis::isPointerType(AI->getType())) ++AI;
    
    // TD ...Merge the formal arg scalar with the actual arg node
    DSNodeHandle &NodeForFormal = Graph.getNodeForValue(AI);
    if (NodeForFormal.getNode())
      NodeForFormal.mergeWith(CallSite.Context[i]);
  }
  
  // Merge returned node in the caller with the "return" node in callee
  if (CallSite.Context[0].getNode() && Graph.getRetNode().getNode())
    Graph.getRetNode().mergeWith(CallSite.Context[0]);
}

DSGraph &TDDataStructures::calculateGraph(Function &F) {
  // Make sure this graph has not already been calculated, or that we don't get
  // into an infinite loop with mutually recursive functions.
  //
  DSGraph *&Graph = DSInfo[&F];
  if (Graph) return *Graph;

  BUDataStructures &BU = getAnalysis<BUDataStructures>();
  DSGraph &BUGraph = BU.getDSGraph(F);
  Graph = new DSGraph(BUGraph);

  const vector<BUDataStructures::CallSite> *CallSitesP = BU.getCallSites(F);
  if (CallSitesP == 0) {
    DEBUG(std::cerr << "  [TD] No callers for: " << F.getName() << "\n");
    return *Graph;  // If no call sites, the graph is the same as the BU graph!
  }

  // Loop over all call sites of this function, merging each one into this
  // graph.
  //
  DEBUG(std::cerr << "  [TD] Inlining callers for: " << F.getName() << "\n");
  const vector<BUDataStructures::CallSite> &CallSites = *CallSitesP;
  for (unsigned c = 0, ce = CallSites.size(); c != ce; ++c) {
    const BUDataStructures::CallSite &CallSite = CallSites[c];  // Copy
    Function &Caller = *CallSite.Caller;
    assert(!Caller.isExternal() && "Externals function cannot 'call'!");
    
    DEBUG(std::cerr << "\t [TD] Inlining caller #" << c << " '"
          << Caller.getName() << "' into callee: " << F.getName() << "\n");
    
    if (&Caller == &F) {
      // Self-recursive call: this can happen after a cycle of calls is inlined.
      ResolveCallSite(*Graph, CallSite);
    } else {
      // Recursively compute the graph for the Caller.  That should
      // be fully resolved except if there is mutual recursion...
      //
      DSGraph &CG = calculateGraph(Caller);  // Graph to inline
      
      DEBUG(std::cerr << "\t\t[TD] Got graph for " << Caller.getName()
                      << " in: " << F.getName() << "\n");

      // These two maps keep track of where scalars in the old graph _used_
      // to point to, and of new nodes matching nodes of the old graph.
      std::map<Value*, DSNodeHandle> OldValMap;
      std::map<const DSNode*, DSNode*> OldNodeMap;

      // Clone the Caller's graph into the current graph, keeping
      // track of where scalars in the old graph _used_ to point...
      // Do this here because it only needs to happens once for each Caller!
      // Strip scalars but not allocas since they are alive in callee.
      // 
      DSNodeHandle RetVal = Graph->cloneInto(CG, OldValMap, OldNodeMap,
                                             /*StripScalars*/ true,
                                             /*StripAllocas*/ false,
                                             /*CopyCallers*/  true,
                                             /*CopyOrigCalls*/false);

      // Make a temporary copy of the call site, and transform the argument node
      // pointers.
      BUDataStructures::CallSite TmpCallSite = CallSite;
      for (unsigned i = 0, e = CallSite.Context.size(); i != e; ++i) {
        const DSNode *OldNode = TmpCallSite.Context[i].getNode();
        TmpCallSite.Context[i].setNode(OldNodeMap[OldNode]);
      }

      ResolveCallSite(*Graph, CallSite);

#if 0
      // If its not a self-recursive call, merge global nodes in the inlined
      // graph with the corresponding global nodes in the current graph
      if (&caller != &callee)
        MergeGlobalNodes(calleeGraph, OldValMap);
#endif
    }
  }
  

#if 0
  // Recompute the Incomplete markers and eliminate unreachable nodes.
  Graph->maskIncompleteMarkers();
  Graph->markIncompleteNodes(/*markFormals*/ ! F.hasInternalLinkage()
                             /*&& FIXME: NEED TO CHECK IF ALL CALLERS FOUND!*/);
  Graph->removeDeadNodes(/*KeepAllGlobals*/ false, /*KeepCalls*/ false);
#endif

  DEBUG(std::cerr << "  [TD] Done inlining callers for: " << F.getName() << " ["
        << Graph->getGraphSize() << "+" << Graph->getFunctionCalls().size()
        << "]\n");

  return *Graph;
}
