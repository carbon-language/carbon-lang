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
#include "Support/StatisticReporter.h"
using std::map;

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


// ResolveArguments - Resolve the formal and actual arguments for a function
// call by merging the nodes for the actual arguments at the call site Call[] 
// (these were copied over from the caller's graph into the callee's graph
// by cloneInto, and the nodes can be found from OldNodeMap) with the
// corresponding nodes for the formal arguments in the callee.
// 
static void ResolveArguments(std::vector<DSNodeHandle> &Call,
                             Function &callee,
                             std::map<Value*, DSNodeHandle> &CalleeValueMap,
                             std::map<const DSNode*, DSNode*> OldNodeMap,
                             bool ignoreNodeMap) {
  // Resolve all of the function formal arguments...
  Function::aiterator AI = callee.abegin();
  for (unsigned i = 2, e = Call.size(); i != e; ++i) {
    // Advance the argument iterator to the first pointer argument...
    while (!isa<PointerType>(AI->getType())) ++AI;
    
    // TD ...Merge the formal arg scalar with the actual arg node
    DSNode* actualArg = Call[i];
    DSNode *nodeForActual = ignoreNodeMap? actualArg : OldNodeMap[actualArg];
    assert(nodeForActual && "No node found for actual arg in callee graph!");
    
    DSNode *nodeForFormal = CalleeValueMap[AI]->getLink(0);
    if (nodeForFormal)
      nodeForFormal->mergeWith(nodeForActual);
    ++AI;
  }
}

// MergeGlobalNodes - Merge all existing global nodes with globals
// inlined from the callee or with globals from the GlobalsGraph.
//
static void MergeGlobalNodes(DSGraph& Graph,
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

// Helper function to push a caller's graph into the calleeGraph,
// once per call between the caller and the callee.
// Remove each such resolved call from the OrigFunctionCalls vector.
// NOTE: This may produce O(N^2) behavior because it uses linear search
// through the vector OrigFunctionCalls to find all calls to this callee.
// 
void TDDataStructures::pushGraphIntoCallee(DSGraph &callerGraph,
                                           DSGraph &calleeGraph,
                                 std::map<Value*, DSNodeHandle> &OldValMap,
                                 std::map<const DSNode*, DSNode*> &OldNodeMap) {

  Function& caller = callerGraph.getFunction();

  // Loop over all function calls in the caller to find those to this callee
  std::vector<std::vector<DSNodeHandle> >& FunctionCalls =
    callerGraph.getOrigFunctionCalls();

  for (unsigned i = 0, ei = FunctionCalls.size(); i != ei; ++i) {
    
    std::vector<DSNodeHandle>& Call = FunctionCalls[i];
    assert(Call.size() >= 2 && "No function pointer in Call?");
    DSNodeHandle& callee = Call[1];
    std::vector<GlobalValue*> funcPtrs(callee->getGlobals());

    // Loop over the function pointers in the call, looking for the callee
    for (unsigned f = 0; f != funcPtrs.size(); ++f) {

      // Must be a function type, so this cast MUST succeed.
      Function &callee = cast<Function>(*funcPtrs[f]);
      if (&callee != &calleeGraph.getFunction())
        continue;

      // Found a call to the callee.  Inline its graph
      // copy caller pointer because inlining may modify the callers vector

      // Merge actual arguments from the caller with formals in the callee.
      // Don't use the old->new node map if this is a self-recursive call.
      ResolveArguments(Call, callee, calleeGraph.getValueMap(), OldNodeMap,
                       /*ignoreNodeMap*/ &caller == &callee);

      // If its not a self-recursive call, merge global nodes in the inlined
      // graph with the corresponding global nodes in the current graph
      if (&caller != &callee)
        MergeGlobalNodes(calleeGraph, OldValMap);

      // Merge returned node in the caller with the "return" node in callee
      if (Call[0])
        calleeGraph.getRetNode()->mergeWith(OldNodeMap[Call[0]]);
      
      // Erase the entry in the globals vector
      funcPtrs.erase(funcPtrs.begin()+f--);
      
    } // for each function pointer in the call node
  } // for each original call node
}


DSGraph &TDDataStructures::calculateGraph(Function &F) {
  // Make sure this graph has not already been calculated, or that we don't get
  // into an infinite loop with mutually recursive functions.
  //
  DSGraph *&Graph = DSInfo[&F];
  if (Graph) return *Graph;

  // Copy the local version into DSInfo...
  DSGraph& BUGraph = getAnalysis<BUDataStructures>().getDSGraph(F);
  Graph = new DSGraph(BUGraph);

  // Find the callers of this function recorded during the BU pass
  std::set<Function*> &PendingCallers = BUGraph.getPendingCallers();

  DEBUG(std::cerr << "  [TD] Inlining callers for: " << F.getName() << "\n");

  for (std::set<Function*>::iterator I=PendingCallers.begin(),
         E=PendingCallers.end(); I != E; ++I) {
    Function& caller = **I;
    assert(! caller.isExternal() && "Externals unexpected in callers list");
    
    DEBUG(std::cerr << "\t [TD] Inlining " << caller.getName()
                    << " into callee: " << F.getName() << "\n");
    
    // These two maps keep track of where scalars in the old graph _used_
    // to point to, and of new nodes matching nodes of the old graph.
    // These remain empty if no other graph is inlined (i.e., self-recursive).
    std::map<const DSNode*, DSNode*> OldNodeMap;
    std::map<Value*, DSNodeHandle> OldValMap;
    
    if (&caller == &F) {
      // Self-recursive call: this can happen after a cycle of calls is inlined.
      pushGraphIntoCallee(*Graph, *Graph, OldValMap, OldNodeMap);
    }
    else {
      // Recursively compute the graph for the caller.  That should
      // be fully resolved except if there is mutual recursion...
      //
      DSGraph &callerGraph = calculateGraph(caller);  // Graph to inline
      
      DEBUG(std::cerr << "\t\t[TD] Got graph for " << caller.getName()
                      << " in: " << F.getName() << "\n");

      // Clone the caller's graph into the current graph, keeping
      // track of where scalars in the old graph _used_ to point...
      // Do this here because it only needs to happens once for each caller!
      // Strip scalars but not allocas since they are visible in callee.
      // 
      DSNode *RetVal = Graph->cloneInto(callerGraph, OldValMap, OldNodeMap,
                                        /*StripScalars*/   true,
                                        /*StripAllocas*/   false,
                                        /*CopyCallers*/    true,
                                        /*CopyOrigCalls*/  false);

      pushGraphIntoCallee(callerGraph, *Graph, OldValMap, OldNodeMap);
    }
  }

  // Recompute the Incomplete markers and eliminate unreachable nodes.
  Graph->maskIncompleteMarkers();
  Graph->markIncompleteNodes(/*markFormals*/ ! F.hasInternalLinkage()
                             /*&& FIXME: NEED TO CHECK IF ALL CALLERS FOUND!*/);
  Graph->removeDeadNodes(/*KeepAllGlobals*/ false, /*KeepCalls*/ false);

  DEBUG(std::cerr << "  [TD] Done inlining callers for: " << F.getName() << " ["
        << Graph->getGraphSize() << "+" << Graph->getFunctionCalls().size()
        << "]\n");

  return *Graph;
}
