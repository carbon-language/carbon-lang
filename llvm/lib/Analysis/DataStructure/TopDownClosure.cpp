//===- TopDownClosure.cpp - Compute the top-down interprocedure closure ---===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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
#include "llvm/Analysis/DSGraph.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
using namespace llvm;

namespace {
  RegisterAnalysis<TDDataStructures>   // Register the pass
  Y("tddatastructure", "Top-down Data Structure Analysis");

  Statistic<> NumTDInlines("tddatastructures", "Number of graphs inlined");
}

void TDDataStructures::markReachableFunctionsExternallyAccessible(DSNode *N,
                                                   hash_set<DSNode*> &Visited) {
  if (!N || Visited.count(N)) return;
  Visited.insert(N);

  for (unsigned i = 0, e = N->getNumLinks(); i != e; ++i) {
    DSNodeHandle &NH = N->getLink(i*N->getPointerSize());
    if (DSNode *NN = NH.getNode()) {
      const std::vector<GlobalValue*> &Globals = NN->getGlobals();
      for (unsigned G = 0, e = Globals.size(); G != e; ++G)
        if (Function *F = dyn_cast<Function>(Globals[G]))
          ArgsRemainIncomplete.insert(F);

      markReachableFunctionsExternallyAccessible(NN, Visited);
    }
  }
}


// run - Calculate the top down data structure graphs for each function in the
// program.
//
bool TDDataStructures::run(Module &M) {
  BUDataStructures &BU = getAnalysis<BUDataStructures>();
  GlobalsGraph = new DSGraph(BU.getGlobalsGraph());
  GlobalsGraph->setPrintAuxCalls();

  // Figure out which functions must not mark their arguments complete because
  // they are accessible outside this compilation unit.  Currently, these
  // arguments are functions which are reachable by global variables in the
  // globals graph.
  const DSGraph::ScalarMapTy &GGSM = GlobalsGraph->getScalarMap();
  hash_set<DSNode*> Visited;
  for (DSGraph::ScalarMapTy::const_iterator I = GGSM.begin(), E = GGSM.end();
       I != E; ++I)
    if (isa<GlobalValue>(I->first))
      markReachableFunctionsExternallyAccessible(I->second.getNode(), Visited);

  // Loop over unresolved call nodes.  Any functions passed into (but not
  // returned!?) from unresolvable call nodes may be invoked outside of the
  // current module.
  const std::vector<DSCallSite> &Calls = GlobalsGraph->getAuxFunctionCalls();
  for (unsigned i = 0, e = Calls.size(); i != e; ++i) {
    const DSCallSite &CS = Calls[i];
    for (unsigned arg = 0, e = CS.getNumPtrArgs(); arg != e; ++arg)
      markReachableFunctionsExternallyAccessible(CS.getPtrArg(arg).getNode(),
                                                 Visited);
  }
  Visited.clear();

  // Functions without internal linkage also have unknown incoming arguments!
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal() && !I->hasInternalLinkage())
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

  // Next calculate the graphs for each unreachable function...
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
    Instruction *CallI = FunctionCalls[i].getCallSite().getInstruction();
    std::pair<BUDataStructures::ActualCalleesTy::const_iterator,
      BUDataStructures::ActualCalleesTy::const_iterator>
         IP = ActualCallees.equal_range(CallI);

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

  // Now fold in the necessary globals from the GlobalsGraph.  A global G
  // must be folded in if it exists in the current graph (i.e., is not dead)
  // and it was not inlined from any of my callers.  If it was inlined from
  // a caller, it would have been fully consistent with the GlobalsGraph
  // in the caller so folding in is not necessary.  Otherwise, this node came
  // solely from this function's BU graph and so has to be made consistent.
  // 
  Graph.updateFromGlobalGraph();

  // Recompute the Incomplete markers.  Depends on whether args are complete
  unsigned Flags
    = HasIncompleteArgs ? DSGraph::MarkFormalArgs : DSGraph::IgnoreFormalArgs;
  Graph.markIncompleteNodes(Flags | DSGraph::IgnoreGlobals);

  // Delete dead nodes.  Treat globals that are unreachable as dead also.
  Graph.removeDeadNodes(DSGraph::RemoveUnreachableGlobals);

  // We are done with computing the current TD Graph! Now move on to
  // inlining the current graph into the graphs for its callees, if any.
  // 
  const std::vector<DSCallSite> &FunctionCalls = Graph.getFunctionCalls();
  if (FunctionCalls.empty()) {
    DEBUG(std::cerr << "  [TD] No callees for: " << Graph.getFunctionNames()
                    << "\n");
    return;
  }

  // Now that we have information about all of the callees, propagate the
  // current graph into the callees.  Clone only the reachable subgraph at
  // each call-site, not the entire graph (even though the entire graph
  // would be cloned only once, this should still be better on average).
  //
  DEBUG(std::cerr << "  [TD] Inlining '" << Graph.getFunctionNames() <<"' into "
                  << FunctionCalls.size() << " call nodes.\n");

  const BUDataStructures::ActualCalleesTy &ActualCallees =
    getAnalysis<BUDataStructures>().getActualCallees();

  // Loop over all the call sites and all the callees at each call site.
  // Clone and merge the reachable subgraph from the call into callee's graph.
  // 
  for (unsigned i = 0, e = FunctionCalls.size(); i != e; ++i) {
    Instruction *CallI = FunctionCalls[i].getCallSite().getInstruction();
    // For each function in the invoked function list at this call site...
    std::pair<BUDataStructures::ActualCalleesTy::const_iterator,
      BUDataStructures::ActualCalleesTy::const_iterator>
          IP = ActualCallees.equal_range(CallI);

    // Multiple callees may have the same graph, so try to inline and merge
    // only once for each <callSite,calleeGraph> pair, not once for each
    // <callSite,calleeFunction> pair; the latter will be correct but slower.
    hash_set<DSGraph*> GraphsSeen;

    // Loop over each actual callee at this call site
    for (BUDataStructures::ActualCalleesTy::const_iterator I = IP.first;
         I != IP.second; ++I) {
      DSGraph& CalleeGraph = getDSGraph(*I->second);
      assert(&CalleeGraph != &Graph && "TD need not inline graph into self!");

      // if this callee graph is already done at this site, skip this callee
      if (GraphsSeen.find(&CalleeGraph) != GraphsSeen.end())
        continue;
      GraphsSeen.insert(&CalleeGraph);

      // Get the root nodes for cloning the reachable subgraph into each callee:
      // -- all global nodes that appear in both the caller and the callee
      // -- return value at this call site, if any
      // -- actual arguments passed at this call site
      // -- callee node at this call site, if this is an indirect call (this may
      //    not be needed for merging, but allows us to create CS and therefore
      //    simplify the merging below).
      hash_set<const DSNode*> RootNodeSet;
      for (DSGraph::ScalarMapTy::const_iterator
             SI = CalleeGraph.getScalarMap().begin(),
             SE = CalleeGraph.getScalarMap().end(); SI != SE; ++SI)
        if (GlobalValue* GV = dyn_cast<GlobalValue>(SI->first)) {
          DSGraph::ScalarMapTy::const_iterator GI=Graph.getScalarMap().find(GV);
          if (GI != Graph.getScalarMap().end())
            RootNodeSet.insert(GI->second.getNode());
        }

      if (const DSNode* RetNode = FunctionCalls[i].getRetVal().getNode())
        RootNodeSet.insert(RetNode);

      for (unsigned j=0, N=FunctionCalls[i].getNumPtrArgs(); j < N; ++j)
        if (const DSNode* ArgTarget = FunctionCalls[i].getPtrArg(j).getNode())
          RootNodeSet.insert(ArgTarget);

      if (FunctionCalls[i].isIndirectCall())
        RootNodeSet.insert(FunctionCalls[i].getCalleeNode());

      DEBUG(std::cerr << "     [TD] Resolving arguments for callee graph '"
            << CalleeGraph.getFunctionNames()
            << "': " << I->second->getFunctionType()->getNumParams()
            << " args\n          at call site (DSCallSite*) 0x"
            << &FunctionCalls[i] << "\n");
      
      DSGraph::NodeMapTy NodeMapInCallee; // map from nodes to clones in callee
      DSGraph::NodeMapTy CompletedMap;    // unused map for nodes not to do
      CalleeGraph.cloneReachableSubgraph(Graph, RootNodeSet,
                                         NodeMapInCallee, CompletedMap,
                                         DSGraph::StripModRefBits |
                                         DSGraph::KeepAllocaBit);

      // Transform our call site info into the cloned version for CalleeGraph
      DSCallSite CS(FunctionCalls[i], NodeMapInCallee);

      // Get the formal argument and return nodes for the called function
      // and merge them with the cloned subgraph.  Global nodes were merged  
      // already by cloneReachableSubgraph() above.
      CalleeGraph.getCallSiteForArguments(*I->second).mergeWith(CS);

      ++NumTDInlines;
    }
  }

  DEBUG(std::cerr << "  [TD] Done inlining into callees for: "
        << Graph.getFunctionNames() << " [" << Graph.getGraphSize() << "+"
        << Graph.getFunctionCalls().size() << "]\n");
}
