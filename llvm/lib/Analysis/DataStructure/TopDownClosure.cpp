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

#include "llvm/Analysis/DataStructure/DataStructure.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Analysis/DataStructure/DSGraph.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
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
      std::vector<Function*> Functions;
      NN->addFullFunctionList(Functions);
      ArgsRemainIncomplete.insert(Functions.begin(), Functions.end());
      markReachableFunctionsExternallyAccessible(NN, Visited);
    }
  }
}


// run - Calculate the top down data structure graphs for each function in the
// program.
//
bool TDDataStructures::runOnModule(Module &M) {
  BUDataStructures &BU = getAnalysis<BUDataStructures>();
  GlobalECs = BU.getGlobalECs();
  GlobalsGraph = new DSGraph(BU.getGlobalsGraph(), GlobalECs);
  GlobalsGraph->setPrintAuxCalls();

  // Figure out which functions must not mark their arguments complete because
  // they are accessible outside this compilation unit.  Currently, these
  // arguments are functions which are reachable by global variables in the
  // globals graph.
  const DSScalarMap &GGSM = GlobalsGraph->getScalarMap();
  hash_set<DSNode*> Visited;
  for (DSScalarMap::global_iterator I=GGSM.global_begin(), E=GGSM.global_end();
       I != E; ++I)
    markReachableFunctionsExternallyAccessible(GGSM.find(*I)->second.getNode(),
                                               Visited);

  // Loop over unresolved call nodes.  Any functions passed into (but not
  // returned!) from unresolvable call nodes may be invoked outside of the
  // current module.
  for (DSGraph::afc_iterator I = GlobalsGraph->afc_begin(),
         E = GlobalsGraph->afc_end(); I != E; ++I)
    for (unsigned arg = 0, e = I->getNumPtrArgs(); arg != e; ++arg)
      markReachableFunctionsExternallyAccessible(I->getPtrArg(arg).getNode(),
                                                 Visited);
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
  GlobalsGraph->removeTriviallyDeadNodes();

  return false;
}


DSGraph &TDDataStructures::getOrCreateDSGraph(Function &F) {
  DSGraph *&G = DSInfo[&F];
  if (G == 0) { // Not created yet?  Clone BU graph...
    G = new DSGraph(getAnalysis<BUDataStructures>().getDSGraph(F), GlobalECs);
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
  for (DSGraph::fc_iterator CI = G.fc_begin(), E = G.fc_end(); CI != E; ++CI) {
    Instruction *CallI = CI->getCallSite().getInstruction();
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
  Graph.maskIncompleteMarkers();

  // If any of the functions has incomplete incoming arguments, don't mark any
  // of them as complete.
  bool HasIncompleteArgs = false;
  for (DSGraph::retnodes_iterator I = Graph.retnodes_begin(),
         E = Graph.retnodes_end(); I != E; ++I)
    if (ArgsRemainIncomplete.count(I->first)) {
      HasIncompleteArgs = true;
      break;
    }

  // Recompute the Incomplete markers.  Depends on whether args are complete
  unsigned Flags
    = HasIncompleteArgs ? DSGraph::MarkFormalArgs : DSGraph::IgnoreFormalArgs;
  Graph.markIncompleteNodes(Flags | DSGraph::IgnoreGlobals);

  // Delete dead nodes.  Treat globals that are unreachable as dead also.
  Graph.removeDeadNodes(DSGraph::RemoveUnreachableGlobals);

  // We are done with computing the current TD Graph! Now move on to
  // inlining the current graph into the graphs for its callees, if any.
  // 
  if (Graph.fc_begin() == Graph.fc_end()) {
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
                  << Graph.getFunctionCalls().size() << " call nodes.\n");

  const BUDataStructures::ActualCalleesTy &ActualCallees =
    getAnalysis<BUDataStructures>().getActualCallees();

  // Loop over all the call sites and all the callees at each call site.  Build
  // a mapping from called DSGraph's to the call sites in this function that
  // invoke them.  This is useful because we can be more efficient if there are
  // multiple call sites to the callees in the graph from this caller.
  std::multimap<DSGraph*, std::pair<Function*, const DSCallSite*> > CallSites;

  for (DSGraph::fc_iterator CI = Graph.fc_begin(), E = Graph.fc_end();
       CI != E; ++CI) {
    Instruction *CallI = CI->getCallSite().getInstruction();
    // For each function in the invoked function list at this call site...
    std::pair<BUDataStructures::ActualCalleesTy::const_iterator,
              BUDataStructures::ActualCalleesTy::const_iterator> 
      IP = ActualCallees.equal_range(CallI);
    // Loop over each actual callee at this call site
    for (BUDataStructures::ActualCalleesTy::const_iterator I = IP.first;
         I != IP.second; ++I) {
      DSGraph& CalleeGraph = getDSGraph(*I->second);
      if (&CalleeGraph != &Graph)
        CallSites.insert(std::make_pair(&CalleeGraph,
                                        std::make_pair(I->second, &*CI)));
    }
  }

  // Now that we built the mapping, actually perform the inlining a callee graph
  // at a time.
  std::multimap<DSGraph*,std::pair<Function*,const DSCallSite*> >::iterator CSI;
  for (CSI = CallSites.begin(); CSI != CallSites.end(); ) {
    DSGraph &CalleeGraph = *CSI->first;
    // Iterate through all of the call sites of this graph, cloning and merging
    // any nodes required by the call.
    ReachabilityCloner RC(CalleeGraph, Graph, 0);

    // Clone over any global nodes that appear in both graphs.
    for (DSScalarMap::global_iterator
           SI = CalleeGraph.getScalarMap().global_begin(),
           SE = CalleeGraph.getScalarMap().global_end(); SI != SE; ++SI) {
      DSScalarMap::const_iterator GI = Graph.getScalarMap().find(*SI);
      if (GI != Graph.getScalarMap().end())
        RC.merge(CalleeGraph.getNodeForValue(*SI), GI->second);
    }

    // Loop over all of the distinct call sites in the caller of the callee.
    for (; CSI != CallSites.end() && CSI->first == &CalleeGraph; ++CSI) {
      Function &CF = *CSI->second.first;
      const DSCallSite &CS = *CSI->second.second;
      DEBUG(std::cerr << "     [TD] Resolving arguments for callee graph '"
            << CalleeGraph.getFunctionNames()
            << "': " << CF.getFunctionType()->getNumParams()
            << " args\n          at call site (DSCallSite*) 0x" << &CS << "\n");
      
      // Get the formal argument and return nodes for the called function and
      // merge them with the cloned subgraph.
      RC.mergeCallSite(CalleeGraph.getCallSiteForArguments(CF), CS);
      ++NumTDInlines;
    }
  }

  DEBUG(std::cerr << "  [TD] Done inlining into callees for: "
        << Graph.getFunctionNames() << " [" << Graph.getGraphSize() << "+"
        << Graph.getFunctionCalls().size() << "]\n");
}

static const Function *getFnForValue(const Value *V) {
  if (const Instruction *I = dyn_cast<Instruction>(V))
    return I->getParent()->getParent();
  else if (const Argument *A = dyn_cast<Argument>(V))
    return A->getParent();
  else if (const BasicBlock *BB = dyn_cast<BasicBlock>(V))
    return BB->getParent();
  return 0;
}

void TDDataStructures::deleteValue(Value *V) {
  if (const Function *F = getFnForValue(V)) {  // Function local value?
    // If this is a function local value, just delete it from the scalar map!
    getDSGraph(*F).getScalarMap().eraseIfExists(V);
    return;
  }

  if (Function *F = dyn_cast<Function>(V)) {
    assert(getDSGraph(*F).getReturnNodes().size() == 1 &&
           "cannot handle scc's");
    delete DSInfo[F];
    DSInfo.erase(F);
    return;
  }

  assert(!isa<GlobalVariable>(V) && "Do not know how to delete GV's yet!");
}

void TDDataStructures::copyValue(Value *From, Value *To) {
  if (From == To) return;
  if (const Function *F = getFnForValue(From)) {  // Function local value?
    // If this is a function local value, just delete it from the scalar map!
    getDSGraph(*F).getScalarMap().copyScalarIfExists(From, To);
    return;
  }

  if (Function *FromF = dyn_cast<Function>(From)) {
    Function *ToF = cast<Function>(To);
    assert(!DSInfo.count(ToF) && "New Function already exists!");
    DSGraph *NG = new DSGraph(getDSGraph(*FromF), GlobalECs);
    DSInfo[ToF] = NG;
    assert(NG->getReturnNodes().size() == 1 && "Cannot copy SCC's yet!");

    // Change the Function* is the returnnodes map to the ToF.
    DSNodeHandle Ret = NG->retnodes_begin()->second;
    NG->getReturnNodes().clear();
    NG->getReturnNodes()[ToF] = Ret;
    return;
  }

  assert(!isa<GlobalVariable>(From) && "Do not know how to copy GV's yet!");
}
