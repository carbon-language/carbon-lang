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
#include "llvm/Support/Timer.h"
#include "llvm/ADT/Statistic.h"
#include <iostream>
using namespace llvm;

#if 0
#define TIME_REGION(VARNAME, DESC) \
   NamedRegionTimer VARNAME(DESC)
#else
#define TIME_REGION(VARNAME, DESC)
#endif

namespace {
  RegisterPass<TDDataStructures>   // Register the pass
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
  BUInfo = &getAnalysis<BUDataStructures>();
  GlobalECs = BUInfo->getGlobalECs();
  GlobalsGraph = new DSGraph(BUInfo->getGlobalsGraph(), GlobalECs);
  GlobalsGraph->setPrintAuxCalls();

  // Figure out which functions must not mark their arguments complete because
  // they are accessible outside this compilation unit.  Currently, these
  // arguments are functions which are reachable by global variables in the
  // globals graph.
  const DSScalarMap &GGSM = GlobalsGraph->getScalarMap();
  hash_set<DSNode*> Visited;
  for (DSScalarMap::global_iterator I=GGSM.global_begin(), E=GGSM.global_end();
       I != E; ++I) {
    DSNode *N = GGSM.find(*I)->second.getNode();
    if (N->isIncomplete())
      markReachableFunctionsExternallyAccessible(N, Visited);
  }

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

#if 0
{TIME_REGION(XXX, "td:Copy graphs");

  // Visit each of the graphs in reverse post-order now!
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal())
      getOrCreateDSGraph(*I);
  return false;
}
#endif


{TIME_REGION(XXX, "td:Compute postorder");

  // Calculate top-down from main...
  if (Function *F = M.getMainFunction())
    ComputePostOrder(*F, VisitedGraph, PostOrder);

  // Next calculate the graphs for each unreachable function...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    ComputePostOrder(*I, VisitedGraph, PostOrder);

  VisitedGraph.clear();   // Release memory!
}

{TIME_REGION(XXX, "td:Inline stuff");

  // Visit each of the graphs in reverse post-order now!
  while (!PostOrder.empty()) {
    InlineCallersIntoGraph(*PostOrder.back());
    PostOrder.pop_back();
  }
}

  // Free the IndCallMap.
  while (!IndCallMap.empty()) {
    delete IndCallMap.begin()->second;
    IndCallMap.erase(IndCallMap.begin());
  }


  ArgsRemainIncomplete.clear();
  GlobalsGraph->removeTriviallyDeadNodes();

  return false;
}


DSGraph &TDDataStructures::getOrCreateDSGraph(Function &F) {
  DSGraph *&G = DSInfo[&F];
  if (G == 0) { // Not created yet?  Clone BU graph...
    G = new DSGraph(getAnalysis<BUDataStructures>().getDSGraph(F), GlobalECs,
                    DSGraph::DontCloneAuxCallNodes);
    assert(G->getAuxFunctionCalls().empty() && "Cloned aux calls?");
    G->setPrintAuxCalls();
    G->setGlobalsGraph(GlobalsGraph);

    // Note that this graph is the graph for ALL of the function in the SCC, not
    // just F.
    for (DSGraph::retnodes_iterator RI = G->retnodes_begin(),
           E = G->retnodes_end(); RI != E; ++RI)
      if (RI->first != &F)
        DSInfo[RI->first] = G;
  }
  return *G;
}


void TDDataStructures::ComputePostOrder(Function &F,hash_set<DSGraph*> &Visited,
                                        std::vector<DSGraph*> &PostOrder) {
  if (F.isExternal()) return;
  DSGraph &G = getOrCreateDSGraph(F);
  if (Visited.count(&G)) return;
  Visited.insert(&G);

  // Recursively traverse all of the callee graphs.
  for (DSGraph::fc_iterator CI = G.fc_begin(), CE = G.fc_end(); CI != CE; ++CI){
    Instruction *CallI = CI->getCallSite().getInstruction();
    for (BUDataStructures::callee_iterator I = BUInfo->callee_begin(CallI),
           E = BUInfo->callee_end(CallI); I != E; ++I)
      ComputePostOrder(*I->second, Visited, PostOrder);
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

/// InlineCallersIntoGraph - Inline all of the callers of the specified DS graph
/// into it, then recompute completeness of nodes in the resultant graph.
void TDDataStructures::InlineCallersIntoGraph(DSGraph &DSG) {
  // Inline caller graphs into this graph.  First step, get the list of call
  // sites that call into this graph.
  std::vector<CallerCallEdge> EdgesFromCaller;
  std::map<DSGraph*, std::vector<CallerCallEdge> >::iterator
    CEI = CallerEdges.find(&DSG);
  if (CEI != CallerEdges.end()) {
    std::swap(CEI->second, EdgesFromCaller);
    CallerEdges.erase(CEI);
  }

  // Sort the caller sites to provide a by-caller-graph ordering.
  std::sort(EdgesFromCaller.begin(), EdgesFromCaller.end());


  // Merge information from the globals graph into this graph.  FIXME: This is
  // stupid.  Instead of us cloning information from the GG into this graph,
  // then having RemoveDeadNodes clone it back, we should do all of this as a
  // post-pass over all of the graphs.  We need to take cloning out of
  // removeDeadNodes and gut removeDeadNodes at the same time first though. :(
  {
    DSGraph &GG = *DSG.getGlobalsGraph();
    ReachabilityCloner RC(DSG, GG,
                          DSGraph::DontCloneCallNodes |
                          DSGraph::DontCloneAuxCallNodes);
    for (DSScalarMap::global_iterator
           GI = DSG.getScalarMap().global_begin(),
           E = DSG.getScalarMap().global_end(); GI != E; ++GI)
      RC.getClonedNH(GG.getNodeForValue(*GI));
  }

  DEBUG(std::cerr << "[TD] Inlining callers into '" << DSG.getFunctionNames()
        << "'\n");

  // Iteratively inline caller graphs into this graph.
  while (!EdgesFromCaller.empty()) {
    DSGraph &CallerGraph = *EdgesFromCaller.back().CallerGraph;

    // Iterate through all of the call sites of this graph, cloning and merging
    // any nodes required by the call.
    ReachabilityCloner RC(DSG, CallerGraph,
                          DSGraph::DontCloneCallNodes |
                          DSGraph::DontCloneAuxCallNodes);

    // Inline all call sites from this caller graph.
    do {
      const DSCallSite &CS = *EdgesFromCaller.back().CS;
      Function &CF = *EdgesFromCaller.back().CalledFunction;
      DEBUG(std::cerr << "   [TD] Inlining graph into Fn '"
            << CF.getName() << "' from ");
      if (CallerGraph.getReturnNodes().empty())
        DEBUG(std::cerr << "SYNTHESIZED INDIRECT GRAPH");
      else
        DEBUG (std::cerr << "Fn '"
               << CS.getCallSite().getInstruction()->
               getParent()->getParent()->getName() << "'");
      DEBUG(std::cerr << ": " << CF.getFunctionType()->getNumParams()
            << " args\n");

      // Get the formal argument and return nodes for the called function and
      // merge them with the cloned subgraph.
      DSCallSite T1 = DSG.getCallSiteForArguments(CF);
      RC.mergeCallSite(T1, CS);
      ++NumTDInlines;

      EdgesFromCaller.pop_back();
    } while (!EdgesFromCaller.empty() &&
             EdgesFromCaller.back().CallerGraph == &CallerGraph);
  }


  // Next, now that this graph is finalized, we need to recompute the
  // incompleteness markers for this graph and remove unreachable nodes.
  DSG.maskIncompleteMarkers();

  // If any of the functions has incomplete incoming arguments, don't mark any
  // of them as complete.
  bool HasIncompleteArgs = false;
  for (DSGraph::retnodes_iterator I = DSG.retnodes_begin(),
         E = DSG.retnodes_end(); I != E; ++I)
    if (ArgsRemainIncomplete.count(I->first)) {
      HasIncompleteArgs = true;
      break;
    }

  // Recompute the Incomplete markers.  Depends on whether args are complete
  unsigned Flags
    = HasIncompleteArgs ? DSGraph::MarkFormalArgs : DSGraph::IgnoreFormalArgs;
  DSG.markIncompleteNodes(Flags | DSGraph::IgnoreGlobals);

  // Delete dead nodes.  Treat globals that are unreachable as dead also.
  DSG.removeDeadNodes(DSGraph::RemoveUnreachableGlobals);

  // We are done with computing the current TD Graph!  Finally, before we can
  // finish processing this function, we figure out which functions it calls and
  // records these call graph edges, so that we have them when we process the
  // callee graphs.
  if (DSG.fc_begin() == DSG.fc_end()) return;

  // Loop over all the call sites and all the callees at each call site, and add
  // edges to the CallerEdges structure for each callee.
  for (DSGraph::fc_iterator CI = DSG.fc_begin(), E = DSG.fc_end();
       CI != E; ++CI) {

    // Handle direct calls efficiently.
    if (CI->isDirectCall()) {
      if (!CI->getCalleeFunc()->isExternal() &&
          !DSG.getReturnNodes().count(CI->getCalleeFunc()))
        CallerEdges[&getDSGraph(*CI->getCalleeFunc())]
          .push_back(CallerCallEdge(&DSG, &*CI, CI->getCalleeFunc()));
      continue;
    }

    Instruction *CallI = CI->getCallSite().getInstruction();
    // For each function in the invoked function list at this call site...
    BUDataStructures::callee_iterator IPI =
      BUInfo->callee_begin(CallI), IPE = BUInfo->callee_end(CallI);

    // Skip over all calls to this graph (SCC calls).
    while (IPI != IPE && &getDSGraph(*IPI->second) == &DSG)
      ++IPI;

    // All SCC calls?
    if (IPI == IPE) continue;

    Function *FirstCallee = IPI->second;
    ++IPI;

    // Skip over more SCC calls.
    while (IPI != IPE && &getDSGraph(*IPI->second) == &DSG)
      ++IPI;

    // If there is exactly one callee from this call site, remember the edge in
    // CallerEdges.
    if (IPI == IPE) {
      if (!FirstCallee->isExternal())
        CallerEdges[&getDSGraph(*FirstCallee)]
          .push_back(CallerCallEdge(&DSG, &*CI, FirstCallee));
      continue;
    }

    // Otherwise, there are multiple callees from this call site, so it must be
    // an indirect call.  Chances are that there will be other call sites with
    // this set of targets.  If so, we don't want to do M*N inlining operations,
    // so we build up a new, private, graph that represents the calls of all
    // calls to this set of functions.
    std::vector<Function*> Callees;
    for (BUDataStructures::ActualCalleesTy::const_iterator I =
           BUInfo->callee_begin(CallI), E = BUInfo->callee_end(CallI);
         I != E; ++I)
      if (!I->second->isExternal())
        Callees.push_back(I->second);
    std::sort(Callees.begin(), Callees.end());

    std::map<std::vector<Function*>, DSGraph*>::iterator IndCallRecI =
      IndCallMap.lower_bound(Callees);

    DSGraph *IndCallGraph;

    // If we already have this graph, recycle it.
    if (IndCallRecI != IndCallMap.end() && IndCallRecI->first == Callees) {
      std::cerr << "  [TD] *** Reuse of indcall graph for " << Callees.size()
                << " callees!\n";
      IndCallGraph = IndCallRecI->second;
    } else {
      // Otherwise, create a new DSGraph to represent this.
      IndCallGraph = new DSGraph(DSG.getGlobalECs(), DSG.getTargetData());

      // Make a nullary dummy call site, which will eventually get some content
      // merged into it.  The actual callee function doesn't matter here, so we
      // just pass it something to keep the ctor happy.
      std::vector<DSNodeHandle> ArgDummyVec;
      DSCallSite DummyCS(CI->getCallSite(), DSNodeHandle(), Callees[0]/*dummy*/,
                         ArgDummyVec);
      IndCallGraph->getFunctionCalls().push_back(DummyCS);

      IndCallRecI = IndCallMap.insert(IndCallRecI,
                                      std::make_pair(Callees, IndCallGraph));

      // Additionally, make sure that each of the callees inlines this graph
      // exactly once.
      DSCallSite *NCS = &IndCallGraph->getFunctionCalls().front();
      for (unsigned i = 0, e = Callees.size(); i != e; ++i) {
        DSGraph& CalleeGraph = getDSGraph(*Callees[i]);
        if (&CalleeGraph != &DSG)
          CallerEdges[&CalleeGraph].push_back(CallerCallEdge(IndCallGraph, NCS,
                                                             Callees[i]));
      }
    }

    // Now that we know which graph to use for this, merge the caller
    // information into the graph, based on information from the call site.
    ReachabilityCloner RC(*IndCallGraph, DSG, 0);
    RC.mergeCallSite(IndCallGraph->getFunctionCalls().front(), *CI);
  }
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

  if (const Function *F = getFnForValue(To)) {
    DSGraph &G = getDSGraph(*F);
    G.getScalarMap().copyScalarIfExists(From, To);
    return;
  }

  std::cerr << *From;
  std::cerr << *To;
  assert(0 && "Do not know how to copy this yet!");
  abort();
}
