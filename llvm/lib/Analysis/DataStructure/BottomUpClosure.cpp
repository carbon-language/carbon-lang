//===- BottomUpClosure.cpp - Compute bottom-up interprocedural closure ----===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the BUDataStructures class, which represents the
// Bottom-Up Interprocedural closure of the data structure graph over the
// program.  This is useful for applications like pool allocation, but **not**
// applications like alias analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure/DataStructure.h"
#include "llvm/Module.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "DSCallSiteIterator.h"
using namespace llvm;

namespace {
  Statistic<> MaxSCC("budatastructure", "Maximum SCC Size in Call Graph");
  Statistic<> NumBUInlines("budatastructures", "Number of graphs inlined");
  Statistic<> NumCallEdges("budatastructures", "Number of 'actual' call edges");
  
  RegisterAnalysis<BUDataStructures>
  X("budatastructure", "Bottom-up Data Structure Analysis");
}

using namespace DS;

// run - Calculate the bottom up data structure graphs for each function in the
// program.
//
bool BUDataStructures::runOnModule(Module &M) {
  LocalDataStructures &LocalDSA = getAnalysis<LocalDataStructures>();
  GlobalsGraph = new DSGraph(LocalDSA.getGlobalsGraph());
  GlobalsGraph->setPrintAuxCalls();

  IndCallGraphMap = new std::map<std::vector<Function*>,
                           std::pair<DSGraph*, std::vector<DSNodeHandle> > >();

  std::vector<Function*> Stack;
  hash_map<Function*, unsigned> ValMap;
  unsigned NextID = 1;

  Function *MainFunc = M.getMainFunction();
  if (MainFunc)
    calculateGraphs(MainFunc, Stack, NextID, ValMap);

  // Calculate the graphs for any functions that are unreachable from main...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal() && !DSInfo.count(I)) {
#ifndef NDEBUG
      if (MainFunc)
        std::cerr << "*** Function unreachable from main: "
                  << I->getName() << "\n";
#endif
      calculateGraphs(I, Stack, NextID, ValMap);     // Calculate all graphs.
    }

  NumCallEdges += ActualCallees.size();

  // If we computed any temporary indcallgraphs, free them now.
  for (std::map<std::vector<Function*>,
         std::pair<DSGraph*, std::vector<DSNodeHandle> > >::iterator I =
         IndCallGraphMap->begin(), E = IndCallGraphMap->end(); I != E; ++I) {
    I->second.second.clear();  // Drop arg refs into the graph.
    delete I->second.first;
  }
  delete IndCallGraphMap;

  // At the end of the bottom-up pass, the globals graph becomes complete.
  // FIXME: This is not the right way to do this, but it is sorta better than
  // nothing!  In particular, externally visible globals and unresolvable call
  // nodes at the end of the BU phase should make things that they point to
  // incomplete in the globals graph.
  // 
  GlobalsGraph->removeTriviallyDeadNodes();
  GlobalsGraph->maskIncompleteMarkers();

  // Merge the globals variables (not the calls) from the globals graph back
  // into the main function's graph so that the main function contains all of
  // the information about global pools and GV usage in the program.
  if (MainFunc && !MainFunc->isExternal()) {
    DSGraph &MainGraph = getOrCreateGraph(MainFunc);
    const DSGraph &GG = *MainGraph.getGlobalsGraph();
    ReachabilityCloner RC(MainGraph, GG, 
                          DSGraph::DontCloneCallNodes |
                          DSGraph::DontCloneAuxCallNodes);

    // Clone the global nodes into this graph.
    for (DSScalarMap::global_iterator I = GG.getScalarMap().global_begin(),
           E = GG.getScalarMap().global_end(); I != E; ++I)
      if (isa<GlobalVariable>(*I))
        RC.getClonedNH(GG.getNodeForValue(*I));

    MainGraph.maskIncompleteMarkers();
    MainGraph.markIncompleteNodes(DSGraph::MarkFormalArgs | 
                                  DSGraph::IgnoreGlobals);
  }

  return false;
}

DSGraph &BUDataStructures::getOrCreateGraph(Function *F) {
  // Has the graph already been created?
  DSGraph *&Graph = DSInfo[F];
  if (Graph) return *Graph;

  // Copy the local version into DSInfo...
  Graph = new DSGraph(getAnalysis<LocalDataStructures>().getDSGraph(*F));

  Graph->setGlobalsGraph(GlobalsGraph);
  Graph->setPrintAuxCalls();

  // Start with a copy of the original call sites...
  Graph->getAuxFunctionCalls() = Graph->getFunctionCalls();
  return *Graph;
}

unsigned BUDataStructures::calculateGraphs(Function *F,
                                           std::vector<Function*> &Stack,
                                           unsigned &NextID, 
                                     hash_map<Function*, unsigned> &ValMap) {
  assert(!ValMap.count(F) && "Shouldn't revisit functions!");
  unsigned Min = NextID++, MyID = Min;
  ValMap[F] = Min;
  Stack.push_back(F);

  // FIXME!  This test should be generalized to be any function that we have
  // already processed, in the case when there isn't a main or there are
  // unreachable functions!
  if (F->isExternal()) {   // sprintf, fprintf, sscanf, etc...
    // No callees!
    Stack.pop_back();
    ValMap[F] = ~0;
    return Min;
  }

  DSGraph &Graph = getOrCreateGraph(F);

  // The edges out of the current node are the call site targets...
  for (DSCallSiteIterator I = DSCallSiteIterator::begin_aux(Graph),
         E = DSCallSiteIterator::end_aux(Graph); I != E; ++I) {
    Function *Callee = *I;
    unsigned M;
    // Have we visited the destination function yet?
    hash_map<Function*, unsigned>::iterator It = ValMap.find(Callee);
    if (It == ValMap.end())  // No, visit it now.
      M = calculateGraphs(Callee, Stack, NextID, ValMap);
    else                    // Yes, get it's number.
      M = It->second;
    if (M < Min) Min = M;
  }

  assert(ValMap[F] == MyID && "SCC construction assumption wrong!");
  if (Min != MyID)
    return Min;         // This is part of a larger SCC!

  // If this is a new SCC, process it now.
  if (Stack.back() == F) {           // Special case the single "SCC" case here.
    DEBUG(std::cerr << "Visiting single node SCC #: " << MyID << " fn: "
                    << F->getName() << "\n");
    Stack.pop_back();
    DSGraph &G = getDSGraph(*F);
    DEBUG(std::cerr << "  [BU] Calculating graph for: " << F->getName()<< "\n");
    calculateGraph(G);
    DEBUG(std::cerr << "  [BU] Done inlining: " << F->getName() << " ["
                    << G.getGraphSize() << "+" << G.getAuxFunctionCalls().size()
                    << "]\n");

    if (MaxSCC < 1) MaxSCC = 1;

    // Should we revisit the graph?
    if (DSCallSiteIterator::begin_aux(G) != DSCallSiteIterator::end_aux(G)) {
      ValMap.erase(F);
      return calculateGraphs(F, Stack, NextID, ValMap);
    } else {
      ValMap[F] = ~0U;
    }
    return MyID;

  } else {
    // SCCFunctions - Keep track of the functions in the current SCC
    //
    hash_set<DSGraph*> SCCGraphs;

    Function *NF;
    std::vector<Function*>::iterator FirstInSCC = Stack.end();
    DSGraph *SCCGraph = 0;
    do {
      NF = *--FirstInSCC;
      ValMap[NF] = ~0U;

      // Figure out which graph is the largest one, in order to speed things up
      // a bit in situations where functions in the SCC have widely different
      // graph sizes.
      DSGraph &NFGraph = getDSGraph(*NF);
      SCCGraphs.insert(&NFGraph);
      // FIXME: If we used a better way of cloning graphs (ie, just splice all
      // of the nodes into the new graph), this would be completely unneeded!
      if (!SCCGraph || SCCGraph->getGraphSize() < NFGraph.getGraphSize())
        SCCGraph = &NFGraph;
    } while (NF != F);

    std::cerr << "Calculating graph for SCC #: " << MyID << " of size: "
              << SCCGraphs.size() << "\n";

    // Compute the Max SCC Size...
    if (MaxSCC < SCCGraphs.size())
      MaxSCC = SCCGraphs.size();

    // First thing first, collapse all of the DSGraphs into a single graph for
    // the entire SCC.  We computed the largest graph, so clone all of the other
    // (smaller) graphs into it.  Discard all of the old graphs.
    //
    for (hash_set<DSGraph*>::iterator I = SCCGraphs.begin(),
           E = SCCGraphs.end(); I != E; ++I) {
      DSGraph &G = **I;
      if (&G != SCCGraph) {
        {
          DSGraph::NodeMapTy NodeMap;
          SCCGraph->cloneInto(G, SCCGraph->getScalarMap(),
                              SCCGraph->getReturnNodes(), NodeMap);
        }
        // Update the DSInfo map and delete the old graph...
        for (DSGraph::retnodes_iterator I = G.retnodes_begin(),
               E = G.retnodes_end(); I != E; ++I)
          DSInfo[I->first] = SCCGraph;
        delete &G;
      }
    }

    // Clean up the graph before we start inlining a bunch again...
    SCCGraph->removeDeadNodes(DSGraph::KeepUnreachableGlobals);

    // Now that we have one big happy family, resolve all of the call sites in
    // the graph...
    calculateGraph(*SCCGraph);
    DEBUG(std::cerr << "  [BU] Done inlining SCC  [" << SCCGraph->getGraphSize()
                    << "+" << SCCGraph->getAuxFunctionCalls().size() << "]\n");

    std::cerr << "DONE with SCC #: " << MyID << "\n";

    // We never have to revisit "SCC" processed functions...
    
    // Drop the stuff we don't need from the end of the stack
    Stack.erase(FirstInSCC, Stack.end());
    return MyID;
  }

  return MyID;  // == Min
}


// releaseMemory - If the pass pipeline is done with this pass, we can release
// our memory... here...
//
void BUDataStructures::releaseMemory() {
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

static bool isVAHackFn(const Function *F) {
  return F->getName() == "printf"  || F->getName() == "sscanf" ||
    F->getName() == "fprintf" || F->getName() == "open" ||
    F->getName() == "sprintf" || F->getName() == "fputs" ||
    F->getName() == "fscanf";
}

// isUnresolvableFunction - Return true if this is an unresolvable
// external function.  A direct or indirect call to this cannot be resolved.
// 
static bool isResolvableFunc(const Function* callee) {
  return !callee->isExternal() || isVAHackFn(callee);
}

void BUDataStructures::calculateGraph(DSGraph &Graph) {
  // Move our call site list into TempFCs so that inline call sites go into the
  // new call site list and doesn't invalidate our iterators!
  std::list<DSCallSite> TempFCs;
  std::list<DSCallSite> &AuxCallsList = Graph.getAuxFunctionCalls();
  TempFCs.swap(AuxCallsList);

  DSGraph::ReturnNodesTy &ReturnNodes = Graph.getReturnNodes();

  bool Printed = false;
  std::vector<Function*> CalledFuncs;
  while (!TempFCs.empty()) {
    DSCallSite &CS = *TempFCs.begin();

    CalledFuncs.clear();

    if (CS.isDirectCall()) {
      Function *F = CS.getCalleeFunc();
      if (isResolvableFunc(F))
        if (F->isExternal()) {  // Call to fprintf, etc.
          TempFCs.erase(TempFCs.begin());
          continue;
        } else {
          CalledFuncs.push_back(F);
        }
    } else {
      DSNode *Node = CS.getCalleeNode();

      if (!Node->isIncomplete())
        for (unsigned i = 0, e = Node->getGlobals().size(); i != e; ++i)
          if (Function *CF = dyn_cast<Function>(Node->getGlobals()[i]))
            if (isResolvableFunc(CF) && !CF->isExternal())
              CalledFuncs.push_back(CF);
    }

    if (CalledFuncs.empty()) {
      // Remember that we could not resolve this yet!
      AuxCallsList.splice(AuxCallsList.end(), TempFCs, TempFCs.begin());
      continue;
    } else {
      DSGraph *GI;

      if (CalledFuncs.size() == 1) {
        Function *Callee = CalledFuncs[0];
        ActualCallees.insert(std::make_pair(CS.getCallSite().getInstruction(),
                                            Callee));

        // Get the data structure graph for the called function.
        GI = &getDSGraph(*Callee);  // Graph to inline
        DEBUG(std::cerr << "    Inlining graph for " << Callee->getName());

        DEBUG(std::cerr << "[" << GI->getGraphSize() << "+"
              << GI->getAuxFunctionCalls().size() << "] into '"
              << Graph.getFunctionNames() << "' [" << Graph.getGraphSize() <<"+"
              << Graph.getAuxFunctionCalls().size() << "]\n");
        Graph.mergeInGraph(CS, *Callee, *GI,
                           DSGraph::KeepModRefBits | 
                           DSGraph::StripAllocaBit|DSGraph::DontCloneCallNodes);
        ++NumBUInlines;
      } else {
        if (!Printed)
          std::cerr << "In Fns: " << Graph.getFunctionNames() << "\n";
        std::cerr << "  calls " << CalledFuncs.size()
                  << " fns from site: " << CS.getCallSite().getInstruction() 
                  << "  " << *CS.getCallSite().getInstruction();
        unsigned NumToPrint = CalledFuncs.size();
        if (NumToPrint > 8) NumToPrint = 8;
        std::cerr << "   Fns =";
        for (std::vector<Function*>::iterator I = CalledFuncs.begin(),
               E = CalledFuncs.end(); I != E && NumToPrint; ++I, --NumToPrint)
          std::cerr << " " << (*I)->getName();
        std::cerr << "\n";

        // See if we already computed a graph for this set of callees.
        std::sort(CalledFuncs.begin(), CalledFuncs.end());
        std::pair<DSGraph*, std::vector<DSNodeHandle> > &IndCallGraph =
          (*IndCallGraphMap)[CalledFuncs];

        if (IndCallGraph.first == 0) {
          std::vector<Function*>::iterator I = CalledFuncs.begin(),
            E = CalledFuncs.end();
          
          // Start with a copy of the first graph.
          GI = IndCallGraph.first = new DSGraph(getDSGraph(**I));
          GI->setGlobalsGraph(Graph.getGlobalsGraph());
          std::vector<DSNodeHandle> &Args = IndCallGraph.second;

          // Get the argument nodes for the first callee.  The return value is
          // the 0th index in the vector.
          GI->getFunctionArgumentsForCall(*I, Args);

          // Merge all of the other callees into this graph.
          for (++I; I != E; ++I) {
            // If the graph already contains the nodes for the function, don't
            // bother merging it in again.
            if (!GI->containsFunction(*I)) {
              DSGraph::NodeMapTy NodeMap;
              GI->cloneInto(getDSGraph(**I), GI->getScalarMap(),
                            GI->getReturnNodes(), NodeMap);
              ++NumBUInlines;
            }

            std::vector<DSNodeHandle> NextArgs;
            GI->getFunctionArgumentsForCall(*I, NextArgs);
            unsigned i = 0, e = Args.size();
            for (; i != e; ++i) {
              if (i == NextArgs.size()) break;
              Args[i].mergeWith(NextArgs[i]);
            }
            for (e = NextArgs.size(); i != e; ++i)
              Args.push_back(NextArgs[i]);
          }
          
          // Clean up the final graph!
          GI->removeDeadNodes(DSGraph::KeepUnreachableGlobals);
        } else {
          std::cerr << "***\n*** RECYCLED GRAPH ***\n***\n";
        }

        GI = IndCallGraph.first;

        // Merge the unified graph into this graph now.
        DEBUG(std::cerr << "    Inlining multi callee graph "
              << "[" << GI->getGraphSize() << "+"
              << GI->getAuxFunctionCalls().size() << "] into '"
              << Graph.getFunctionNames() << "' [" << Graph.getGraphSize() <<"+"
              << Graph.getAuxFunctionCalls().size() << "]\n");

        Graph.mergeInGraph(CS, IndCallGraph.second, *GI,
                           DSGraph::KeepModRefBits | 
                           DSGraph::StripAllocaBit |
                           DSGraph::DontCloneCallNodes);
        ++NumBUInlines;
      }
    }
    TempFCs.erase(TempFCs.begin());
  }

  // Recompute the Incomplete markers
  assert(Graph.getInlinedGlobals().empty());
  Graph.maskIncompleteMarkers();
  Graph.markIncompleteNodes(DSGraph::MarkFormalArgs);

  // Delete dead nodes.  Treat globals that are unreachable but that can
  // reach live nodes as live.
  Graph.removeDeadNodes(DSGraph::KeepUnreachableGlobals);

  // When this graph is finalized, clone the globals in the graph into the
  // globals graph to make sure it has everything, from all graphs.
  DSScalarMap &MainSM = Graph.getScalarMap();
  ReachabilityCloner RC(*GlobalsGraph, Graph, DSGraph::StripAllocaBit);

  // Clone everything reachable from globals in the function graph into the
  // globals graph.
  for (DSScalarMap::global_iterator I = MainSM.global_begin(),
         E = MainSM.global_end(); I != E; ++I) 
    RC.getClonedNH(MainSM[*I]);

  //Graph.writeGraphToFile(std::cerr, "bu_" + F.getName());
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

/// deleteValue/copyValue - Interfaces to update the DSGraphs in the program.
/// These correspond to the interfaces defined in the AliasAnalysis class.
void BUDataStructures::deleteValue(Value *V) {
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

void BUDataStructures::copyValue(Value *From, Value *To) {
  if (From == To) return;
  if (const Function *F = getFnForValue(From)) {  // Function local value?
    // If this is a function local value, just delete it from the scalar map!
    getDSGraph(*F).getScalarMap().copyScalarIfExists(From, To);
    return;
  }

  if (Function *FromF = dyn_cast<Function>(From)) {
    Function *ToF = cast<Function>(To);
    assert(!DSInfo.count(ToF) && "New Function already exists!");
    DSGraph *NG = new DSGraph(getDSGraph(*FromF));
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
