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
#include "llvm/Analysis/DataStructure/DSGraph.h"
#include "llvm/Module.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

namespace {
  Statistic<> MaxSCC("budatastructure", "Maximum SCC Size in Call Graph");
  Statistic<> NumBUInlines("budatastructures", "Number of graphs inlined");
  Statistic<> NumCallEdges("budatastructures", "Number of 'actual' call edges");
  
  RegisterAnalysis<BUDataStructures>
  X("budatastructure", "Bottom-up Data Structure Analysis");
}

// run - Calculate the bottom up data structure graphs for each function in the
// program.
//
bool BUDataStructures::runOnModule(Module &M) {
  LocalDataStructures &LocalDSA = getAnalysis<LocalDataStructures>();
  GlobalECs = LocalDSA.getGlobalECs();

  GlobalsGraph = new DSGraph(LocalDSA.getGlobalsGraph(), GlobalECs);
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

  // Mark external globals incomplete.
  GlobalsGraph->markIncompleteNodes(DSGraph::IgnoreGlobals);

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
  Graph = new DSGraph(getAnalysis<LocalDataStructures>().getDSGraph(*F),
                      GlobalECs);

  Graph->setGlobalsGraph(GlobalsGraph);
  Graph->setPrintAuxCalls();

  // Start with a copy of the original call sites...
  Graph->getAuxFunctionCalls() = Graph->getFunctionCalls();
  return *Graph;
}

static bool isVAHackFn(const Function *F) {
  return F->getName() == "printf"  || F->getName() == "sscanf" ||
    F->getName() == "fprintf" || F->getName() == "open" ||
    F->getName() == "sprintf" || F->getName() == "fputs" ||
    F->getName() == "fscanf";
}

static bool isResolvableFunc(const Function* callee) {
  return !callee->isExternal() || isVAHackFn(callee);
}

static void GetAllCallees(const DSCallSite &CS, 
                          std::vector<Function*> &Callees) {
  if (CS.isDirectCall()) {
    if (isResolvableFunc(CS.getCalleeFunc()))
      Callees.push_back(CS.getCalleeFunc());
  } else if (!CS.getCalleeNode()->isIncomplete()) {
    // Get all callees.
    unsigned OldSize = Callees.size();
    CS.getCalleeNode()->addFullFunctionList(Callees);
    
    // If any of the callees are unresolvable, remove the whole batch!
    for (unsigned i = OldSize, e = Callees.size(); i != e; ++i)
      if (!isResolvableFunc(Callees[i])) {
        Callees.erase(Callees.begin()+OldSize, Callees.end());
        return;
      }
  }
}


/// GetAllAuxCallees - Return a list containing all of the resolvable callees in
/// the aux list for the specified graph in the Callees vector.
static void GetAllAuxCallees(DSGraph &G, std::vector<Function*> &Callees) {
  Callees.clear();
  for (DSGraph::afc_iterator I = G.afc_begin(), E = G.afc_end(); I != E; ++I)
    GetAllCallees(*I, Callees);
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

  // Find all callee functions.
  std::vector<Function*> CalleeFunctions;
  GetAllAuxCallees(Graph, CalleeFunctions);

  // The edges out of the current node are the call site targets...
  for (unsigned i = 0, e = CalleeFunctions.size(); i != e; ++i) {
    Function *Callee = CalleeFunctions[i];
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

    // Should we revisit the graph?  Only do it if there are now new resolvable
    // callees.
    GetAllAuxCallees(Graph, CalleeFunctions);
    if (!CalleeFunctions.empty()) {
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
        SCCGraph->cloneInto(G);

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
void BUDataStructures::releaseMyMemory() {
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

    // Fast path for noop calls.  Note that we don't care about merging globals
    // in the callee with nodes in the caller here.
    if (CS.getRetVal().isNull() && CS.getNumPtrArgs() == 0) {
      TempFCs.erase(TempFCs.begin());
      continue;
    } else if (CS.isDirectCall() && isVAHackFn(CS.getCalleeFunc())) {
      TempFCs.erase(TempFCs.begin());
      continue;
    }

    GetAllCallees(CS, CalledFuncs);

    if (CalledFuncs.empty()) {
      // Remember that we could not resolve this yet!
      AuxCallsList.splice(AuxCallsList.end(), TempFCs, TempFCs.begin());
      continue;
    } else {
      DSGraph *GI;
      Instruction *TheCall = CS.getCallSite().getInstruction();

      if (CalledFuncs.size() == 1) {
        Function *Callee = CalledFuncs[0];
        ActualCallees.insert(std::make_pair(TheCall, Callee));

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
        std::cerr << "   Fns =";
        unsigned NumPrinted = 0;

        for (std::vector<Function*>::iterator I = CalledFuncs.begin(),
               E = CalledFuncs.end(); I != E; ++I) {
          if (NumPrinted++ < 8) std::cerr << " " << (*I)->getName();

          // Add the call edges to the call graph.
          ActualCallees.insert(std::make_pair(TheCall, *I));
        }
        std::cerr << "\n";

        // See if we already computed a graph for this set of callees.
        std::sort(CalledFuncs.begin(), CalledFuncs.end());
        std::pair<DSGraph*, std::vector<DSNodeHandle> > &IndCallGraph =
          (*IndCallGraphMap)[CalledFuncs];

        if (IndCallGraph.first == 0) {
          std::vector<Function*>::iterator I = CalledFuncs.begin(),
            E = CalledFuncs.end();
          
          // Start with a copy of the first graph.
          GI = IndCallGraph.first = new DSGraph(getDSGraph(**I), GlobalECs);
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
              GI->cloneInto(getDSGraph(**I));
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
