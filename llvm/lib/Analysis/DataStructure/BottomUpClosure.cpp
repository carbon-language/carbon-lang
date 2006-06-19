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
#define DEBUG_TYPE "bu_dsa"
#include "llvm/Analysis/DataStructure/DataStructure.h"
#include "llvm/Analysis/DataStructure/DSGraph.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Timer.h"
#include <iostream>
using namespace llvm;

namespace {
  Statistic<> MaxSCC("budatastructure", "Maximum SCC Size in Call Graph");
  Statistic<> NumBUInlines("budatastructures", "Number of graphs inlined");
  Statistic<> NumCallEdges("budatastructures", "Number of 'actual' call edges");

  cl::opt<bool>
  AddGlobals("budatastructures-annotate-calls",
	     cl::desc("Annotate call sites with functions as they are resolved"));
  cl::opt<bool>
  UpdateGlobals("budatastructures-update-from-globals",
		cl::desc("Update local graph from global graph when processing function"));

  RegisterAnalysis<BUDataStructures>
  X("budatastructure", "Bottom-up Data Structure Analysis");
}

static bool GetAllCallees(const DSCallSite &CS,
                          std::vector<Function*> &Callees);

/// BuildGlobalECs - Look at all of the nodes in the globals graph.  If any node
/// contains multiple globals, DSA will never, ever, be able to tell the globals
/// apart.  Instead of maintaining this information in all of the graphs
/// throughout the entire program, store only a single global (the "leader") in
/// the graphs, and build equivalence classes for the rest of the globals.
static void BuildGlobalECs(DSGraph &GG, std::set<GlobalValue*> &ECGlobals) {
  DSScalarMap &SM = GG.getScalarMap();
  EquivalenceClasses<GlobalValue*> &GlobalECs = SM.getGlobalECs();
  for (DSGraph::node_iterator I = GG.node_begin(), E = GG.node_end();
       I != E; ++I) {
    if (I->getGlobalsList().size() <= 1) continue;

    // First, build up the equivalence set for this block of globals.
    const std::vector<GlobalValue*> &GVs = I->getGlobalsList();
    GlobalValue *First = GVs[0];
    for (unsigned i = 1, e = GVs.size(); i != e; ++i)
      GlobalECs.unionSets(First, GVs[i]);

    // Next, get the leader element.
    assert(First == GlobalECs.getLeaderValue(First) &&
           "First did not end up being the leader?");

    // Next, remove all globals from the scalar map that are not the leader.
    assert(GVs[0] == First && "First had to be at the front!");
    for (unsigned i = 1, e = GVs.size(); i != e; ++i) {
      ECGlobals.insert(GVs[i]);
      SM.erase(SM.find(GVs[i]));
    }

    // Finally, change the global node to only contain the leader.
    I->clearGlobals();
    I->addGlobal(First);
  }

  DEBUG(GG.AssertGraphOK());
}

/// EliminateUsesOfECGlobals - Once we have determined that some globals are in
/// really just equivalent to some other globals, remove the globals from the
/// specified DSGraph (if present), and merge any nodes with their leader nodes.
static void EliminateUsesOfECGlobals(DSGraph &G,
                                     const std::set<GlobalValue*> &ECGlobals) {
  DSScalarMap &SM = G.getScalarMap();
  EquivalenceClasses<GlobalValue*> &GlobalECs = SM.getGlobalECs();

  bool MadeChange = false;
  for (DSScalarMap::global_iterator GI = SM.global_begin(), E = SM.global_end();
       GI != E; ) {
    GlobalValue *GV = *GI++;
    if (!ECGlobals.count(GV)) continue;

    const DSNodeHandle &GVNH = SM[GV];
    assert(!GVNH.isNull() && "Global has null NH!?");

    // Okay, this global is in some equivalence class.  Start by finding the
    // leader of the class.
    GlobalValue *Leader = GlobalECs.getLeaderValue(GV);

    // If the leader isn't already in the graph, insert it into the node
    // corresponding to GV.
    if (!SM.global_count(Leader)) {
      GVNH.getNode()->addGlobal(Leader);
      SM[Leader] = GVNH;
    } else {
      // Otherwise, the leader is in the graph, make sure the nodes are the
      // merged in the specified graph.
      const DSNodeHandle &LNH = SM[Leader];
      if (LNH.getNode() != GVNH.getNode())
        LNH.mergeWith(GVNH);
    }

    // Next step, remove the global from the DSNode.
    GVNH.getNode()->removeGlobal(GV);

    // Finally, remove the global from the ScalarMap.
    SM.erase(GV);
    MadeChange = true;
  }

  DEBUG(if(MadeChange) G.AssertGraphOK());
}

static void AddGlobalToNode(BUDataStructures* B, DSCallSite D, Function* F) {
  if(!AddGlobals)
    return;
  if(D.isIndirectCall()) {
    DSGraph* GI = &B->getDSGraph(D.getCaller());
    DSNodeHandle& NHF = GI->getNodeForValue(F);
    DSCallSite DL = GI->getDSCallSiteForCallSite(D.getCallSite());
    if (DL.getCalleeNode() != NHF.getNode() || NHF.isNull()) {
      if (NHF.isNull()) {
        DSNode *N = new DSNode(F->getType()->getElementType(), GI);   // Create the node
        N->addGlobal(F);
        NHF.setTo(N,0);
        DEBUG(std::cerr << "Adding " << F->getName() << " to a call node in "
             << D.getCaller().getName() << "\n");
      }
      DL.getCalleeNode()->mergeWith(NHF, 0);
    }
  }
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
        std::cerr << "*** BU: Function unreachable from main: "
                  << I->getName() << "\n";
#endif
      calculateGraphs(I, Stack, NextID, ValMap);     // Calculate all graphs.
    }

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

  // Grow the equivalence classes for the globals to include anything that we
  // now know to be aliased.
  std::set<GlobalValue*> ECGlobals;
  BuildGlobalECs(*GlobalsGraph, ECGlobals);
  if (!ECGlobals.empty()) {
    NamedRegionTimer X("Bottom-UP EC Cleanup");
    std::cerr << "Eliminating " << ECGlobals.size() << " EC Globals!\n";
    for (hash_map<Function*, DSGraph*>::iterator I = DSInfo.begin(),
           E = DSInfo.end(); I != E; ++I)
      EliminateUsesOfECGlobals(*I->second, ECGlobals);
  }

  // Merge the globals variables (not the calls) from the globals graph back
  // into the main function's graph so that the main function contains all of
  // the information about global pools and GV usage in the program.
  if (MainFunc && !MainFunc->isExternal()) {
    DSGraph &MainGraph = getOrCreateGraph(MainFunc);
    const DSGraph &GG = *MainGraph.getGlobalsGraph();
    ReachabilityCloner RC(MainGraph, GG, DSGraph::DontCloneCallNodes |
			  DSGraph::DontCloneAuxCallNodes);

    // Clone the global nodes into this graph.
    for (DSScalarMap::global_iterator I = GG.getScalarMap().global_begin(),
           E = GG.getScalarMap().global_end(); I != E; ++I)
      if (isa<GlobalVariable>(*I))
        RC.getClonedNH(GG.getNodeForValue(*I));

    MainGraph.maskIncompleteMarkers();
    MainGraph.markIncompleteNodes(DSGraph::MarkFormalArgs |
                                  DSGraph::IgnoreGlobals);

    //Debug messages if along the way we didn't resolve a call site
    //also update the call graph and callsites we did find.
    for(DSGraph::afc_iterator ii = MainGraph.afc_begin(),
	  ee = MainGraph.afc_end(); ii != ee; ++ii) {
      std::vector<Function*> Funcs;
      GetAllCallees(*ii, Funcs);
      std::cerr << "Lost site\n";
      for (std::vector<Function*>::iterator iif = Funcs.begin(), eef = Funcs.end();
	   iif != eef; ++iif) {
	AddGlobalToNode(this, *ii, *iif);
	std::cerr << "Adding\n";
	ActualCallees.insert(std::make_pair(ii->getCallSite().getInstruction(), *iif));
      }
    }

  }

  NumCallEdges += ActualCallees.size();

  return false;
}

DSGraph &BUDataStructures::getOrCreateGraph(Function *F) {
  // Has the graph already been created?
  DSGraph *&Graph = DSInfo[F];
  if (Graph) return *Graph;

  DSGraph &LocGraph = getAnalysis<LocalDataStructures>().getDSGraph(*F);

  // Steal the local graph.
  Graph = new DSGraph(GlobalECs, LocGraph.getTargetData());
  Graph->spliceFrom(LocGraph);

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
    F->getName() == "fscanf" || F->getName() == "malloc" ||
    F->getName() == "free";
}

static bool isResolvableFunc(const Function* callee) {
  return !callee->isExternal() || isVAHackFn(callee);
}

//returns true if all callees were resolved
static bool GetAllCallees(const DSCallSite &CS,
                          std::vector<Function*> &Callees) {
  if (CS.isDirectCall()) {
    if (isResolvableFunc(CS.getCalleeFunc())) {
      Callees.push_back(CS.getCalleeFunc());
      return true;
    } else
      return false;
  } else {
    // Get all callees.
    bool retval = CS.getCalleeNode()->isComplete();
    unsigned OldSize = Callees.size();
    CS.getCalleeNode()->addFullFunctionList(Callees);

    // If any of the callees are unresolvable, remove that one
    for (unsigned i = OldSize; i != Callees.size(); ++i)
      if (!isResolvableFunc(Callees[i])) {
        Callees.erase(Callees.begin()+i);
        --i;
       retval = false;
      }
    return retval;
    //return false;
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
  if (UpdateGlobals)
    Graph.updateFromGlobalGraph();

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
    bool redo = calculateGraph(G);
    DEBUG(std::cerr << "  [BU] Done inlining: " << F->getName() << " ["
                    << G.getGraphSize() << "+" << G.getAuxFunctionCalls().size()
                    << "]\n");

    if (MaxSCC < 1) MaxSCC = 1;

    // Should we revisit the graph?  Only do it if there are now new resolvable
    // callees.
    if (redo) {
      DEBUG(std::cerr << "Recalculating " << F->getName() << " due to new knowledge\n");
      ValMap.erase(F);
      return calculateGraphs(F, Stack, NextID, ValMap);
    } else {
      ValMap[F] = ~0U;
    }
    return MyID;

  } else {
    // SCCFunctions - Keep track of the functions in the current SCC
    //
    std::vector<DSGraph*> SCCGraphs;

    unsigned SCCSize = 1;
    Function *NF = Stack.back();
    ValMap[NF] = ~0U;
    DSGraph &SCCGraph = getDSGraph(*NF);

    // First thing first, collapse all of the DSGraphs into a single graph for
    // the entire SCC.  Splice all of the graphs into one and discard all of the
    // old graphs.
    //
    while (NF != F) {
      Stack.pop_back();
      NF = Stack.back();
      ValMap[NF] = ~0U;

      DSGraph &NFG = getDSGraph(*NF);

      // Update the Function -> DSG map.
      for (DSGraph::retnodes_iterator I = NFG.retnodes_begin(),
             E = NFG.retnodes_end(); I != E; ++I)
        DSInfo[I->first] = &SCCGraph;

      SCCGraph.spliceFrom(NFG);
      delete &NFG;

      ++SCCSize;
    }
    Stack.pop_back();

    std::cerr << "Calculating graph for SCC #: " << MyID << " of size: "
              << SCCSize << "\n";

    // Compute the Max SCC Size.
    if (MaxSCC < SCCSize)
      MaxSCC = SCCSize;

    // Clean up the graph before we start inlining a bunch again...
    SCCGraph.removeDeadNodes(DSGraph::KeepUnreachableGlobals);

    // Now that we have one big happy family, resolve all of the call sites in
    // the graph...
    bool redo = calculateGraph(SCCGraph);
    DEBUG(std::cerr << "  [BU] Done inlining SCC  [" << SCCGraph.getGraphSize()
                    << "+" << SCCGraph.getAuxFunctionCalls().size() << "]\n");

    if (redo) {
      DEBUG(std::cerr << "MISSING REDO\n");
    }

    std::cerr << "DONE with SCC #: " << MyID << "\n";

    // We never have to revisit "SCC" processed functions...
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

DSGraph &BUDataStructures::CreateGraphForExternalFunction(const Function &Fn) {
  Function *F = const_cast<Function*>(&Fn);
  DSGraph *DSG = new DSGraph(GlobalECs, GlobalsGraph->getTargetData());
  DSInfo[F] = DSG;
  DSG->setGlobalsGraph(GlobalsGraph);
  DSG->setPrintAuxCalls();

  // Add function to the graph.
  DSG->getReturnNodes().insert(std::make_pair(F, DSNodeHandle()));

  if (F->getName() == "free") { // Taking the address of free.

    // Free should take a single pointer argument, mark it as heap memory.
    DSNode *N = new DSNode(0, DSG);
    N->setHeapNodeMarker();
    DSG->getNodeForValue(F->arg_begin()).mergeWith(N);

  } else {
    std::cerr << "Unrecognized external function: " << F->getName() << "\n";
    abort();
  }

  return *DSG;
}


bool BUDataStructures::calculateGraph(DSGraph &Graph) {
  // If this graph contains the main function, clone the globals graph into this
  // graph before we inline callees and other fun stuff.
  bool ContainsMain = false;
  DSGraph::ReturnNodesTy &ReturnNodes = Graph.getReturnNodes();

  for (DSGraph::ReturnNodesTy::iterator I = ReturnNodes.begin(),
         E = ReturnNodes.end(); I != E; ++I)
    if (I->first->hasExternalLinkage() && I->first->getName() == "main") {
      ContainsMain = true;
      break;
    }

  // If this graph contains main, copy the contents of the globals graph over.
  // Note that this is *required* for correctness.  If a callee contains a use
  // of a global, we have to make sure to link up nodes due to global-argument
  // bindings.
  if (ContainsMain) {
    const DSGraph &GG = *Graph.getGlobalsGraph();
    ReachabilityCloner RC(Graph, GG,
                          DSGraph::DontCloneCallNodes |
                          DSGraph::DontCloneAuxCallNodes);

    // Clone the global nodes into this graph.
    for (DSScalarMap::global_iterator I = GG.getScalarMap().global_begin(),
           E = GG.getScalarMap().global_end(); I != E; ++I)
      if (isa<GlobalVariable>(*I))
        RC.getClonedNH(GG.getNodeForValue(*I));
  }


  // Move our call site list into TempFCs so that inline call sites go into the
  // new call site list and doesn't invalidate our iterators!
  std::list<DSCallSite> TempFCs;
  std::list<DSCallSite> &AuxCallsList = Graph.getAuxFunctionCalls();
  TempFCs.swap(AuxCallsList);
  //remember what we've seen (or will see)
  unsigned oldSize = TempFCs.size();

  bool Printed = false;
  bool missingNode = false;

  while (!TempFCs.empty()) {
    DSCallSite &CS = *TempFCs.begin();
    Instruction *TheCall = CS.getCallSite().getInstruction();
    DSGraph *GI;

    // Fast path for noop calls.  Note that we don't care about merging globals
    // in the callee with nodes in the caller here.
    if (CS.isDirectCall()) {
      if (!isVAHackFn(CS.getCalleeFunc()) && isResolvableFunc(CS.getCalleeFunc())) {
        Function* Callee = CS.getCalleeFunc();
        ActualCallees.insert(std::make_pair(TheCall, Callee));
	
        assert(doneDSGraph(Callee) && "Direct calls should always be precomputed");
        GI = &getDSGraph(*Callee);  // Graph to inline
        DEBUG(std::cerr << "    Inlining graph for " << Callee->getName());
        DEBUG(std::cerr << "[" << GI->getGraphSize() << "+"
              << GI->getAuxFunctionCalls().size() << "] into '"
              << Graph.getFunctionNames() << "' [" << Graph.getGraphSize() <<"+"
              << Graph.getAuxFunctionCalls().size() << "]\n");
        Graph.mergeInGraph(CS, *Callee, *GI,
                           DSGraph::StripAllocaBit|DSGraph::DontCloneCallNodes);
        ++NumBUInlines;
      } else {
        DEBUG(std::cerr << "Graph " << Graph.getFunctionNames() << " Call Site " <<
              CS.getCallSite().getInstruction() << " never resolvable\n");
      }
      --oldSize;
      TempFCs.pop_front();
      continue;
    } else {
      std::vector<Function*> CalledFuncs;
      bool resolved = GetAllCallees(CS, CalledFuncs);

      if (CalledFuncs.empty()) {
        DEBUG(std::cerr << "Graph " << Graph.getFunctionNames() << " Call Site " <<
              CS.getCallSite().getInstruction() << " delayed\n");
      } else {
        DEBUG(
        if (!Printed)
          std::cerr << "In Fns: " << Graph.getFunctionNames() << "\n";
        std::cerr << "  calls " << CalledFuncs.size()
                  << " fns from site: " << CS.getCallSite().getInstruction()
                  << "  " << *CS.getCallSite().getInstruction();
        std::cerr << "   Fns =";
        );
        unsigned NumPrinted = 0;

        for (std::vector<Function*>::iterator I = CalledFuncs.begin(),
               E = CalledFuncs.end(); I != E; ++I) {
          DEBUG(if (NumPrinted++ < 8) std::cerr << " " << (*I)->getName(););

          // Add the call edges to the call graph.
          ActualCallees.insert(std::make_pair(TheCall, *I));
        }
        DEBUG(std::cerr << "\n");

        // See if we already computed a graph for this set of callees.
        std::sort(CalledFuncs.begin(), CalledFuncs.end());
        std::pair<DSGraph*, std::vector<DSNodeHandle> > &IndCallGraph =
          (*IndCallGraphMap)[CalledFuncs];

        if (IndCallGraph.first == 0) {
          std::vector<Function*>::iterator I = CalledFuncs.begin(),
            E = CalledFuncs.end();

          // Start with a copy of the first graph.
          if (!doneDSGraph(*I)) {
            AuxCallsList.splice(AuxCallsList.end(), TempFCs, TempFCs.begin());
            missingNode = true;
            continue;
          }

          AddGlobalToNode(this, CS, *I);

          GI = IndCallGraph.first = new DSGraph(getDSGraph(**I), GlobalECs);
          GI->setGlobalsGraph(Graph.getGlobalsGraph());
          std::vector<DSNodeHandle> &Args = IndCallGraph.second;

          // Get the argument nodes for the first callee.  The return value is
          // the 0th index in the vector.
          GI->getFunctionArgumentsForCall(*I, Args);

          // Merge all of the other callees into this graph.
          bool locMissing = false;
          for (++I; I != E && !locMissing; ++I) {
            AddGlobalToNode(this, CS, *I);
            // If the graph already contains the nodes for the function, don't
            // bother merging it in again.
            if (!GI->containsFunction(*I)) {
              if (!doneDSGraph(*I)) {
                locMissing = true;
                break;
              }

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
          if (locMissing) {
            AuxCallsList.splice(AuxCallsList.end(), TempFCs, TempFCs.begin());
            missingNode = true;
            continue;
          }

          // Clean up the final graph!
          GI->removeDeadNodes(DSGraph::KeepUnreachableGlobals);
        } else {
          DEBUG(std::cerr << "***\n*** RECYCLED GRAPH ***\n***\n");
          for (std::vector<Function*>::iterator I = CalledFuncs.begin(), E = CalledFuncs.end(); I != E; ++I) {
            AddGlobalToNode(this, CS, *I);
          }
        }

        GI = IndCallGraph.first;

        if (AlreadyInlined[CS.getCallSite()] != CalledFuncs) {
         AlreadyInlined[CS.getCallSite()].swap(CalledFuncs);

          // Merge the unified graph into this graph now.
          DEBUG(std::cerr << "    Inlining multi callee graph "
                << "[" << GI->getGraphSize() << "+"
                << GI->getAuxFunctionCalls().size() << "] into '"
                << Graph.getFunctionNames() << "' [" << Graph.getGraphSize() <<"+"
                << Graph.getAuxFunctionCalls().size() << "]\n");

          Graph.mergeInGraph(CS, IndCallGraph.second, *GI,
                             DSGraph::StripAllocaBit |
                             DSGraph::DontCloneCallNodes);

          ++NumBUInlines;
        } else {
          DEBUG(std::cerr << "   Skipping already inlined graph\n");
        }
      }
      AuxCallsList.splice(AuxCallsList.end(), TempFCs, TempFCs.begin());
    }
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
  AuxCallsList.sort();
  AuxCallsList.unique();
  //conditionally prune the call list keeping only one copy of each actual
  //CallSite
  if (AuxCallsList.size() > 100) {
    DEBUG(std::cerr << "Reducing Aux from " << AuxCallsList.size());
    std::map<CallSite, std::list<DSCallSite>::iterator> keepers;
    TempFCs.swap(AuxCallsList);
    for( std::list<DSCallSite>::iterator ii = TempFCs.begin(), ee = TempFCs.end();
         ii != ee; ++ii)
      keepers[ii->getCallSite()] = ii;
    for (std::map<CallSite, std::list<DSCallSite>::iterator>::iterator
           ii = keepers.begin(), ee = keepers.end();
         ii != ee; ++ii)
      AuxCallsList.splice(AuxCallsList.end(), TempFCs, ii->second);
    DEBUG(std::cerr << " to " << AuxCallsList.size() << "\n");
  }
  return missingNode || oldSize != AuxCallsList.size();
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
