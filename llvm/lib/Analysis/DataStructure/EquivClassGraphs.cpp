//===- EquivClassGraphs.cpp - Merge equiv-class graphs & inline bottom-up -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is the same as the complete bottom-up graphs, but
// with functions partitioned into equivalence classes and a single merged
// DS graph for all functions in an equivalence class.  After this merging,
// graphs are inlined bottom-up on the SCCs of the final (CBU) call graph.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ECGraphs"
#include "llvm/Analysis/DataStructure/DataStructure.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/DataStructure/DSGraph.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include <iostream>
using namespace llvm;

namespace {
  RegisterPass<EquivClassGraphs> X("eqdatastructure",
                    "Equivalence-class Bottom-up Data Structure Analysis");
  Statistic<> NumEquivBUInlines("equivdatastructures",
                                "Number of graphs inlined");
  Statistic<> NumFoldGraphInlines("Inline equiv-class graphs bottom up",
                                  "Number of graphs inlined");
}

#ifndef NDEBUG
template<typename GT>
static void CheckAllGraphs(Module *M, GT &ECGraphs) {
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!I->isExternal()) {
      DSGraph &G = ECGraphs.getDSGraph(*I);
      if (G.retnodes_begin()->first != I)
        continue;  // Only check a graph once.

      DSGraph::NodeMapTy GlobalsGraphNodeMapping;
      G.computeGToGGMapping(GlobalsGraphNodeMapping);
    }
}
#endif

// getSomeCalleeForCallSite - Return any one callee function at a call site.
//
Function *EquivClassGraphs::getSomeCalleeForCallSite(const CallSite &CS) const{
  Function *thisFunc = CS.getCaller();
  assert(thisFunc && "getSomeCalleeForCallSite(): Not a valid call site?");
  DSGraph &DSG = getDSGraph(*thisFunc);
  DSNode *calleeNode = DSG.getNodeForValue(CS.getCalledValue()).getNode();
  std::map<DSNode*, Function *>::const_iterator I =
    OneCalledFunction.find(calleeNode);
  return (I == OneCalledFunction.end())? NULL : I->second;
}

// runOnModule - Calculate the bottom up data structure graphs for each function
// in the program.
//
bool EquivClassGraphs::runOnModule(Module &M) {
  CBU = &getAnalysis<CompleteBUDataStructures>();
  GlobalECs = CBU->getGlobalECs();
  DEBUG(CheckAllGraphs(&M, *CBU));

  GlobalsGraph = new DSGraph(CBU->getGlobalsGraph(), GlobalECs);
  GlobalsGraph->setPrintAuxCalls();

  ActualCallees = CBU->getActualCallees();

  // Find equivalence classes of functions called from common call sites.
  // Fold the CBU graphs for all functions in an equivalence class.
  buildIndirectFunctionSets(M);

  // Stack of functions used for Tarjan's SCC-finding algorithm.
  std::vector<DSGraph*> Stack;
  std::map<DSGraph*, unsigned> ValMap;
  unsigned NextID = 1;

  Function *MainFunc = M.getMainFunction();
  if (MainFunc && !MainFunc->isExternal()) {
    processSCC(getOrCreateGraph(*MainFunc), Stack, NextID, ValMap);
  } else {
    std::cerr << "Fold Graphs: No 'main' function found!\n";
  }

  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal())
      processSCC(getOrCreateGraph(*I), Stack, NextID, ValMap);

  DEBUG(CheckAllGraphs(&M, *this));

  getGlobalsGraph().removeTriviallyDeadNodes();
  getGlobalsGraph().markIncompleteNodes(DSGraph::IgnoreGlobals);

  // Merge the globals variables (not the calls) from the globals graph back
  // into the main function's graph so that the main function contains all of
  // the information about global pools and GV usage in the program.
  if (MainFunc && !MainFunc->isExternal()) {
    DSGraph &MainGraph = getOrCreateGraph(*MainFunc);
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

  // Final processing.  Note that dead node elimination may actually remove
  // globals from a function graph that are immediately used.  If there are no
  // scalars pointing to the node (e.g. because the only use is a direct store
  // to a scalar global) we have to make sure to rematerialize the globals back
  // into the graphs here, or clients will break!
  for (Module::global_iterator GI = M.global_begin(), E = M.global_end();
       GI != E; ++GI)
    // This only happens to first class typed globals.
    if (GI->getType()->getElementType()->isFirstClassType())
      for (Value::use_iterator UI = GI->use_begin(), E = GI->use_end();
           UI != E; ++UI)
        // This only happens to direct uses by instructions.
        if (Instruction *User = dyn_cast<Instruction>(*UI)) {
          DSGraph &DSG = getOrCreateGraph(*User->getParent()->getParent());
          if (!DSG.getScalarMap().count(GI)) {
            // If this global does not exist in the graph, but it is immediately
            // used by an instruction in the graph, clone it over from the
            // globals graph.
            ReachabilityCloner RC(DSG, *GlobalsGraph, 0);
            RC.getClonedNH(GlobalsGraph->getNodeForValue(GI));
          }
        }

  return false;
}


// buildIndirectFunctionSets - Iterate over the module looking for indirect
// calls to functions.  If a call site can invoke any functions [F1, F2... FN],
// unify the N functions together in the FuncECs set.
//
void EquivClassGraphs::buildIndirectFunctionSets(Module &M) {
  const ActualCalleesTy& AC = CBU->getActualCallees();

  // Loop over all of the indirect calls in the program.  If a call site can
  // call multiple different functions, we need to unify all of the callees into
  // the same equivalence class.
  Instruction *LastInst = 0;
  Function *FirstFunc = 0;
  for (ActualCalleesTy::const_iterator I=AC.begin(), E=AC.end(); I != E; ++I) {
    if (I->second->isExternal())
      continue;                         // Ignore functions we cannot modify

    CallSite CS = CallSite::get(I->first);

    if (CS.getCalledFunction()) {       // Direct call:
      FuncECs.insert(I->second);        // -- Make sure function has equiv class
      FirstFunc = I->second;            // -- First callee at this site
    } else {                            // Else indirect call
      // DEBUG(std::cerr << "CALLEE: " << I->second->getName()
      //       << " from : " << I->first);
      if (I->first != LastInst) {
        // This is the first callee from this call site.
        LastInst = I->first;
        FirstFunc = I->second;
        // Instead of storing the lastInst For Indirection call Sites we store
        // the DSNode for the function ptr arguemnt
        Function *thisFunc = LastInst->getParent()->getParent();
        DSGraph &TFG = CBU->getDSGraph(*thisFunc);
        DSNode *calleeNode = TFG.getNodeForValue(CS.getCalledValue()).getNode();
        OneCalledFunction[calleeNode] = FirstFunc;
        FuncECs.insert(I->second);
      } else {
        // This is not the first possible callee from a particular call site.
        // Union the callee in with the other functions.
        FuncECs.unionSets(FirstFunc, I->second);
#ifndef NDEBUG
        Function *thisFunc = LastInst->getParent()->getParent();
        DSGraph &TFG = CBU->getDSGraph(*thisFunc);
        DSNode *calleeNode = TFG.getNodeForValue(CS.getCalledValue()).getNode();
        assert(OneCalledFunction.count(calleeNode) > 0 && "Missed a call?");
#endif
      }
    }

    // Now include all functions that share a graph with any function in the
    // equivalence class.  More precisely, if F is in the class, and G(F) is
    // its graph, then we include all other functions that are also in G(F).
    // Currently, that is just the functions in the same call-graph-SCC as F.
    //
    DSGraph& funcDSGraph = CBU->getDSGraph(*I->second);
    for (DSGraph::retnodes_iterator RI = funcDSGraph.retnodes_begin(),
           RE = funcDSGraph.retnodes_end(); RI != RE; ++RI)
      FuncECs.unionSets(FirstFunc, RI->first);
  }

  // Now that all of the equivalences have been built, merge the graphs for
  // each equivalence class.
  //
  DEBUG(std::cerr << "\nIndirect Function Equivalence Sets:\n");
  for (EquivalenceClasses<Function*>::iterator EQSI = FuncECs.begin(), E =
         FuncECs.end(); EQSI != E; ++EQSI) {
    if (!EQSI->isLeader()) continue;

    EquivalenceClasses<Function*>::member_iterator SI =
      FuncECs.member_begin(EQSI);
    assert(SI != FuncECs.member_end() && "Empty equiv set??");
    EquivalenceClasses<Function*>::member_iterator SN = SI;
    ++SN;
    if (SN == FuncECs.member_end())
      continue;   // Single function equivalence set, no merging to do.

    Function* LF = *SI;

#ifndef NDEBUG
    DEBUG(std::cerr <<"  Equivalence set for leader " << LF->getName() <<" = ");
    for (SN = SI; SN != FuncECs.member_end(); ++SN)
      DEBUG(std::cerr << " " << (*SN)->getName() << "," );
    DEBUG(std::cerr << "\n");
#endif

    // This equiv class has multiple functions: merge their graphs.  First,
    // clone the CBU graph for the leader and make it the common graph for the
    // equivalence graph.
    DSGraph &MergedG = getOrCreateGraph(*LF);

    // Record the argument nodes for use in merging later below.
    std::vector<DSNodeHandle> ArgNodes;

    for (Function::arg_iterator AI = LF->arg_begin(), E = LF->arg_end();
         AI != E; ++AI)
      if (DS::isPointerType(AI->getType()))
        ArgNodes.push_back(MergedG.getNodeForValue(AI));

    // Merge in the graphs of all other functions in this equiv. class.  Note
    // that two or more functions may have the same graph, and it only needs
    // to be merged in once.
    std::set<DSGraph*> GraphsMerged;
    GraphsMerged.insert(&CBU->getDSGraph(*LF));

    for (++SI; SI != FuncECs.member_end(); ++SI) {
      Function *F = *SI;
      DSGraph *&FG = DSInfo[F];

      DSGraph &CBUGraph = CBU->getDSGraph(*F);
      if (GraphsMerged.insert(&CBUGraph).second) {
        // Record the "folded" graph for the function.
        for (DSGraph::retnodes_iterator I = CBUGraph.retnodes_begin(),
               E = CBUGraph.retnodes_end(); I != E; ++I) {
          assert(DSInfo[I->first] == 0 && "Graph already exists for Fn!");
          DSInfo[I->first] = &MergedG;
        }

        // Clone this member of the equivalence class into MergedG.
        MergedG.cloneInto(CBUGraph);
      }

      // Merge the return nodes of all functions together.
      MergedG.getReturnNodes()[LF].mergeWith(MergedG.getReturnNodes()[F]);

      // Merge the function arguments with all argument nodes found so far.
      // If there are extra function args, add them to the vector of argNodes
      Function::arg_iterator AI2 = F->arg_begin(), AI2end = F->arg_end();
      for (unsigned arg = 0, numArgs = ArgNodes.size();
           arg != numArgs && AI2 != AI2end; ++AI2, ++arg)
        if (DS::isPointerType(AI2->getType()))
          ArgNodes[arg].mergeWith(MergedG.getNodeForValue(AI2));

      for ( ; AI2 != AI2end; ++AI2)
        if (DS::isPointerType(AI2->getType()))
          ArgNodes.push_back(MergedG.getNodeForValue(AI2));
      DEBUG(MergedG.AssertGraphOK());
    }
  }
  DEBUG(std::cerr << "\n");
}


DSGraph &EquivClassGraphs::getOrCreateGraph(Function &F) {
  // Has the graph already been created?
  DSGraph *&Graph = DSInfo[&F];
  if (Graph) return *Graph;

  DSGraph &CBUGraph = CBU->getDSGraph(F);

  // Copy the CBU graph...
  Graph = new DSGraph(CBUGraph, GlobalECs);   // updates the map via reference
  Graph->setGlobalsGraph(&getGlobalsGraph());
  Graph->setPrintAuxCalls();

  // Make sure to update the DSInfo map for all functions in the graph!
  for (DSGraph::retnodes_iterator I = Graph->retnodes_begin();
       I != Graph->retnodes_end(); ++I)
    if (I->first != &F) {
      DSGraph *&FG = DSInfo[I->first];
      assert(FG == 0 && "Merging function in SCC twice?");
      FG = Graph;
    }

  return *Graph;
}


unsigned EquivClassGraphs::
processSCC(DSGraph &FG, std::vector<DSGraph*> &Stack, unsigned &NextID,
           std::map<DSGraph*, unsigned> &ValMap) {
  std::map<DSGraph*, unsigned>::iterator It = ValMap.lower_bound(&FG);
  if (It != ValMap.end() && It->first == &FG)
    return It->second;

  DEBUG(std::cerr << "    ProcessSCC for function " << FG.getFunctionNames()
                  << "\n");

  unsigned Min = NextID++, MyID = Min;
  ValMap[&FG] = Min;
  Stack.push_back(&FG);

  // The edges out of the current node are the call site targets...
  for (DSGraph::fc_iterator CI = FG.fc_begin(), CE = FG.fc_end();
       CI != CE; ++CI) {
    Instruction *Call = CI->getCallSite().getInstruction();

    // Loop over all of the actually called functions...
    for (callee_iterator I = callee_begin(Call), E = callee_end(Call);
         I != E; ++I)
      if (!I->second->isExternal()) {
        // Process the callee as necessary.
        unsigned M = processSCC(getOrCreateGraph(*I->second),
                                Stack, NextID, ValMap);
        if (M < Min) Min = M;
      }
  }

  assert(ValMap[&FG] == MyID && "SCC construction assumption wrong!");
  if (Min != MyID)
    return Min;         // This is part of a larger SCC!

  // If this is a new SCC, process it now.
  bool MergedGraphs = false;
  while (Stack.back() != &FG) {
    DSGraph *NG = Stack.back();
    ValMap[NG] = ~0U;

    // If the SCC found is not the same as those found in CBU, make sure to
    // merge the graphs as appropriate.
    FG.cloneInto(*NG);

    // Update the DSInfo map and delete the old graph...
    for (DSGraph::retnodes_iterator I = NG->retnodes_begin();
         I != NG->retnodes_end(); ++I)
      DSInfo[I->first] = &FG;

    // Remove NG from the ValMap since the pointer may get recycled.
    ValMap.erase(NG);
    delete NG;
    MergedGraphs = true;
    Stack.pop_back();
  }

  // Clean up the graph before we start inlining a bunch again.
  if (MergedGraphs)
    FG.removeTriviallyDeadNodes();

  Stack.pop_back();

  processGraph(FG);
  ValMap[&FG] = ~0U;
  return MyID;
}


/// processGraph - Process the CBU graphs for the program in bottom-up order on
/// the SCC of the __ACTUAL__ call graph.  This builds final folded CBU graphs.
void EquivClassGraphs::processGraph(DSGraph &G) {
  DEBUG(std::cerr << "    ProcessGraph for function "
                  << G.getFunctionNames() << "\n");

  hash_set<Instruction*> calls;

  // Else we need to inline some callee graph.  Visit all call sites.
  // The edges out of the current node are the call site targets...
  unsigned i = 0;
  for (DSGraph::fc_iterator CI = G.fc_begin(), CE = G.fc_end(); CI != CE;
       ++CI, ++i) {
    const DSCallSite &CS = *CI;
    Instruction *TheCall = CS.getCallSite().getInstruction();

    assert(calls.insert(TheCall).second &&
           "Call instruction occurs multiple times in graph??");

    if (CS.getRetVal().isNull() && CS.getNumPtrArgs() == 0)
      continue;

    // Inline the common callee graph into the current graph, if the callee
    // graph has not changed.  Note that all callees should have the same
    // graph so we only need to do this once.
    //
    DSGraph* CalleeGraph = NULL;
    callee_iterator I = callee_begin(TheCall), E = callee_end(TheCall);
    unsigned TNum, Num;

    // Loop over all potential callees to find the first non-external callee.
    for (TNum = 0, Num = std::distance(I, E); I != E; ++I, ++TNum)
      if (!I->second->isExternal())
        break;

    // Now check if the graph has changed and if so, clone and inline it.
    if (I != E) {
      Function *CalleeFunc = I->second;

      // Merge the callee's graph into this graph, if not already the same.
      // Callees in the same equivalence class (which subsumes those
      // in the same SCCs) have the same graph.  Note that all recursion
      // including self-recursion have been folded in the equiv classes.
      //
      CalleeGraph = &getOrCreateGraph(*CalleeFunc);
      if (CalleeGraph != &G) {
        ++NumFoldGraphInlines;
        G.mergeInGraph(CS, *CalleeFunc, *CalleeGraph,
                       DSGraph::StripAllocaBit |
                       DSGraph::DontCloneCallNodes |
                       DSGraph::DontCloneAuxCallNodes);
        DEBUG(std::cerr << "    Inlining graph [" << i << "/"
              << G.getFunctionCalls().size()-1
              << ":" << TNum << "/" << Num-1 << "] for "
              << CalleeFunc->getName() << "["
              << CalleeGraph->getGraphSize() << "+"
              << CalleeGraph->getAuxFunctionCalls().size()
              << "] into '" /*<< G.getFunctionNames()*/ << "' ["
              << G.getGraphSize() << "+" << G.getAuxFunctionCalls().size()
              << "]\n");
      }
    }

#ifndef NDEBUG
    // Now loop over the rest of the callees and make sure they have the
    // same graph as the one inlined above.
    if (CalleeGraph)
      for (++I, ++TNum; I != E; ++I, ++TNum)
        if (!I->second->isExternal())
          assert(CalleeGraph == &getOrCreateGraph(*I->second) &&
                 "Callees at a call site have different graphs?");
#endif
  }

  // Recompute the Incomplete markers.
  G.maskIncompleteMarkers();
  G.markIncompleteNodes(DSGraph::MarkFormalArgs);

  // Delete dead nodes.  Treat globals that are unreachable but that can
  // reach live nodes as live.
  G.removeDeadNodes(DSGraph::KeepUnreachableGlobals);

  // When this graph is finalized, clone the globals in the graph into the
  // globals graph to make sure it has everything, from all graphs.
  ReachabilityCloner RC(*G.getGlobalsGraph(), G, DSGraph::StripAllocaBit);

  // Clone everything reachable from globals in the function graph into the
  // globals graph.
  DSScalarMap &MainSM = G.getScalarMap();
  for (DSScalarMap::global_iterator I = MainSM.global_begin(),
         E = MainSM.global_end(); I != E; ++I)
    RC.getClonedNH(MainSM[*I]);

  DEBUG(std::cerr << "  -- DONE ProcessGraph for function "
                  << G.getFunctionNames() << "\n");
}
