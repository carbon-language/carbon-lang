//===- BottomUpClosure.cpp - Compute bottom-up interprocedural closure ----===//
//
// This file implements the BUDataStructures class, which represents the
// Bottom-Up Interprocedural closure of the data structure graph over the
// program.  This is useful for applications like pool allocation, but **not**
// applications like alias analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DSGraph.h"
#include "llvm/Module.h"
#include "Support/Statistic.h"
#include "Support/hash_map"

namespace {
  Statistic<> MaxSCC("budatastructure", "Maximum SCC Size in Call Graph");
  
  RegisterAnalysis<BUDataStructures>
  X("budatastructure", "Bottom-up Data Structure Analysis Closure");
}

using namespace DS;

// isCompleteNode - Return true if we know all of the targets of this node, and
// if the call sites are not external.
//
static inline bool isCompleteNode(DSNode *N) {
  if (N->NodeType & DSNode::Incomplete) return false;
  const std::vector<GlobalValue*> &Callees = N->getGlobals();
  for (unsigned i = 0, e = Callees.size(); i != e; ++i)
    if (Callees[i]->isExternal()) {
      GlobalValue &FI = cast<Function>(*Callees[i]);
      if (FI.getName() != "printf"  && FI.getName() != "sscanf" &&
          FI.getName() != "fprintf" && FI.getName() != "open" &&
          FI.getName() != "sprintf" && FI.getName() != "fputs" &&
          FI.getName() != "fscanf")
        return false;  // External function found...
    }
  return true;  // otherwise ok
}

struct CallSiteIterator {
  // FCs are the edges out of the current node are the call site targets...
  std::vector<DSCallSite> *FCs;
  unsigned CallSite;
  unsigned CallSiteEntry;

  CallSiteIterator(std::vector<DSCallSite> &CS) : FCs(&CS) {
    CallSite = 0; CallSiteEntry = 0;
    advanceToNextValid();
  }

  // End iterator ctor...
  CallSiteIterator(std::vector<DSCallSite> &CS, bool) : FCs(&CS) {
    CallSite = FCs->size(); CallSiteEntry = 0;
  }

  void advanceToNextValid() {
    while (CallSite < FCs->size()) {
      if (DSNode *CalleeNode = (*FCs)[CallSite].getCallee().getNode()) {
        if (CallSiteEntry || isCompleteNode(CalleeNode)) {
          const std::vector<GlobalValue*> &Callees = CalleeNode->getGlobals();
          
          if (CallSiteEntry < Callees.size())
            return;
        }
        CallSiteEntry = 0;
        ++CallSite;
      }
    }
  }
public:
  static CallSiteIterator begin(DSGraph &G) { return G.getAuxFunctionCalls(); }
  static CallSiteIterator end(DSGraph &G) {
    return CallSiteIterator(G.getAuxFunctionCalls(), true);
  }
  static CallSiteIterator begin(std::vector<DSCallSite> &CSs) { return CSs; }
  static CallSiteIterator end(std::vector<DSCallSite> &CSs) {
    return CallSiteIterator(CSs, true);
  }
  bool operator==(const CallSiteIterator &CSI) const {
    return CallSite == CSI.CallSite && CallSiteEntry == CSI.CallSiteEntry;
  }
  bool operator!=(const CallSiteIterator &CSI) const { return !operator==(CSI);}

  unsigned getCallSiteIdx() const { return CallSite; }
  DSCallSite &getCallSite() const { return (*FCs)[CallSite]; }

  Function* operator*() const {
    DSNode *Node = (*FCs)[CallSite].getCallee().getNode();
    return cast<Function>(Node->getGlobals()[CallSiteEntry]);
  }

  CallSiteIterator& operator++() {                // Preincrement
    ++CallSiteEntry;
    advanceToNextValid();
    return *this;
  }
  CallSiteIterator operator++(int) { // Postincrement
    CallSiteIterator tmp = *this; ++*this; return tmp; 
  }
};



// run - Calculate the bottom up data structure graphs for each function in the
// program.
//
bool BUDataStructures::run(Module &M) {
  GlobalsGraph = new DSGraph();
  GlobalsGraph->setPrintAuxCalls();

  Function *MainFunc = M.getMainFunction();
  if (MainFunc)
    calculateReachableGraphs(MainFunc);

  // Calculate the graphs for any functions that are unreachable from main...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal() && DSInfo.find(I) == DSInfo.end()) {
#ifndef NDEBUG
      if (MainFunc)
        std::cerr << "*** Function unreachable from main: "
                  << I->getName() << "\n";
#endif
      calculateReachableGraphs(I);    // Calculate all graphs...
    }
  return false;
}

void BUDataStructures::calculateReachableGraphs(Function *F) {
  std::vector<Function*> Stack;
  hash_map<Function*, unsigned> ValMap;
  unsigned NextID = 1;
  calculateGraphs(F, Stack, NextID, ValMap);
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
  assert(ValMap.find(F) == ValMap.end() && "Shouldn't revisit functions!");
  unsigned Min = NextID++, MyID = Min;
  ValMap[F] = Min;
  Stack.push_back(F);

  if (F->isExternal()) {   // sprintf, fprintf, sscanf, etc...
    // No callees!
    Stack.pop_back();
    ValMap[F] = ~0;
    return Min;
  }

  DSGraph &Graph = getOrCreateGraph(F);

  // The edges out of the current node are the call site targets...
  for (CallSiteIterator I = CallSiteIterator::begin(Graph),
         E = CallSiteIterator::end(Graph); I != E; ++I) {
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
    DSGraph &G = calculateGraph(*F);

    if (MaxSCC < 1) MaxSCC = 1;

    // Should we revisit the graph?
    if (CallSiteIterator::begin(G) != CallSiteIterator::end(G)) {
      ValMap.erase(F);
      return calculateGraphs(F, Stack, NextID, ValMap);
    } else {
      ValMap[F] = ~0U;
    }
    return MyID;

  } else {
    // SCCFunctions - Keep track of the functions in the current SCC
    //
    hash_set<Function*> SCCFunctions;

    Function *NF;
    std::vector<Function*>::iterator FirstInSCC = Stack.end();
    do {
      NF = *--FirstInSCC;
      ValMap[NF] = ~0U;
      SCCFunctions.insert(NF);
    } while (NF != F);

    std::cerr << "Identified SCC #: " << MyID << " of size: "
              << (Stack.end()-FirstInSCC) << "\n";

    // Compute the Max SCC Size...
    if (MaxSCC < unsigned(Stack.end()-FirstInSCC))
      MaxSCC = Stack.end()-FirstInSCC;

    std::vector<Function*>::iterator I = Stack.end();
    do {
      --I;
#ifndef NDEBUG
      /*DEBUG*/(std::cerr << "  Fn #" << (Stack.end()-I) << "/"
            << (Stack.end()-FirstInSCC) << " in SCC: "
            << (*I)->getName());
      DSGraph &G = getDSGraph(**I);
      std::cerr << " [" << G.getGraphSize() << "+"
                << G.getAuxFunctionCalls().size() << "] ";
#endif

      // Eliminate all call sites in the SCC that are not to functions that are
      // in the SCC.
      inlineNonSCCGraphs(**I, SCCFunctions);

#ifndef NDEBUG
      std::cerr << "after Non-SCC's [" << G.getGraphSize() << "+"
                << G.getAuxFunctionCalls().size() << "]\n";
#endif
    } while (I != FirstInSCC);

    I = Stack.end();
    do {
      --I;
#ifndef NDEBUG
      /*DEBUG*/(std::cerr << "  Fn #" << (Stack.end()-I) << "/"
            << (Stack.end()-FirstInSCC) << " in SCC: "
            << (*I)->getName());
      DSGraph &G = getDSGraph(**I);
      std::cerr << " [" << G.getGraphSize() << "+"
                << G.getAuxFunctionCalls().size() << "] ";
#endif
      // Inline all graphs into the SCC nodes...
      calculateSCCGraph(**I, SCCFunctions);

#ifndef NDEBUG
      std::cerr << "after [" << G.getGraphSize() << "+"
                << G.getAuxFunctionCalls().size() << "]\n";
#endif
    } while (I != FirstInSCC);


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
  for (hash_map<const Function*, DSGraph*>::iterator I = DSInfo.begin(),
         E = DSInfo.end(); I != E; ++I)
    delete I->second;

  // Empty map so next time memory is released, data structures are not
  // re-deleted.
  DSInfo.clear();
  delete GlobalsGraph;
  GlobalsGraph = 0;
}

DSGraph &BUDataStructures::calculateGraph(Function &F) {
  DSGraph &Graph = getDSGraph(F);
  DEBUG(std::cerr << "  [BU] Calculating graph for: " << F.getName() << "\n");

  // Move our call site list into TempFCs so that inline call sites go into the
  // new call site list and doesn't invalidate our iterators!
  std::vector<DSCallSite> TempFCs;
  std::vector<DSCallSite> &AuxCallsList = Graph.getAuxFunctionCalls();
  TempFCs.swap(AuxCallsList);

  // Loop over all of the resolvable call sites
  unsigned LastCallSiteIdx = ~0U;
  for (CallSiteIterator I = CallSiteIterator::begin(TempFCs),
         E = CallSiteIterator::end(TempFCs); I != E; ++I) {
    // If we skipped over any call sites, they must be unresolvable, copy them
    // to the real call site list.
    LastCallSiteIdx++;
    for (; LastCallSiteIdx < I.getCallSiteIdx(); ++LastCallSiteIdx)
      AuxCallsList.push_back(TempFCs[LastCallSiteIdx]);
    LastCallSiteIdx = I.getCallSiteIdx();
    
    // Resolve the current call...
    Function *Callee = *I;
    DSCallSite &CS = I.getCallSite();

    if (Callee->isExternal()) {
      // Ignore this case, simple varargs functions we cannot stub out!
    } else if (Callee == &F) {
      // Self recursion... simply link up the formal arguments with the
      // actual arguments...
      DEBUG(std::cerr << "    Self Inlining: " << F.getName() << "\n");

      // Handle self recursion by resolving the arguments and return value
      Graph.mergeInGraph(CS, Graph, 0);

    } else {
      // Get the data structure graph for the called function.
      //
      DSGraph &GI = getDSGraph(*Callee);  // Graph to inline
      
      DEBUG(std::cerr << "    Inlining graph for " << Callee->getName()
            << "[" << GI.getGraphSize() << "+"
            << GI.getAuxFunctionCalls().size() << "] into: " << F.getName()
            << "[" << Graph.getGraphSize() << "+"
            << Graph.getAuxFunctionCalls().size() << "]\n");
#if 0
      Graph.writeGraphToFile(std::cerr, "bu_" + F.getName() + "_before_" +
                             Callee->getName());
#endif
      
      // Handle self recursion by resolving the arguments and return value
      Graph.mergeInGraph(CS, GI,
                         DSGraph::KeepModRefBits | 
                         DSGraph::StripAllocaBit | DSGraph::DontCloneCallNodes);

#if 0
      Graph.writeGraphToFile(std::cerr, "bu_" + F.getName() + "_after_" +
                             Callee->getName());
#endif
    }
  }

  // Make sure to catch any leftover unresolvable calls...
  for (++LastCallSiteIdx; LastCallSiteIdx < TempFCs.size(); ++LastCallSiteIdx)
    AuxCallsList.push_back(TempFCs[LastCallSiteIdx]);

  TempFCs.clear();

  // Recompute the Incomplete markers.  If there are any function calls left
  // now that are complete, we must loop!
  Graph.maskIncompleteMarkers();
  Graph.markIncompleteNodes(DSGraph::MarkFormalArgs);
  // FIXME: materialize nodes from the globals graph as neccesary...
  Graph.removeDeadNodes(DSGraph::KeepUnreachableGlobals);

  DEBUG(std::cerr << "  [BU] Done inlining: " << F.getName() << " ["
        << Graph.getGraphSize() << "+" << Graph.getAuxFunctionCalls().size()
        << "]\n");

  //Graph.writeGraphToFile(std::cerr, "bu_" + F.getName());

  return Graph;
}


// inlineNonSCCGraphs - This method is almost like the other two calculate graph
// methods.  This one is used to inline function graphs (from functions outside
// of the SCC) into functions in the SCC.  It is not supposed to touch functions
// IN the SCC at all.
//
DSGraph &BUDataStructures::inlineNonSCCGraphs(Function &F,
                                             hash_set<Function*> &SCCFunctions){
  DSGraph &Graph = getDSGraph(F);
  DEBUG(std::cerr << "  [BU] Inlining Non-SCC graphs for: "
                  << F.getName() << "\n");

  // Move our call site list into TempFCs so that inline call sites go into the
  // new call site list and doesn't invalidate our iterators!
  std::vector<DSCallSite> TempFCs;
  std::vector<DSCallSite> &AuxCallsList = Graph.getAuxFunctionCalls();
  TempFCs.swap(AuxCallsList);

  // Loop over all of the resolvable call sites
  unsigned LastCallSiteIdx = ~0U;
  for (CallSiteIterator I = CallSiteIterator::begin(TempFCs),
         E = CallSiteIterator::end(TempFCs); I != E; ++I) {
    // If we skipped over any call sites, they must be unresolvable, copy them
    // to the real call site list.
    LastCallSiteIdx++;
    for (; LastCallSiteIdx < I.getCallSiteIdx(); ++LastCallSiteIdx)
      AuxCallsList.push_back(TempFCs[LastCallSiteIdx]);
    LastCallSiteIdx = I.getCallSiteIdx();
    
    // Resolve the current call...
    Function *Callee = *I;
    DSCallSite &CS = I.getCallSite();

    if (Callee->isExternal()) {
      // Ignore this case, simple varargs functions we cannot stub out!
    } else if (SCCFunctions.count(Callee)) {
      // Calling a function in the SCC, ignore it for now!
      DEBUG(std::cerr << "    SCC CallSite for: " << Callee->getName() << "\n");
      AuxCallsList.push_back(CS);
    } else {
      // Get the data structure graph for the called function.
      //
      DSGraph &GI = getDSGraph(*Callee);  // Graph to inline

      DEBUG(std::cerr << "    Inlining graph for " << Callee->getName()
            << "[" << GI.getGraphSize() << "+"
            << GI.getAuxFunctionCalls().size() << "] into: " << F.getName()
            << "[" << Graph.getGraphSize() << "+"
            << Graph.getAuxFunctionCalls().size() << "]\n");

      // Handle self recursion by resolving the arguments and return value
      Graph.mergeInGraph(CS, GI,
                         DSGraph::KeepModRefBits | DSGraph::StripAllocaBit |
                         DSGraph::DontCloneCallNodes);
    }
  }

  // Make sure to catch any leftover unresolvable calls...
  for (++LastCallSiteIdx; LastCallSiteIdx < TempFCs.size(); ++LastCallSiteIdx)
    AuxCallsList.push_back(TempFCs[LastCallSiteIdx]);

  TempFCs.clear();

  // Recompute the Incomplete markers.  If there are any function calls left
  // now that are complete, we must loop!
  Graph.maskIncompleteMarkers();
  Graph.markIncompleteNodes(DSGraph::MarkFormalArgs);
  Graph.removeDeadNodes(DSGraph::KeepUnreachableGlobals);

  DEBUG(std::cerr << "  [BU] Done Non-SCC inlining: " << F.getName() << " ["
        << Graph.getGraphSize() << "+" << Graph.getAuxFunctionCalls().size()
        << "]\n");
  //Graph.writeGraphToFile(std::cerr, "nscc_" + F.getName());
  return Graph;
}


DSGraph &BUDataStructures::calculateSCCGraph(Function &F,
                                             hash_set<Function*> &SCCFunctions){
  DSGraph &Graph = getDSGraph(F);
  DEBUG(std::cerr << "  [BU] Calculating SCC graph for: " << F.getName()<<"\n");

  std::vector<DSCallSite> UnresolvableCalls;
  hash_map<Function*, DSCallSite> SCCCallSiteMap;
  std::vector<DSCallSite> &AuxCallsList = Graph.getAuxFunctionCalls();

  while (1) {  // Loop until we run out of resolvable call sites!
    // Move our call site list into TempFCs so that inline call sites go into
    // the new call site list and doesn't invalidate our iterators!
    std::vector<DSCallSite> TempFCs;
    TempFCs.swap(AuxCallsList);

    // Loop over all of the resolvable call sites
    unsigned LastCallSiteIdx = ~0U;
    CallSiteIterator I = CallSiteIterator::begin(TempFCs),
      E = CallSiteIterator::end(TempFCs);
    if (I == E) {
      TempFCs.swap(AuxCallsList);
      break;  // Done when no resolvable call sites exist
    }

    for (; I != E; ++I) {
      // If we skipped over any call sites, they must be unresolvable, copy them
      // to the unresolvable site list.
      LastCallSiteIdx++;
      for (; LastCallSiteIdx < I.getCallSiteIdx(); ++LastCallSiteIdx)
        UnresolvableCalls.push_back(TempFCs[LastCallSiteIdx]);
      LastCallSiteIdx = I.getCallSiteIdx();
      
      // Resolve the current call...
      Function *Callee = *I;
      DSCallSite &CS = I.getCallSite();
      
      if (Callee->isExternal()) {
        // Ignore this case, simple varargs functions we cannot stub out!
      } else if (Callee == &F) {
        // Self recursion... simply link up the formal arguments with the
        // actual arguments...
        DEBUG(std::cerr << "    Self Inlining: " << F.getName() << "\n");
        
        // Handle self recursion by resolving the arguments and return value
        Graph.mergeInGraph(CS, Graph, 0);
      } else if (SCCCallSiteMap.count(Callee)) {
        // We have already seen a call site in the SCC for this function, just
        // merge the two call sites together and we are done.
        SCCCallSiteMap.find(Callee)->second.mergeWith(CS);
      } else {
        // Get the data structure graph for the called function.
        //
        DSGraph &GI = getDSGraph(*Callee);  // Graph to inline
        DEBUG(std::cerr << "    Inlining graph for " << Callee->getName()
              << "[" << GI.getGraphSize() << "+"
              << GI.getAuxFunctionCalls().size() << "] into: " << F.getName()
              << "[" << Graph.getGraphSize() << "+"
              << Graph.getAuxFunctionCalls().size() << "]\n");
        
        // Handle self recursion by resolving the arguments and return value
        Graph.mergeInGraph(CS, GI,
                           DSGraph::KeepModRefBits | DSGraph::StripAllocaBit |
                           DSGraph::DontCloneCallNodes);

        if (SCCFunctions.count(Callee))
          SCCCallSiteMap.insert(std::make_pair(Callee, CS));
      }
    }
    
    // Make sure to catch any leftover unresolvable calls...
    for (++LastCallSiteIdx; LastCallSiteIdx < TempFCs.size(); ++LastCallSiteIdx)
      UnresolvableCalls.push_back(TempFCs[LastCallSiteIdx]);
  }

  // Reset the SCCCallSiteMap...
  SCCCallSiteMap.clear();

  AuxCallsList.insert(AuxCallsList.end(), UnresolvableCalls.begin(),
                      UnresolvableCalls.end());
  UnresolvableCalls.clear();


  // Recompute the Incomplete markers.  If there are any function calls left
  // now that are complete, we must loop!
  Graph.maskIncompleteMarkers();
  Graph.markIncompleteNodes(DSGraph::MarkFormalArgs);

  // FIXME: materialize nodes from the globals graph as neccesary...

  Graph.removeDeadNodes(DSGraph::KeepUnreachableGlobals);

  DEBUG(std::cerr << "  [BU] Done inlining: " << F.getName() << " ["
        << Graph.getGraphSize() << "+" << Graph.getAuxFunctionCalls().size()
        << "]\n");
  //Graph.writeGraphToFile(std::cerr, "bu_" + F.getName());
  return Graph;
}
