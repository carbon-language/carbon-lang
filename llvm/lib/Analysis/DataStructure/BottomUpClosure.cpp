//===- BottomUpClosure.cpp - Compute the bottom up interprocedure closure -===//
//
// This file implements the BUDataStructures class, which represents the
// Bottom-Up Interprocedural closure of the data structure graph over the
// program.  This is useful for applications like pool allocation, but **not**
// applications like pointer analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "Support/StatisticReporter.h"
#include <set>
using std::map;

static RegisterAnalysis<BUDataStructures>
X("budatastructure", "Bottom-up Data Structure Analysis Closure");
AnalysisID BUDataStructures::ID = X;

// releaseMemory - If the pass pipeline is done with this pass, we can release
// our memory... here...
//
void BUDataStructures::releaseMemory() {
  for (map<const Function*, DSGraph*>::iterator I = DSInfo.begin(),
         E = DSInfo.end(); I != E; ++I)
    delete I->second;

  // Empty map so next time memory is released, data structures are not
  // re-deleted.
  DSInfo.clear();
}

// run - Calculate the bottom up data structure graphs for each function in the
// program.
//
bool BUDataStructures::run(Module &M) {
  // Simply calculate the graphs for each function...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal())
      calculateGraph(*I);
  return false;
}


// ResolveArguments - Resolve the formal and actual arguments for a function
// call.
//
static void ResolveArguments(std::vector<DSNodeHandle> &Call, Function &F,
                             map<Value*, DSNodeHandle> &ValueMap) {
  // Resolve all of the function arguments...
  Function::aiterator AI = F.abegin();
  for (unsigned i = 2, e = Call.size(); i != e; ++i) {
    // Advance the argument iterator to the first pointer argument...
    while (!isa<PointerType>(AI->getType())) ++AI;
    
    // Add the link from the argument scalar to the provided value
    DSNode *NN = ValueMap[AI];
    NN->addEdgeTo(Call[i]);
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

DSGraph &BUDataStructures::calculateGraph(Function &F) {
  // Make sure this graph has not already been calculated, or that we don't get
  // into an infinite loop with mutually recursive functions.
  //
  DSGraph *&Graph = DSInfo[&F];
  if (Graph) return *Graph;

  // Copy the local version into DSInfo...
  Graph = new DSGraph(getAnalysis<LocalDataStructures>().getDSGraph(F));

  // Populate the GlobalsGraph with globals from this one.
  Graph->GlobalsGraph->cloneGlobals(*Graph, /*cloneCalls*/ false);

  // Save a copy of the original call nodes for the top-down pass
  Graph->saveOrigFunctionCalls();

  // Start resolving calls...
  std::vector<std::vector<DSNodeHandle> > &FCs = Graph->getFunctionCalls();

  DEBUG(std::cerr << "  [BU] Inlining: " << F.getName() << "\n");

  // Add F to the PendingCallers list of each direct callee for use in the
  // top-down pass so we don't have to compute this again.  We don't want
  // to do it for indirect callees inlined later, so remember which calls
  // are in the original FCs set.
  std::set<const DSNode*> directCallees;
  for (unsigned i = 0; i < FCs.size(); ++i)
    directCallees.insert(FCs[i][1]); // ptr to function node

  bool Inlined;
  do {
    Inlined = false;

    for (unsigned i = 0; i != FCs.size(); ++i) {
      // Copy the call, because inlining graphs may invalidate the FCs vector.
      std::vector<DSNodeHandle> Call = FCs[i];

      // If the function list is not incomplete...
      if ((Call[1]->NodeType & DSNode::Incomplete) == 0) {
        // Start inlining all of the functions we can... some may not be
        // inlinable if they are external...
        //
        std::vector<GlobalValue*> Callees(Call[1]->getGlobals());

        // Loop over the functions, inlining whatever we can...
        for (unsigned c = 0; c != Callees.size(); ++c) {
          // Must be a function type, so this cast MUST succeed.
          Function &FI = cast<Function>(*Callees[c]);
          if (&FI == &F) {
            // Self recursion... simply link up the formal arguments with the
            // actual arguments...

            DEBUG(std::cerr << "\t[BU] Self Inlining: " << F.getName() << "\n");

            if (Call[0]) // Handle the return value if present...
              Graph->RetNode->mergeWith(Call[0]);

            // Resolve the arguments in the call to the actual values...
            ResolveArguments(Call, F, Graph->getValueMap());

            // Erase the entry in the callees vector
            Callees.erase(Callees.begin()+c--);
          } else if (!FI.isExternal()) {
            DEBUG(std::cerr << "\t[BU] In " << F.getName() << " inlining: "
                  << FI.getName() << "\n");
            
            // Get the data structure graph for the called function, closing it
            // if possible (which is only impossible in the case of mutual
            // recursion...
            //
            DSGraph &GI = calculateGraph(FI);  // Graph to inline

            DEBUG(std::cerr << "\t\t[BU] Got graph for " << FI.getName()
                  << " in: " << F.getName() << "\n");

            // Clone the callee's graph into the current graph, keeping
            // track of where scalars in the old graph _used_ to point,
            // and of the new nodes matching nodes of the old graph.
            std::map<Value*, DSNodeHandle> OldValMap;
            std::map<const DSNode*, DSNode*> OldNodeMap;

            // The clone call may invalidate any of the vectors in the data
            // structure graph.  Strip locals and don't copy the list of callers
            DSNode *RetVal = Graph->cloneInto(GI, OldValMap, OldNodeMap,
                                              /*StripScalars*/   true,
                                              /*StripAllocas*/   true,
                                              /*CopyCallers*/    false,
                                              /*CopyOrigCalls*/  false);

            ResolveArguments(Call, FI, OldValMap);

            if (Call[0])  // Handle the return value if present
              RetVal->mergeWith(Call[0]);

            // Merge global value nodes in the inlined graph with the global
            // value nodes in the current graph if there are duplicates.
            //
            MergeGlobalNodes(*Graph, OldValMap);

            // If this was an original call, add F to the PendingCallers list
            if (directCallees.find(Call[1]) != directCallees.end())
              GI.addCaller(F);

            // Erase the entry in the Callees vector
            Callees.erase(Callees.begin()+c--);

          } else if (FI.getName() == "printf" || FI.getName() == "sscanf" ||
                     FI.getName() == "fprintf" || FI.getName() == "open" ||
                     FI.getName() == "sprintf") {

            // Erase the entry in the globals vector
            Callees.erase(Callees.begin()+c--);
          }
        }

        if (Callees.empty()) {         // Inlined all of the function calls?
          // Erase the call if it is resolvable...
          FCs.erase(FCs.begin()+i--);  // Don't skip a the next call...
          Inlined = true;
        } else if (Callees.size() != Call[1]->getGlobals().size()) {
          // Was able to inline SOME, but not all of the functions.  Construct a
          // new global node here.
          //
          assert(0 && "Unimpl!");
          Inlined = true;
        }
      }
    }

    // Recompute the Incomplete markers.  If there are any function calls left
    // now that are complete, we must loop!
    if (Inlined) {
      Graph->maskIncompleteMarkers();
      Graph->markIncompleteNodes();
      Graph->removeDeadNodes(/*KeepAllGlobals*/ false, /*KeepCalls*/ true);
    }
  } while (Inlined && !FCs.empty());

  // Copy any unresolved call nodes into the Globals graph and
  // filter out unresolved call nodes inlined from the callee.
  if (!FCs.empty())
    Graph->GlobalsGraph->cloneCalls(*Graph);

  Graph->maskIncompleteMarkers();
  Graph->markIncompleteNodes();
  Graph->removeDeadNodes(/*KeepAllGlobals*/ false, /*KeepCalls*/ false);

  DEBUG(std::cerr << "  [BU] Done inlining: " << F.getName() << " ["
        << Graph->getGraphSize() << "+" << Graph->getFunctionCalls().size()
        << "]\n");

  return *Graph;
}
