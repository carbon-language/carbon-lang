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
using std::map;

AnalysisID BUDataStructures::ID(AnalysisID::create<BUDataStructures>());

// releaseMemory - If the pass pipeline is done with this pass, we can release
// our memory... here...
//
void BUDataStructures::releaseMemory() {
  for (map<Function*, DSGraph*>::iterator I = DSInfo.begin(),
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

// MergeGlobalNodes - Merge global value nodes in the inlined graph with the
// global value nodes in the current graph if there are duplicates.
//
static void MergeGlobalNodes(map<Value*, DSNodeHandle> &ValMap,
                             map<Value*, DSNodeHandle> &OldValMap) {
  // Loop over all of the nodes inlined, if any of them are global variable
  // nodes, we must make sure they get properly added or merged with the ValMap.
  //
  for (map<Value*, DSNodeHandle>::iterator I = OldValMap.begin(),
         E = OldValMap.end(); I != E; ++I)
    if (isa<GlobalValue>(I->first)) {
      DSNodeHandle &NH = ValMap[I->first];  // Look up global in ValMap.
      if (NH == 0) {        // No entry for the global yet?
        NH = I->second;     // Add the one just inlined...
      } else {
        NH->mergeWith(I->second); // Merge the two globals together.
      }
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

  // Save a copy of the original call nodes for the top-down pass
  Graph->saveOrigFunctionCalls();
  
  // Start resolving calls...
  std::vector<std::vector<DSNodeHandle> > &FCs = Graph->getFunctionCalls();

  DEBUG(cerr << "Inlining: " << F.getName() << "\n");

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
        std::vector<GlobalValue*> Globals(Call[1]->getGlobals());

        // Loop over the functions, inlining whatever we can...
        for (unsigned g = 0; g != Globals.size(); ++g) {
          // Must be a function type, so this cast MUST succeed.
          Function &FI = cast<Function>(*Globals[g]);
          if (&FI == &F) {
            // Self recursion... simply link up the formal arguments with the
            // actual arguments...
           
            DEBUG(cerr << "Self Inlining: " << F.getName() << "\n");

            if (Call[0]) // Handle the return value if present...
              Graph->RetNode->mergeWith(Call[0]);

            // Resolve the arguments in the call to the actual values...
            ResolveArguments(Call, F, Graph->getValueMap());

            // Erase the entry in the globals vector
            Globals.erase(Globals.begin()+g--);
          } else if (!FI.isExternal()) {
            DEBUG(std::cerr << "In " << F.getName() << " inlining: "
                  << FI.getName() << "\n");
            
            // Get the data structure graph for the called function, closing it
            // if possible (which is only impossible in the case of mutual
            // recursion...
            //
            DSGraph &GI = calculateGraph(FI);  // Graph to inline

            DEBUG(cerr << "Got graph for " << FI.getName() << " in: "
                  << F.getName() << "\n");

            // Remember the callers for each callee for use in the top-down
            // pass so we don't have to compute this again
            GI.addCaller(F);

            // Clone the callee's graph into the current graph, keeping
            // track of where scalars in the old graph _used_ to point
            // and of the new nodes matching nodes of the old graph ...
            std::map<Value*, DSNodeHandle> OldValMap;
            std::map<const DSNode*, DSNode*> OldNodeMap; // unused

            // The clone call may invalidate any of the vectors in the data
            // structure graph.
            DSNode *RetVal = Graph->cloneInto(GI, OldValMap, OldNodeMap);

            ResolveArguments(Call, FI, OldValMap);

            if (Call[0])  // Handle the return value if present
              RetVal->mergeWith(Call[0]);
            
            // Merge global value nodes in the inlined graph with the global
            // value nodes in the current graph if there are duplicates.
            //
            MergeGlobalNodes(Graph->getValueMap(), OldValMap);

            // Erase the entry in the globals vector
            Globals.erase(Globals.begin()+g--);
          }
        }

        if (Globals.empty()) {         // Inlined all of the function calls?
          // Erase the call if it is resolvable...
          FCs.erase(FCs.begin()+i--);  // Don't skip a the next call...
          Inlined = true;
        } else if (Globals.size() != Call[1]->getGlobals().size()) {
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
      Graph->removeDeadNodes();
    }
  } while (Inlined && !FCs.empty());

  return *Graph;
}
