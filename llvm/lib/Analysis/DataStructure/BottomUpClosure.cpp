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
using std::map;

static RegisterAnalysis<BUDataStructures>
X("budatastructure", "Bottom-up Data Structure Analysis Closure");

using namespace DS;

// run - Calculate the bottom up data structure graphs for each function in the
// program.
//
bool BUDataStructures::run(Module &M) {
  GlobalsGraph = new DSGraph();

  // Simply calculate the graphs for each function...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal())
      calculateGraph(*I, 0);
  return false;
}

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
  delete GlobalsGraph;
  GlobalsGraph = 0;
}


// Return true if a graph was inlined
// Can not modify the part of the AuxCallList < FirstResolvableCall.
//
bool BUDataStructures::ResolveFunctionCalls(DSGraph &G,
                                            unsigned &FirstResolvableCall,
                                   std::map<Function*, DSCallSite> &InProcess,
                                            unsigned Indent) {
  std::vector<DSCallSite> &FCs = G.getAuxFunctionCalls();
  bool Changed = false;

  // Loop while there are call sites that we can resolve!
  while (FirstResolvableCall != FCs.size()) {
    DSCallSite Call = FCs[FirstResolvableCall];

    // If the function list is incomplete...
    if (Call.getCallee().getNode()->NodeType & DSNode::Incomplete) {
      // If incomplete, we cannot resolve it, so leave it at the beginning of
      // the call list with the other unresolvable calls...
      ++FirstResolvableCall;
    } else {
      // Start inlining all of the functions we can... some may not be
      // inlinable if they are external...
      //
      const std::vector<GlobalValue*> &Callees =
        Call.getCallee().getNode()->getGlobals();

      bool hasExternalTarget = false;
      
      // Loop over the functions, inlining whatever we can...
      for (unsigned c = 0, e = Callees.size(); c != e; ++c) {
        // Must be a function type, so this cast should succeed unless something
        // really wierd is happening.
        Function &FI = cast<Function>(*Callees[c]);

        if (FI.getName() == "printf" || FI.getName() == "sscanf" ||
            FI.getName() == "fprintf" || FI.getName() == "open" ||
            FI.getName() == "sprintf" || FI.getName() == "fputs") {
          // Ignore
        } else if (FI.isExternal()) {
          // If the function is external, then we cannot resolve this call site!
          hasExternalTarget = true;
          break;
        } else {
          std::map<Function*, DSCallSite>::iterator I =
            InProcess.lower_bound(&FI);

          if (I != InProcess.end() && I->first == &FI) {  // Recursion detected?
            // Merge two call sites to eliminate recursion...
            Call.mergeWith(I->second);

            DEBUG(std::cerr << std::string(Indent*2, ' ')
                  << "* Recursion detected for function " << FI.getName()<<"\n");
          } else {
            DEBUG(std::cerr << std::string(Indent*2, ' ')
                  << "Inlining: " << FI.getName() << "\n");
            
            // Get the data structure graph for the called function, closing it
            // if possible...
            //
            DSGraph &GI = calculateGraph(FI, Indent+1);  // Graph to inline

            DEBUG(std::cerr << std::string(Indent*2, ' ')
                  << "Got graph for: " << FI.getName() << "["
                  << GI.getGraphSize() << "+"
                  << GI.getAuxFunctionCalls().size() << "] "
                  << " in: " << G.getFunction().getName() << "["
                  << G.getGraphSize() << "+"
                  << G.getAuxFunctionCalls().size() << "]\n");

            // Keep track of how many call sites are added by the inlining...
            unsigned NumCalls = FCs.size();

            // Resolve the arguments and return value
            G.mergeInGraph(Call, GI, DSGraph::StripAllocaBit |
                           DSGraph::DontCloneCallNodes);

            // Added a call site?
            if (FCs.size() != NumCalls) {
              // Otherwise we need to inline the graph.  Temporarily add the
              // current function to the InProcess map to be able to handle
              // recursion successfully.
              //
              I = InProcess.insert(I, std::make_pair(&FI, Call));

              // ResolveFunctionCalls - Resolve the function calls that just got
              // inlined...
              //
              Changed |= ResolveFunctionCalls(G, NumCalls, InProcess, Indent+1);
              
              // Now that we are done processing the inlined graph, remove our
              // cycle detector record...
              //
              //InProcess.erase(I);
            }
          }
        }
      }

      if (hasExternalTarget) {
        // If we cannot resolve this call site...
        ++FirstResolvableCall;
      } else {
        Changed = true;
        FCs.erase(FCs.begin()+FirstResolvableCall);
      }
    }
  }

  return Changed;
}

DSGraph &BUDataStructures::calculateGraph(Function &F, unsigned Indent) {
  // Make sure this graph has not already been calculated, or that we don't get
  // into an infinite loop with mutually recursive functions.
  //
  DSGraph *&GraphPtr = DSInfo[&F];
  if (GraphPtr) return *GraphPtr;

  // Copy the local version into DSInfo...
  GraphPtr = new DSGraph(getAnalysis<LocalDataStructures>().getDSGraph(F));
  DSGraph &Graph = *GraphPtr;

  Graph.setGlobalsGraph(GlobalsGraph);
  Graph.setPrintAuxCalls();

  // Start resolving calls...
  std::vector<DSCallSite> &FCs = Graph.getAuxFunctionCalls();

  // Start with a copy of the original call sites...
  FCs = Graph.getFunctionCalls();

  DEBUG(std::cerr << std::string(Indent*2, ' ')
                  << "[BU] Calculating graph for: " << F.getName() << "\n");

  bool Changed;
  while (1) {
    unsigned FirstResolvableCall = 0;
    std::map<Function *, DSCallSite> InProcess;

    // Insert a call site for self to handle self recursion...
    std::vector<DSNodeHandle> Args;
    Args.reserve(F.asize());
    for (Function::aiterator I = F.abegin(), E = F.aend(); I != E; ++I)
      if (isPointerType(I->getType()))
        Args.push_back(Graph.getNodeForValue(I));

    InProcess.insert(std::make_pair(&F, 
           DSCallSite(*(CallInst*)0, Graph.getRetNode(),(DSNode*)0,Args)));

    Changed = ResolveFunctionCalls(Graph, FirstResolvableCall, InProcess,
                                   Indent);

    if (Changed) {
      Graph.maskIncompleteMarkers();
      Graph.markIncompleteNodes();
      Graph.removeDeadNodes();
      break;
    } else {
      break;
    }
  }

#if 0  
  bool Inlined;
  do {
    Inlined = false;

    for (unsigned i = 0; i != FCs.size(); ++i) {
      // Copy the call, because inlining graphs may invalidate the FCs vector.
      DSCallSite Call = FCs[i];

      // If the function list is complete...
      if ((Call.getCallee().getNode()->NodeType & DSNode::Incomplete)==0) {
        // Start inlining all of the functions we can... some may not be
        // inlinable if they are external...
        //
        std::vector<GlobalValue*> Callees =
          Call.getCallee().getNode()->getGlobals();

        unsigned OldNumCalls = FCs.size();

        // Loop over the functions, inlining whatever we can...
        for (unsigned c = 0; c != Callees.size(); ++c) {
          // Must be a function type, so this cast MUST succeed.
          Function &FI = cast<Function>(*Callees[c]);

          if (&FI == &F) {
            // Self recursion... simply link up the formal arguments with the
            // actual arguments...
            DEBUG(std::cerr << std::string(Indent*2, ' ')
                  << "[BU] Self Inlining: " << F.getName() << "\n");

            // Handle self recursion by resolving the arguments and return value
            Graph.mergeInGraph(Call, Graph, DSGraph::StripAllocaBit);

            // Erase the entry in the callees vector
            Callees.erase(Callees.begin()+c--);

          } else if (!FI.isExternal()) {
            DEBUG(std::cerr << std::string(Indent*2, ' ')
                  << "[BU] In " << F.getName() << " inlining: "
                  << FI.getName() << "\n");
            
            // Get the data structure graph for the called function, closing it
            // if possible (which is only impossible in the case of mutual
            // recursion...
            //
            DSGraph &GI = calculateGraph(FI, Indent+1);  // Graph to inline

            DEBUG(std::cerr << std::string(Indent*2, ' ')
                  << "[BU] Got graph for " << FI.getName()
                  << " in: " << F.getName() << "[" << GI.getGraphSize() << "+"
                  << GI.getAuxFunctionCalls().size() << "]\n");

            // Handle self recursion by resolving the arguments and return value
            Graph.mergeInGraph(Call, GI, DSGraph::StripAllocaBit |
                                DSGraph::DontCloneCallNodes);

            // Erase the entry in the Callees vector
            Callees.erase(Callees.begin()+c--);

          } else if (FI.getName() == "printf" || FI.getName() == "sscanf" ||
                     FI.getName() == "fprintf" || FI.getName() == "open" ||
                     FI.getName() == "sprintf" || FI.getName() == "fputs") {
            // FIXME: These special cases (eg printf) should go away when we can
            // define functions that take a variable number of arguments.

            // FIXME: at the very least, this should update mod/ref info
            // Erase the entry in the globals vector
            Callees.erase(Callees.begin()+c--);
          }
        }

        if (Callees.empty()) {         // Inlined all of the function calls?
          // Erase the call if it is resolvable...
          FCs.erase(FCs.begin()+i--);  // Don't skip a the next call...
          Inlined = true;
        } else if (Callees.size() !=
                   Call.getCallee().getNode()->getGlobals().size()) {
          // Was able to inline SOME, but not all of the functions.  Construct a
          // new global node here.
          //
          assert(0 && "Unimpl!");
          Inlined = true;
        }


#if 0
        // If we just inlined a function that had call nodes, chances are that
        // the call nodes are redundant with ones we already have.  Eliminate
        // those call nodes now.
        //
        if (FCs.size() >= OldNumCalls)
          Graph.removeTriviallyDeadNodes();
#endif
      }

      if (FCs.size() > 200) {
        std::cerr << "Aborted inlining fn: '" << F.getName() << "'!"
                  << std::endl;
        Graph.maskIncompleteMarkers();
        Graph.markIncompleteNodes();
        Graph.removeDeadNodes();
        Graph.writeGraphToFile(std::cerr, "crap."+F.getName());
        exit(1);
        return Graph;
      }

    }

    // Recompute the Incomplete markers.  If there are any function calls left
    // now that are complete, we must loop!
    if (Inlined) {
      Graph.maskIncompleteMarkers();
      Graph.markIncompleteNodes();
      Graph.removeDeadNodes();
    }
    
  } while (Inlined && !FCs.empty());
#endif

  DEBUG(std::cerr << std::string(Indent*2, ' ')
        << "[BU] Done inlining: " << F.getName() << " ["
        << Graph.getGraphSize() << "+" << Graph.getAuxFunctionCalls().size()
        << "]\n");

  Graph.writeGraphToFile(std::cerr, "bu_" + F.getName());

  return Graph;
}
