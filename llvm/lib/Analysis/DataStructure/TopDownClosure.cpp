//===- TopDownClosure.cpp - Compute the top-down interprocedure closure ---===//
//
// This file implements the TDDataStructures class, which represents the
// Top-down Interprocedural closure of the data structure graph over the
// program.  This is useful (but not strictly necessary?) for applications
// like pointer analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DSGraph.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "Support/Statistic.h"

namespace {
  RegisterAnalysis<TDDataStructures>   // Register the pass
  Y("tddatastructure", "Top-down Data Structure Analysis Closure");
}

// run - Calculate the top down data structure graphs for each function in the
// program.
//
bool TDDataStructures::run(Module &M) {
  BUDataStructures &BU = getAnalysis<BUDataStructures>();
  GlobalsGraph = new DSGraph();

  // Calculate top-down from main...
  if (Function *F = M.getMainFunction())
    calculateGraph(*F);

  // Next calculate the graphs for each function unreachable function...
  for (Module::reverse_iterator I = M.rbegin(), E = M.rend(); I != E; ++I)
    if (!I->isExternal())
      calculateGraph(*I);

  GraphDone.clear();    // Free temporary memory...
  return false;
}

// releaseMemory - If the pass pipeline is done with this pass, we can release
// our memory... here...
//
// FIXME: This should be releaseMemory and will work fine, except that LoadVN
// has no way to extend the lifetime of the pass, which screws up ds-aa.
//
void TDDataStructures::releaseMyMemory() {
  for (hash_map<const Function*, DSGraph*>::iterator I = DSInfo.begin(),
         E = DSInfo.end(); I != E; ++I)
    delete I->second;

  // Empty map so next time memory is released, data structures are not
  // re-deleted.
  DSInfo.clear();
  delete GlobalsGraph;
  GlobalsGraph = 0;
}

/// ResolveCallSite - This method is used to link the actual arguments together
/// with the formal arguments for a function call in the top-down closure.  This
/// method assumes that the call site arguments have been mapped into nodes
/// local to the specified graph.
///
void TDDataStructures::ResolveCallSite(DSGraph &Graph,
                                       const DSCallSite &CallSite) {
  // Resolve all of the function formal arguments...
  Function &F = Graph.getFunction();
  Function::aiterator AI = F.abegin();

  for (unsigned i = 0, e = CallSite.getNumPtrArgs(); i != e; ++i, ++AI) {
    // Advance the argument iterator to the first pointer argument...
    while (!DS::isPointerType(AI->getType())) ++AI;
    
    // TD ...Merge the formal arg scalar with the actual arg node
    DSNodeHandle &NodeForFormal = Graph.getNodeForValue(AI);
    assert(NodeForFormal.getNode() && "Pointer argument has no dest node!");
    NodeForFormal.mergeWith(CallSite.getPtrArg(i));
  }
  
  // Merge returned node in the caller with the "return" node in callee
  if (CallSite.getRetVal().getNode() && Graph.getRetNode().getNode())
    Graph.getRetNode().mergeWith(CallSite.getRetVal());
}

DSGraph &TDDataStructures::getOrCreateDSGraph(Function &F) {
  DSGraph *&G = DSInfo[&F];
  if (G == 0) { // Not created yet?  Clone BU graph...
    G = new DSGraph(getAnalysis<BUDataStructures>().getDSGraph(F));
    G->getAuxFunctionCalls().clear();
    G->setPrintAuxCalls();
    G->setGlobalsGraph(GlobalsGraph);
  }
  return *G;
}

void TDDataStructures::calculateGraph(Function &F) {
  // Make sure this graph has not already been calculated, and that we don't get
  // into an infinite loop with mutually recursive functions.
  //
  if (GraphDone.count(&F)) return;
  GraphDone.insert(&F);

  // Get the current functions graph...
  DSGraph &Graph = getOrCreateDSGraph(F);

  const std::vector<DSCallSite> &CallSites = Graph.getFunctionCalls();
  if (CallSites.empty()) {
    DEBUG(std::cerr << "  [TD] No callees for: " << F.getName() << "\n");
    return;  // If no call sites, there is nothing more to do here
  }

  // Loop over all of the call sites, building a multi-map from Callees to
  // DSCallSite*'s.  With this map we can then loop over each callee, cloning
  // this graph once into it, then resolving arguments.
  //
  std::multimap<Function*, const DSCallSite*> CalleeSites;
  for (unsigned i = 0, e = CallSites.size(); i != e; ++i) {
    const DSCallSite &CS = CallSites[i];
    const std::vector<GlobalValue*> Callees =
      CS.getCallee().getNode()->getGlobals();

    // Loop over all of the functions that this call may invoke...
    for (unsigned c = 0, e = Callees.size(); c != e; ++c)
      if (Function *F = dyn_cast<Function>(Callees[c]))  // If this is a fn...
        if (!F->isExternal())                            // If it's not external
          CalleeSites.insert(std::make_pair(F, &CS));    // Keep track of it!
  }

  // Now that we have information about all of the callees, propogate the
  // current graph into the callees.
  //
  DEBUG(std::cerr << "  [TD] Inlining '" << F.getName() << "' into "
                  << CalleeSites.size() << " callees.\n");

  // Loop over all the callees...
  for (std::multimap<Function*, const DSCallSite*>::iterator
         I = CalleeSites.begin(), E = CalleeSites.end(); I != E; )
    if (I->first == &F) {  // Bottom-up pass takes care of self loops!
      ++I;
    } else {
      // For each callee...
      Function *Callee = I->first;
      DSGraph &CG = getOrCreateDSGraph(*Callee);  // Get the callee's graph...
      
      DEBUG(std::cerr << "\t [TD] Inlining into callee '" << Callee->getName()
            << "'\n");
      
      // Clone our current graph into the callee...
      hash_map<Value*, DSNodeHandle> OldValMap;
      hash_map<const DSNode*, DSNodeHandle> OldNodeMap;
      CG.cloneInto(Graph, OldValMap, OldNodeMap,
                   DSGraph::StripModRefBits |
                   DSGraph::KeepAllocaBit | DSGraph::DontCloneCallNodes);
      OldValMap.clear();  // We don't care about the ValMap

      // Loop over all of the invocation sites of the callee, resolving
      // arguments to our graph.  This loop may iterate multiple times if the
      // current function calls this callee multiple times with different
      // signatures.
      //
      for (; I != E && I->first == Callee; ++I) {
        // Map call site into callee graph
        DSCallSite NewCS(*I->second, OldNodeMap);
        
        // Resolve the return values...
        NewCS.getRetVal().mergeWith(CG.getRetNode());
        
        // Resolve all of the arguments...
        Function::aiterator AI = Callee->abegin();
        for (unsigned i = 0, e = NewCS.getNumPtrArgs();
             i != e && AI != Callee->aend(); ++i, ++AI) {
          // Advance the argument iterator to the first pointer argument...
          while (!DS::isPointerType(AI->getType())) {
            ++AI;
#ifndef NDEBUG
            if (AI == Callee->aend())
              std::cerr << "Bad call to Function: " << Callee->getName()<< "\n";
#endif
            assert(AI != Callee->aend() &&
                   "# Args provided is not # Args required!");
          }
          
          // Add the link from the argument scalar to the provided value
          DSNodeHandle &NH = CG.getNodeForValue(AI);
          assert(NH.getNode() && "Pointer argument without scalarmap entry?");
          NH.mergeWith(NewCS.getPtrArg(i));
        }
      }

      // Done with the nodemap...
      OldNodeMap.clear();

      // Recompute the Incomplete markers and eliminate unreachable nodes.
      CG.maskIncompleteMarkers();
      CG.markIncompleteNodes(F.hasInternalLinkage() ? DSGraph::IgnoreFormalArgs:
                             DSGraph::MarkFormalArgs
                             /*&& FIXME: NEED TO CHECK IF ALL CALLERS FOUND!*/);
      CG.removeDeadNodes(DSGraph::RemoveUnreachableGlobals);
    }

  DEBUG(std::cerr << "  [TD] Done inlining into callees for: " << F.getName()
        << " [" << Graph.getGraphSize() << "+"
        << Graph.getFunctionCalls().size() << "]\n");

  
  // Loop over all the callees... making sure they are all resolved now...
  Function *LastFunc = 0;
  for (std::multimap<Function*, const DSCallSite*>::iterator
         I = CalleeSites.begin(), E = CalleeSites.end(); I != E; ++I)
    if (I->first != LastFunc) {  // Only visit each callee once...
      LastFunc = I->first;
      calculateGraph(*I->first);
    }
}

