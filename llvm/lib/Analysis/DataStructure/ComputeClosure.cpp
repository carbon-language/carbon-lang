//===- ComputeClosure.cpp - Implement interprocedural closing of graphs ---===//
//
// Compute the interprocedural closure of a data structure graph
//
//===----------------------------------------------------------------------===//

// DEBUG_IP_CLOSURE - Define this to debug the act of linking up graphs
//#define DEBUG_IP_CLOSURE 1

#include "llvm/Analysis/DataStructure.h"
#include "llvm/iOther.h"
#include "Support/STLExtras.h"
#include <algorithm>
#ifdef DEBUG_IP_CLOSURE
#include "llvm/Assembly/Writer.h"
#endif

// Make all of the pointers that point to Val also point to N.
//
static void copyEdgesFromTo(PointerVal Val, DSNode *N) {
  assert(Val.Index == 0 && "copyEdgesFromTo:index != 0 TODO");

  const vector<PointerValSet*> &PVSToUpdate(Val.Node->getReferrers());
  for (unsigned i = 0, e = PVSToUpdate.size(); i != e; ++i)
    PVSToUpdate[i]->add(N);  // TODO: support index
}

static void ResolveNodesTo(const PointerVal &FromPtr,
                           const PointerValSet &ToVals) {
  assert(FromPtr.Index == 0 &&
         "Resolved node return pointer should be index 0!");
  assert(isa<ShadowDSNode>(FromPtr.Node) &&
         "Resolved node should be a shadow!");
  ShadowDSNode *Shadow = cast<ShadowDSNode>(FromPtr.Node);
  assert(Shadow->isCriticalNode() && "Shadow node should be a critical node!");
  Shadow->resetCriticalMark();

  // Make everything that pointed to the shadow node also point to the values in
  // ToVals...
  //
  for (unsigned i = 0, e = ToVals.size(); i != e; ++i)
    copyEdgesFromTo(ToVals[i], Shadow);

  // Make everything that pointed to the shadow node now also point to the
  // values it is equivalent to...
  const vector<PointerValSet*> &PVSToUpdate(Shadow->getReferrers());
  for (unsigned i = 0, e = PVSToUpdate.size(); i != e; ++i)
    PVSToUpdate[i]->add(ToVals);
}


// ResolveNodeTo - The specified node is now known to point to the set of values
// in ToVals, instead of the old shadow node subgraph that it was pointing to.
//
static void ResolveNodeTo(DSNode *Node, const PointerValSet &ToVals) {
  assert(Node->getNumLinks() == 1 && "Resolved node can only be a scalar!!");

  const PointerValSet &PVS = Node->getLink(0);

  // Only resolve the first pointer, although there many be many pointers here.
  // The problem is that the inlined function might return one of the arguments
  // to the function, and if so, extra values can be added to the arg or call
  // node that point to what the other one got resolved to.  Since these will
  // be added to the end of the PVS pointed in, we just ignore them.
  //
  ResolveNodesTo(PVS[0], ToVals);
}

// isResolvableCallNode - Return true if node is a call node and it is a call
// node that we can inline...
//
static bool isResolvableCallNode(CallDSNode *CN) {
  // Only operate on call nodes with direct method calls
  Function *F = CN->getCall()->getCalledFunction();
  if (F == 0) return false;

  // Only work on call nodes with direct calls to methods with bodies.
  return !F->isExternal();
}


// computeClosure - Replace all of the resolvable call nodes with the contents
// of their corresponding method data structure graph...
//
void FunctionDSGraph::computeClosure(const DataStructure &DS) {
  typedef pair<vector<PointerValSet>, CallInst *> CallDescriptor;
  map<CallDescriptor, PointerValSet> CallMap;

  unsigned NumInlines = 0;

  // Loop over the resolvable call nodes...
  vector<CallDSNode*>::iterator NI;
  NI = std::find_if(CallNodes.begin(), CallNodes.end(), isResolvableCallNode);
  while (NI != CallNodes.end()) {
    CallDSNode *CN = *NI;
    Function *F = CN->getCall()->getCalledFunction();

    if (NumInlines++ == 30) {      // CUTE hack huh?
      cerr << "Infinite (?) recursion halted\n";
      return;
    }

    CallNodes.erase(NI);                 // Remove the call node from the graph

    unsigned CallNodeOffset = NI-CallNodes.begin();

    // Find out if we have already incorporated this node... if so, it will be
    // in the CallMap...
    //
    CallDescriptor FDesc(CN->getArgs(), CN->getCall());
    map<CallDescriptor, PointerValSet>::iterator CMI = CallMap.find(FDesc);

    // Hold the set of values that correspond to the incorporated methods
    // return set.
    //
    PointerValSet RetVals;

    if (CMI != CallMap.end()) {
      // We have already inlined an identical function call!
      RetVals = CMI->second;
    } else {
      // Get the datastructure graph for the new method.  Note that we are not
      // allowed to modify this graph because it will be the cached graph that
      // is returned by other users that want the local datastructure graph for
      // a method.
      //
      const FunctionDSGraph &NewFunction = DS.getDSGraph(F);

      // StartNode - The first node of the incorporated graph, last node of the
      // preexisting data structure graph...
      //
      unsigned StartArgNode   = ArgNodes.size();
      unsigned StartAllocNode = AllocNodes.size();

      // Incorporate a copy of the called function graph into the current graph,
      // allowing us to do local transformations to local graph to link
      // arguments to call values, and call node to return value...
      //
      RetVals = cloneFunctionIntoSelf(NewFunction, F == Func);
      CallMap[FDesc] = RetVals;

      // If the call node has arguments, process them now!
      if (CN->getNumArgs()) {
        // The ArgNodes of the incorporated graph should be the nodes starting
        // at StartNode, ordered the same way as the call arguments.  The arg
        // nodes are seperated by a single shadow node, but that shadow node
        // might get eliminated in the process of optimization.
        //
        for (unsigned i = 0, e = CN->getNumArgs(); i != e; ++i) {
          // Get the arg node of the incorporated method...
          ArgDSNode *ArgNode = ArgNodes[StartArgNode];
          
          // Now we make all of the nodes inside of the incorporated method
          // point to the real arguments values, not to the shadow nodes for the
          // argument.
          //
          ResolveNodeTo(ArgNode, CN->getArgValues(i));
          
          // Remove the argnode from the set of nodes in this method...
          ArgNodes.erase(ArgNodes.begin()+StartArgNode);
            
          // ArgNode is no longer useful, delete now!
          delete ArgNode;
        }
      }

      // Loop through the nodes, deleting alloca nodes in the inlined function.
      // Since the memory has been released, we cannot access their pointer
      // fields (with defined results at least), so it is not possible to use
      // any pointers to the alloca.  Drop them now, and remove the alloca's
      // since they are dead (we just removed all links to them).
      //
      for (unsigned i = StartAllocNode; i != AllocNodes.size(); ++i)
        if (AllocNodes[i]->isAllocaNode()) {
          AllocDSNode *NDS = AllocNodes[i];
          NDS->removeAllIncomingEdges();          // These edges are invalid now
          delete NDS;                             // Node is dead
          AllocNodes.erase(AllocNodes.begin()+i); // Remove slot in Nodes array
          --i;                                    // Don't skip the next node
        }
    }

    // If the function returns a pointer value...  Resolve values pointing to
    // the shadow nodes pointed to by CN to now point the values in RetVals...
    //
    if (CN->getNumLinks()) ResolveNodeTo(CN, RetVals);

    // Now the call node is completely destructable.  Eliminate it now.
    delete CN;

    bool Changed = true;
    while (Changed) {
      // Eliminate shadow nodes that are not distinguishable from some other
      // node in the graph...
      //
      Changed = UnlinkUndistinguishableShadowNodes();

      // Eliminate shadow nodes that are now extraneous due to linking...
      Changed |= RemoveUnreachableShadowNodes();
    }

    //if (F == Func) return;  // Only do one self inlining
    
    // Move on to the next call node...
    NI = std::find_if(CallNodes.begin(), CallNodes.end(), isResolvableCallNode);
  }
}
