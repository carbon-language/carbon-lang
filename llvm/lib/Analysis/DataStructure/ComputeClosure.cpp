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
  unsigned ValIdx = Val.Index;
  unsigned NLinks = N->getNumLinks();

  const vector<PointerValSet*> &PVSsToUpdate(Val.Node->getReferrers());
  for (unsigned i = 0, e = PVSsToUpdate.size(); i != e; ++i) {
    // Loop over all of the pointers pointing to Val...
    PointerValSet &PVS = *PVSsToUpdate[i];
    for (unsigned j = 0, je = PVS.size(); j != je; ++j) {
      if (PVS[j].Node == Val.Node && PVS[j].Index >= ValIdx && 
          PVS[j].Index < ValIdx+NLinks)
        PVS.add(PointerVal(N, PVS[j].Index-ValIdx));
    }
  }
}

static void ResolveNodesTo(const PointerVal &FromPtr,
                           const PointerValSet &ToVals) {
  assert(FromPtr.Index == 0 &&
         "Resolved node return pointer should be index 0!");
  DSNode *N = FromPtr.Node;

  // Make everything that pointed to the shadow node also point to the values in
  // ToVals...
  //
  for (unsigned i = 0, e = ToVals.size(); i != e; ++i)
    copyEdgesFromTo(ToVals[i], N);

  // Make everything that pointed to the shadow node now also point to the
  // values it is equivalent to...
  const vector<PointerValSet*> &PVSToUpdate(N->getReferrers());
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
  // Note that this cannot be a real vector because the keys will be changing
  // as nodes are eliminated!
  //
  typedef pair<vector<PointerValSet>, CallInst *> CallDescriptor;
  vector<pair<CallDescriptor, PointerValSet> > CallMap;

  unsigned NumInlines = 0;

  // Loop over the resolvable call nodes...
  vector<CallDSNode*>::iterator NI;
  NI = std::find_if(CallNodes.begin(), CallNodes.end(), isResolvableCallNode);
  while (NI != CallNodes.end()) {
    CallDSNode *CN = *NI;
    Function *F = CN->getCall()->getCalledFunction();

    if (NumInlines++ == 100) {      // CUTE hack huh?
      cerr << "Infinite (?) recursion halted\n";
      return;
    }

    CallNodes.erase(NI);                 // Remove the call node from the graph

    unsigned CallNodeOffset = NI-CallNodes.begin();

    // Find out if we have already incorporated this node... if so, it will be
    // in the CallMap...
    //
    
#if 0
    cerr << "\nSearching for: " << (void*)CN->getCall() << ": ";
    for (unsigned X = 0; X != CN->getArgs().size(); ++X) {
      cerr << " " << X << " is\n";
      CN->getArgs().first[X].print(cerr);
    }
#endif

    const vector<PointerValSet> &Args = CN->getArgs();
    PointerValSet *CMI = 0;
    for (unsigned i = 0, e = CallMap.size(); i != e; ++i) {
#if 0
      cerr << "Found: " << (void*)CallMap[i].first.second << ": ";
      for (unsigned X = 0; X != CallMap[i].first.first.size(); ++X) {
        cerr << " " << X << " is\n"; CallMap[i].first.first[X].print(cerr);
      }
#endif

      // Look to see if the function call takes a superset of the values we are
      // providing as input
      // 
      CallDescriptor &CD = CallMap[i].first;
      if (CD.second == CN->getCall() && CD.first.size() == Args.size()) {
        bool FoundMismatch = false;
        for (unsigned j = 0, je = Args.size(); j != je; ++j) {
          PointerValSet ArgSet = CD.first[j];
          if (ArgSet.add(Args[j])) {
            FoundMismatch = true; break;
          }            
        }

        if (!FoundMismatch) { CMI = &CallMap[i].second; break; }
      }
    }

    // Hold the set of values that correspond to the incorporated methods
    // return set.
    //
    PointerValSet RetVals;

    if (CMI) {
      // We have already inlined an identical function call!
      RetVals = *CMI;
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
      RetVals = cloneFunctionIntoSelf(NewFunction, false);
      CallMap.push_back(make_pair(CallDescriptor(CN->getArgs(), CN->getCall()),
                                  RetVals));

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
      Changed = UnlinkUndistinguishableNodes();

      // Eliminate shadow nodes that are now extraneous due to linking...
      Changed |= RemoveUnreachableNodes();
    }

    //if (F == Func) return;  // Only do one self inlining
    
    // Move on to the next call node...
    NI = std::find_if(CallNodes.begin(), CallNodes.end(), isResolvableCallNode);
  }
}
