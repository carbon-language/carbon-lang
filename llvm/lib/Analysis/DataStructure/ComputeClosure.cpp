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

// copyEdgesFromTo - Make a copy of all of the edges to Node to also point
// PV.  If there are edges out of Node, the edges are added to the subgraph
// starting at PV.
//
static void copyEdgesFromTo(DSNode *Node, const PointerValSet &PVS) {
  // Make all of the pointers that pointed to Node now also point to PV...
  const vector<PointerValSet*> &PVSToUpdate(Node->getReferrers());
  for (unsigned i = 0, e = PVSToUpdate.size(); i != e; ++i)
    for (unsigned pn = 0, pne = PVS.size(); pn != pne; ++pn)
      PVSToUpdate[i]->add(PVS[pn]);
}

static void CalculateNodeMapping(ShadowDSNode *Shadow, DSNode *Node,
                              multimap<ShadowDSNode *, DSNode *> &NodeMapping) {
#ifdef DEBUG_IP_CLOSURE
  cerr << "Mapping " << (void*)Shadow << " to " << (void*)Node << "\n";
  cerr << "Type = '" << Shadow->getType() << "' and '"
       << Node->getType() << "'\n";
  cerr << "Shadow Node:\n";
  Shadow->print(cerr);
  cerr << "\nMapped Node:\n";
  Node->print(cerr);
#endif
  assert(Shadow->getType() == Node->getType() &&
         "Shadow and mapped nodes disagree about type!");
  
  multimap<ShadowDSNode *, DSNode *>::iterator
    NI = NodeMapping.lower_bound(Shadow),
    NE = NodeMapping.upper_bound(Shadow);

  for (; NI != NE; ++NI)
    if (NI->second == Node) return;       // Already processed node, return.

  NodeMapping.insert(make_pair(Shadow, Node));   // Add a mapping...

  // Loop over all of the outgoing links in the shadow node...
  //
  assert(Node->getNumLinks() == Shadow->getNumLinks() &&
         "Same type, but different number of links?");
  for (unsigned i = 0, e = Shadow->getNumLinks(); i != e; ++i) {
    PointerValSet &Link = Shadow->getLink(i);

    // Loop over all of the values coming out of this pointer...
    for (unsigned l = 0, le = Link.size(); l != le; ++l) {
      // If the outgoing node points to a shadow node, map the shadow node to
      // all of the outgoing values in Node.
      //
      if (ShadowDSNode *ShadOut = dyn_cast<ShadowDSNode>(Link[l].Node)) {
        PointerValSet &NLink = Node->getLink(i);
        for (unsigned ol = 0, ole = NLink.size(); ol != ole; ++ol)
          CalculateNodeMapping(ShadOut, NLink[ol].Node, NodeMapping);
      }
    }
  }
}


static void ResolveNodesTo(const PointerVal &FromPtr,
                           const PointerValSet &ToVals) {
  assert(FromPtr.Index == 0 &&
         "Resolved node return pointer should be index 0!");
  if (!isa<ShadowDSNode>(FromPtr.Node)) return;
  
  ShadowDSNode *Shadow = cast<ShadowDSNode>(FromPtr.Node);
  Shadow->resetCriticalMark();

  typedef multimap<ShadowDSNode *, DSNode *> ShadNodeMapTy;
  ShadNodeMapTy NodeMapping;
  for (unsigned i = 0, e = ToVals.size(); i != e; ++i)
    CalculateNodeMapping(Shadow, ToVals[i].Node, NodeMapping);

  // Now loop through the shadow node graph, mirroring the edges in the shadow
  // graph onto the realized graph...
  //
  for (ShadNodeMapTy::iterator I = NodeMapping.begin(),
         E = NodeMapping.end(); I != E; ++I) {
    DSNode *Node = I->second;
    ShadowDSNode *ShadNode = I->first;
    PointerValSet PVSx;
    PVSx.add(Node);
    copyEdgesFromTo(ShadNode, PVSx);

    // Must loop over edges in the shadow graph, adding edges in the real graph
    // that correspond to to the edges, but are mapped into real values by the
    // NodeMapping.
    //
    for (unsigned i = 0, e = Node->getNumLinks(); i != e; ++i) {
      const PointerValSet &ShadLinks = ShadNode->getLink(i);
      PointerValSet &NewLinks = Node->getLink(i);

      // Add a link to all of the nodes pointed to by the shadow field...
      for (unsigned l = 0, le = ShadLinks.size(); l != le; ++l) {
        DSNode *ShadLink = ShadLinks[l].Node;

        if (ShadowDSNode *SL = dyn_cast<ShadowDSNode>(ShadLink)) {
          // Loop over all of the values in the range 
          ShadNodeMapTy::iterator St = NodeMapping.lower_bound(SL),
                                  En = NodeMapping.upper_bound(SL);
          if (St != En) {
            for (; St != En; ++St)
              NewLinks.add(PointerVal(St->second, ShadLinks[l].Index));
          } else {
            // We must retain the shadow node...
            NewLinks.add(ShadLinks[l]);
          }
        } else {
          // Otherwise, add a direct link to the data structure pointed to by
          // the shadow node...
          NewLinks.add(ShadLinks[l]);
        }
      }
    }
  }
}


// ResolveNodeTo - The specified node is now known to point to the set of values
// in ToVals, instead of the old shadow node subgraph that it was pointing to.
//
static void ResolveNodeTo(DSNode *Node, const PointerValSet &ToVals) {
  assert(Node->getNumLinks() == 1 && "Resolved node can only be a scalar!!");

  PointerValSet PVS = Node->getLink(0);

  for (unsigned i = 0, e = PVS.size(); i != e; ++i)
    ResolveNodesTo(PVS[i], ToVals);
}

// isResolvableCallNode - Return true if node is a call node and it is a call
// node that we can inline...
//
static bool isResolvableCallNode(DSNode *N) {
  // Only operate on call nodes...
  CallDSNode *CN = dyn_cast<CallDSNode>(N);
  if (CN == 0) return false;

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
  vector<DSNode*>::iterator NI = std::find_if(Nodes.begin(), Nodes.end(),
                                              isResolvableCallNode);

  map<Function*, unsigned> InlineCount; // FIXME

  // Loop over the resolvable call nodes...
  while (NI != Nodes.end()) {
    CallDSNode *CN = cast<CallDSNode>(*NI);
    Function *F = CN->getCall()->getCalledFunction();
    //if (F == Func) return;  // Do not do self inlining

    // FIXME: Gross hack to prevent explosions when inlining a recursive func.
    if (InlineCount[F]++ > 2) return;

    Nodes.erase(NI);                     // Remove the call node from the graph

    unsigned CallNodeOffset = NI-Nodes.begin();

    // StartNode - The first node of the incorporated graph, last node of the
    // preexisting data structure graph...
    //
    unsigned StartNode = Nodes.size();

    // Hold the set of values that correspond to the incorporated methods
    // return set.
    //
    PointerValSet RetVals;

    if (F != Func) {  // If this is not a recursive call...
      // Get the datastructure graph for the new method.  Note that we are not
      // allowed to modify this graph because it will be the cached graph that
      // is returned by other users that want the local datastructure graph for
      // a method.
      //
      const FunctionDSGraph &NewFunction = DS.getDSGraph(F);

      unsigned StartShadowNodes = ShadowNodes.size();

      // Incorporate a copy of the called function graph into the current graph,
      // allowing us to do local transformations to local graph to link
      // arguments to call values, and call node to return value...
      //
      RetVals = cloneFunctionIntoSelf(NewFunction, false);

      // Only detail is that we need to reset all of the critical shadow nodes
      // in the incorporated graph, because they are now no longer critical.
      //
      for (unsigned i = StartShadowNodes, e = ShadowNodes.size(); i != e; ++i)
        ShadowNodes[i]->resetCriticalMark();

    } else {     // We are looking at a recursive function!
      StartNode = 0;  // Arg nodes start at 0 now...
      RetVals = RetNode;
    }

    // If the function returns a pointer value...  Resolve values pointing to
    // the shadow nodes pointed to by CN to now point the values in RetVals...
    //
    if (CN->getNumLinks()) ResolveNodeTo(CN, RetVals);

    // If the call node has arguments, process them now!
    if (CN->getNumArgs()) {
      // The ArgNodes of the incorporated graph should be the nodes starting at
      // StartNode, ordered the same way as the call arguments.  The arg nodes
      // are seperated by a single shadow node, but that shadow node might get
      // eliminated in the process of optimization.
      //
      unsigned ArgOffset = StartNode;
      for (unsigned i = 0, e = CN->getNumArgs(); i != e; ++i) {
        // Get the arg node of the incorporated method...
        while (!isa<ArgDSNode>(Nodes[ArgOffset]))  // Scan for next arg node
          ArgOffset++;
        ArgDSNode *ArgNode = cast<ArgDSNode>(Nodes[ArgOffset]);

        // Now we make all of the nodes inside of the incorporated method point
        // to the real arguments values, not to the shadow nodes for the
        // argument.
        //
        ResolveNodeTo(ArgNode, CN->getArgValues(i));

        if (StartNode) {        // Not Self recursion?
          // Remove the argnode from the set of nodes in this method...
          Nodes.erase(Nodes.begin()+ArgOffset);

          // ArgNode is no longer useful, delete now!
          delete ArgNode;
        } else {
          ArgOffset++;  // Step to the next argument...
        }
      }
    }

    // Loop through the nodes, deleting alloc nodes in the inlined function...
    // Since the memory has been released, we cannot access their pointer
    // fields (with defined results at least), so it is not possible to use any
    // pointers to the alloca.  Drop them now, and remove the alloca's since
    // they are dead (we just removed all links to them).  Only do this if we
    // are not self recursing though.  :)
    //
    if (StartNode)  // Don't do this if self recursing...
      for (unsigned i = StartNode; i != Nodes.size(); ++i)
        if (NewDSNode *NDS = dyn_cast<NewDSNode>(Nodes[i]))
          if (NDS->isAllocaNode()) {
            NDS->removeAllIncomingEdges();  // These edges are invalid now!
            delete NDS;                     // Node is dead
            Nodes.erase(Nodes.begin()+i);   // Remove slot in Nodes array
            --i;                            // Don't skip the next node
          }


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
    NI = std::find_if(Nodes.begin(), Nodes.end(), isResolvableCallNode);
  }
}
