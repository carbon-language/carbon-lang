//===- ShadowNodeEliminate.cpp - Optimize away shadow nodes ---------------===//
//
// This file contains two shadow node optimizations:
//   1. UnlinkUndistinguishableShadowNodes - Often, after unification, shadow
//      nodes are left around that should not exist anymore.  An example is when
//      a shadow gets unified with a 'new' node, the following graph gets
//      generated:  %X -> Shadow, %X -> New.  Since all of the edges to the
//      shadow node also all go to the New node, we can eliminate the shadow.
//
//   2. RemoveUnreachableShadowNodes - Remove shadow nodes that are not
//      reachable from some other node in the graph.  Unreachable shadow nodes
//      are left lying around because other transforms don't go to the trouble
//      or removing them, since this pass exists.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Value.h"
#include "Support/STLExtras.h"
#include <algorithm>

//#define DEBUG_NODE_ELIMINATE 1

// NodesAreEquivalent - Check to see if the nodes are equivalent in all ways
// except node type.  Since we know N1 is a shadow node, N2 is allowed to be
// any type.
//
static bool NodesAreEquivalent(const ShadowDSNode *N1, const DSNode *N2) {
  assert(N1 != N2 && "A node is always equivalent to itself!");

  // Perform simple, fast checks first...
  if (N1->getType() != N2->getType() ||  // Must have same type...
      N1->isCriticalNode())              // Must not be a critical node...
    return false;

#if 0
  return true;
#else

  // The shadow node is considered equivalent if it has a subset of the incoming
  // edges that N2 does...
  if (N1->getReferrers().size() > N2->getReferrers().size()) return false;

  // Check to see if the referring (incoming) pointers are all the same...
  std::vector<PointerValSet*> N1R = N1->getReferrers();
  std::vector<PointerValSet*> N2R = N2->getReferrers();
  sort(N1R.begin(), N1R.end());
  sort(N2R.begin(), N2R.end());

  // The nodes are equivalent if the incoming edges to N1 are a subset of N2.
  unsigned i1 = 0, e1 = N1R.size();
  unsigned i2 = 0, e2 = N2R.size();
  for (; i1 != e1 && i2 < e2; ++i1, ++i2) {
    while (N1R[i1] > N2R[i2] && i2 < e2)
      ++i2;

    if (N1R[i1] < N2R[i2]) return false;  // Error case...
  }

  return i1 == e1 && i2 <= e2;
#endif
}

// IndistinguishableShadowNode - A shadow node is indistinguishable if some
// other node (shadow or otherwise) has exactly the same incoming and outgoing
// links to it (or if there are no edges coming in, in which it is trivially
// dead).
//
static bool IndistinguishableShadowNode(const ShadowDSNode *SN) {
  if (SN->getReferrers().empty()) return true;  // Node is trivially dead

  // Pick a random referrer... Ptr is the things that the referrer points to.
  // Since SN is in the Ptr set, look through the set seeing if there are any
  // other nodes that are exactly equilivant to SN (with the exception of node
  // type), but are not SN.  If anything exists, then SN is indistinguishable.
  //
  const PointerValSet &Ptr = *SN->getReferrers()[0];

  for (unsigned i = 0, e = Ptr.size(); i != e; ++i)
    if (Ptr[i].Index == 0 && Ptr[i].Node != cast<DSNode>(SN) &&
        NodesAreEquivalent(SN, Ptr[i].Node))
      return true;

  // Otherwise, nothing found, perhaps next time....
  return false;
}


// UnlinkUndistinguishableShadowNodes - Eliminate shadow nodes that are not
// distinguishable from some other node in the graph...
//
bool FunctionDSGraph::UnlinkUndistinguishableShadowNodes() {
  bool Changed = false;
  // Loop over all of the shadow nodes, checking to see if they are
  // indistinguishable from some other node.  If so, eliminate the node!
  //
  for (vector<ShadowDSNode*>::iterator I = ShadowNodes.begin();
       I != ShadowNodes.end(); )
    if (IndistinguishableShadowNode(*I)) {
#ifdef DEBUG_NODE_ELIMINATE
      cerr << "Found Indistinguishable Shadow Node:\n";
      (*I)->print(cerr);
#endif
      (*I)->removeAllIncomingEdges();
      // Don't need to dropAllRefs, because nothing can point to it now
      delete *I;
      
      I = ShadowNodes.erase(I);
      Changed = true;
    } else {
      ++I;
    }
  return Changed;
}

static void MarkReferredNodesReachable(DSNode *N, vector<ShadowDSNode*> &Nodes,
                                       vector<bool> &Reachable);

static inline void MarkReferredNodeSetReachable(const PointerValSet &PVS,
                                                vector<ShadowDSNode*> &Nodes,
                                                vector<bool> &Reachable) {
  for (unsigned i = 0, e = PVS.size(); i != e; ++i)
    if (ShadowDSNode *Shad = dyn_cast<ShadowDSNode>(PVS[i].Node))
      MarkReferredNodesReachable(Shad, Nodes, Reachable);
}

static void MarkReferredNodesReachable(DSNode *N, vector<ShadowDSNode*> &Nodes,
                                       vector<bool> &Reachable) {
  assert(Nodes.size() == Reachable.size());
  ShadowDSNode *Shad = dyn_cast<ShadowDSNode>(N);

  if (Shad) {
    vector<ShadowDSNode*>::iterator I =
      std::find(Nodes.begin(), Nodes.end(), Shad);
    unsigned i = I-Nodes.begin();
    if (Reachable[i]) return;  // Recursion detected, abort...
    Reachable[i] = true;
  }

  for (unsigned i = 0, e = N->getNumLinks(); i != e; ++i)
    MarkReferredNodeSetReachable(N->getLink(i), Nodes, Reachable);

  const std::vector<PointerValSet> *Links = N->getAuxLinks();
  if (Links)
    for (unsigned i = 0, e = Links->size(); i != e; ++i)
      MarkReferredNodeSetReachable((*Links)[i], Nodes, Reachable);  
}

bool FunctionDSGraph::RemoveUnreachableShadowNodes() {
  bool Changed = false;
  while (1) {

    // Reachable - Contains true if there is an edge from a nonshadow node to
    // the numbered node...
    //
    vector<bool> Reachable(ShadowNodes.size());

    // Mark all shadow nodes that have edges from other nodes as reachable.  
    // Recursively mark any shadow nodes pointed to by the newly live shadow
    // nodes as also alive.
    //
    for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
      // Loop over all of the nodes referred and mark them live if they are
      // shadow nodes...
      MarkReferredNodesReachable(Nodes[i], ShadowNodes, Reachable);

    // Mark all nodes in the return set as being reachable...
    MarkReferredNodeSetReachable(RetNode, ShadowNodes, Reachable);

    // Mark all nodes in the value map as being reachable...
    for (std::map<Value*, PointerValSet>::iterator I = ValueMap.begin(),
           E = ValueMap.end(); I != E; ++I)
      MarkReferredNodeSetReachable(I->second, ShadowNodes, Reachable);


    // At this point, all reachable shadow nodes have a true value in the
    // Reachable vector.  This means that any shadow nodes without an entry in
    // the reachable vector are not reachable and should be removed.  This is 
    // a two part process, because we must drop all references before we delete
    // the shadow nodes [in case cycles exist].
    // 
    vector<ShadowDSNode*> DeadNodes;
    for (unsigned i = 0; i != ShadowNodes.size(); ++i)
      if (!Reachable[i]) {
        // Track all unreachable nodes...
#if DEBUG_NODE_ELIMINATE
        cerr << "Unreachable node eliminated:\n";
        ShadowNodes[i]->print(cerr);
#endif
        DeadNodes.push_back(ShadowNodes[i]);
        ShadowNodes[i]->dropAllReferences();  // Drop references to other nodes
        Reachable.erase(Reachable.begin()+i); // Remove from reachable...
        ShadowNodes.erase(ShadowNodes.begin()+i);   // Remove node entry
        --i;  // Don't skip the next node.
      }

    if (DeadNodes.empty()) return Changed;      // No more dead nodes...

    Changed = true;

    // All dead nodes are in the DeadNodes vector... delete them now.
    for_each(DeadNodes.begin(), DeadNodes.end(), deleter<DSNode>);
  }
}
