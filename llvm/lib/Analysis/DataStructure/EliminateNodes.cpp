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

bool AllocDSNode::isEquivalentTo(DSNode *Node) const {
  if (AllocDSNode *N = dyn_cast<AllocDSNode>(Node))
    return N->Allocation == Allocation;
  return false;
}

bool GlobalDSNode::isEquivalentTo(DSNode *Node) const {
  if (GlobalDSNode *G = dyn_cast<GlobalDSNode>(Node))
    return G->Val == Val;
  return false;
}

bool CallDSNode::isEquivalentTo(DSNode *Node) const {
  if (CallDSNode *C = dyn_cast<CallDSNode>(Node))
    return C->CI == CI && C->ArgLinks == ArgLinks;
  return false;
}

bool ArgDSNode::isEquivalentTo(DSNode *Node) const {
  return false;
}

// NodesAreEquivalent - Check to see if the nodes are equivalent in all ways
// except node type.  Since we know N1 is a shadow node, N2 is allowed to be
// any type.
//
bool ShadowDSNode::isEquivalentTo(DSNode *Node) const {
  return !isCriticalNode();              // Must not be a critical node...
}



// isIndistinguishableNode - A node is indistinguishable if some other node
// has exactly the same incoming links to it and if the node considers itself
// to be the same as the other node...
//
bool isIndistinguishableNode(DSNode *DN) {
  if (DN->getReferrers().empty()) {       // No referrers...
    if (isa<ShadowDSNode>(DN) || isa<AllocDSNode>(DN))
      return true;  // Node is trivially dead
    else
      return false;
  }
  
  // Pick a random referrer... Ptr is the things that the referrer points to.
  // Since DN is in the Ptr set, look through the set seeing if there are any
  // other nodes that are exactly equilivant to DN (with the exception of node
  // type), but are not DN.  If anything exists, then DN is indistinguishable.
  //
  const std::vector<PointerValSet*> &Refs = DN->getReferrers();
  for (unsigned R = 0, RE = Refs.size(); R != RE; ++R) {
    const PointerValSet &Ptr = *Refs[R];

    for (unsigned i = 0, e = Ptr.size(); i != e; ++i) {
      DSNode *N2 = Ptr[i].Node;
      if (Ptr[i].Index == 0 && N2 != cast<DSNode>(DN) &&
          DN->getType() == N2->getType() && DN->isEquivalentTo(N2)) {

        // Otherwise, the nodes can be merged.  Make sure that N2 contains all
        // of the  outgoing edges (fields) that DN does...
        //
        assert(DN->getNumLinks() == N2->getNumLinks() &&
               "Same type, diff # fields?");
        for (unsigned i = 0, e = DN->getNumLinks(); i != e; ++i)
          N2->getLink(i).add(DN->getLink(i));
        
        // Now make sure that all of the nodes that point to the shadow node
        // also  point to the node that we are merging it with...
        //
        const std::vector<PointerValSet*> &Refs = DN->getReferrers();
        for (unsigned i = 0, e = Refs.size(); i != e; ++i) {
          PointerValSet &PVS = *Refs[i];
          // FIXME: this is incorrect if the referring pointer has index != 0
          //
          PVS.add(N2);
        }
        return true;
      }
    }
  }

  // Otherwise, nothing found, perhaps next time....
  return false;
}

template<typename NodeTy>
bool removeIndistinguishableNode(std::vector<NodeTy*> &Nodes) {
  bool Changed = false;
  std::vector<NodeTy*>::iterator I = Nodes.begin();
  while (I != Nodes.end()) {
    if (isIndistinguishableNode(*I)) {
#ifdef DEBUG_NODE_ELIMINATE
      cerr << "Found Indistinguishable Node:\n";
      (*I)->print(cerr);
#endif
      (*I)->removeAllIncomingEdges();
      delete *I;
      I = Nodes.erase(I);
      Changed = true;
    } else {
      ++I;
    }
  }
  return Changed;
}

// UnlinkUndistinguishableShadowNodes - Eliminate shadow nodes that are not
// distinguishable from some other node in the graph...
//
bool FunctionDSGraph::UnlinkUndistinguishableShadowNodes() {
  // Loop over all of the shadow nodes, checking to see if they are
  // indistinguishable from some other node.  If so, eliminate the node!
  //
  return
    removeIndistinguishableNode(AllocNodes) |
    removeIndistinguishableNode(ShadowNodes) |
    removeIndistinguishableNode(GlobalNodes);
}

static void MarkReferredNodesReachable(DSNode *N,
                                       vector<ShadowDSNode*> &ShadowNodes,
                                       vector<bool> &ReachableShadowNodes,
                                       vector<AllocDSNode*>  &AllocNodes,
                                       vector<bool> &ReachableAllocNodes);

static inline void MarkReferredNodeSetReachable(const PointerValSet &PVS,
                                            vector<ShadowDSNode*> &ShadowNodes,
                                            vector<bool> &ReachableShadowNodes,
                                            vector<AllocDSNode*>  &AllocNodes,
                                            vector<bool> &ReachableAllocNodes) {
  for (unsigned i = 0, e = PVS.size(); i != e; ++i)
    if (isa<ShadowDSNode>(PVS[i].Node) || isa<ShadowDSNode>(PVS[i].Node))
      MarkReferredNodesReachable(PVS[i].Node, ShadowNodes, ReachableShadowNodes,
                                 AllocNodes, ReachableAllocNodes);
}

static void MarkReferredNodesReachable(DSNode *N,
                                       vector<ShadowDSNode*> &ShadowNodes,
                                       vector<bool> &ReachableShadowNodes,
                                       vector<AllocDSNode*>  &AllocNodes,
                                       vector<bool> &ReachableAllocNodes) {
  assert(ShadowNodes.size() == ReachableShadowNodes.size());
  assert(AllocNodes.size()  == ReachableAllocNodes.size());

  if (ShadowDSNode *Shad = dyn_cast<ShadowDSNode>(N)) {
    vector<ShadowDSNode*>::iterator I =
      std::find(ShadowNodes.begin(), ShadowNodes.end(), Shad);
    unsigned i = I-ShadowNodes.begin();
    if (ReachableShadowNodes[i]) return;  // Recursion detected, abort...
    ReachableShadowNodes[i] = true;
  } else if (AllocDSNode *Alloc = dyn_cast<AllocDSNode>(N)) {
    vector<AllocDSNode*>::iterator I =
      std::find(AllocNodes.begin(), AllocNodes.end(), Alloc);
    unsigned i = I-AllocNodes.begin();
    if (ReachableAllocNodes[i]) return;  // Recursion detected, abort...
    ReachableAllocNodes[i] = true;
  }

  for (unsigned i = 0, e = N->getNumLinks(); i != e; ++i)
    MarkReferredNodeSetReachable(N->getLink(i),
                                 ShadowNodes, ReachableShadowNodes,
                                 AllocNodes, ReachableAllocNodes);

  const std::vector<PointerValSet> *Links = N->getAuxLinks();
  if (Links)
    for (unsigned i = 0, e = Links->size(); i != e; ++i)
      MarkReferredNodeSetReachable((*Links)[i],
                                   ShadowNodes, ReachableShadowNodes,
                                   AllocNodes, ReachableAllocNodes);
}

bool FunctionDSGraph::RemoveUnreachableShadowNodes() {
  bool Changed = false;
  while (1) {
    // Reachable*Nodes - Contains true if there is an edge from a reachable
    // node to the numbered node...
    //
    vector<bool> ReachableShadowNodes(ShadowNodes.size());
    vector<bool> ReachableAllocNodes (AllocNodes.size());

    // Mark all shadow nodes that have edges from other nodes as reachable.  
    // Recursively mark any shadow nodes pointed to by the newly live shadow
    // nodes as also alive.
    //
    for (unsigned i = 0, e = ArgNodes.size(); i != e; ++i)
      MarkReferredNodesReachable(ArgNodes[i],
                                 ShadowNodes, ReachableShadowNodes,
                                 AllocNodes, ReachableAllocNodes);

    for (unsigned i = 0, e = GlobalNodes.size(); i != e; ++i)
      MarkReferredNodesReachable(GlobalNodes[i],
                                 ShadowNodes, ReachableShadowNodes,
                                 AllocNodes, ReachableAllocNodes);

    for (unsigned i = 0, e = CallNodes.size(); i != e; ++i)
      MarkReferredNodesReachable(CallNodes[i],
                                 ShadowNodes, ReachableShadowNodes,
                                 AllocNodes, ReachableAllocNodes);

    // Mark all nodes in the return set as being reachable...
    MarkReferredNodeSetReachable(RetNode,
                                 ShadowNodes, ReachableShadowNodes,
                                 AllocNodes, ReachableAllocNodes);

    // Mark all nodes in the value map as being reachable...
    for (std::map<Value*, PointerValSet>::iterator I = ValueMap.begin(),
           E = ValueMap.end(); I != E; ++I)
      MarkReferredNodeSetReachable(I->second,
                                   ShadowNodes, ReachableShadowNodes,
                                   AllocNodes, ReachableAllocNodes);

    // At this point, all reachable shadow nodes have a true value in the
    // Reachable vector.  This means that any shadow nodes without an entry in
    // the reachable vector are not reachable and should be removed.  This is 
    // a two part process, because we must drop all references before we delete
    // the shadow nodes [in case cycles exist].
    // 
    bool LocalChange = false;
    for (unsigned i = 0; i != ShadowNodes.size(); ++i)
      if (!ReachableShadowNodes[i]) {
        // Track all unreachable nodes...
#if DEBUG_NODE_ELIMINATE
        cerr << "Unreachable node eliminated:\n";
        ShadowNodes[i]->print(cerr);
#endif
        ShadowNodes[i]->removeAllIncomingEdges();
        delete ShadowNodes[i];

        // Remove from reachable...
        ReachableShadowNodes.erase(ReachableShadowNodes.begin()+i);
        ShadowNodes.erase(ShadowNodes.begin()+i);   // Remove node entry
        --i;  // Don't skip the next node.
        LocalChange = true;
      }

    for (unsigned i = 0; i != AllocNodes.size(); ++i)
      if (!ReachableAllocNodes[i]) {
        // Track all unreachable nodes...
#if DEBUG_NODE_ELIMINATE
        cerr << "Unreachable node eliminated:\n";
        AllocNodes[i]->print(cerr);
#endif
        AllocNodes[i]->removeAllIncomingEdges();
        delete AllocNodes[i];

        // Remove from reachable...
        ReachableAllocNodes.erase(ReachableAllocNodes.begin()+i);
        AllocNodes.erase(AllocNodes.begin()+i);   // Remove node entry
        --i;  // Don't skip the next node.
        LocalChange = true;
      }

    if (!LocalChange) return Changed;      // No more dead nodes...

    Changed = true;
  }
}
