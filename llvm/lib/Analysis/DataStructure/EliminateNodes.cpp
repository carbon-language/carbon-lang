//===- EliminateNodes.cpp - Prune unneccesary nodes in the graph ----------===//
//
// This file contains two node optimizations:
//   1. UnlinkUndistinguishableNodes - Often, after unification, shadow
//      nodes are left around that should not exist anymore.  An example is when
//      a shadow gets unified with a 'new' node, the following graph gets
//      generated:  %X -> Shadow, %X -> New.  Since all of the edges to the
//      shadow node also all go to the New node, we can eliminate the shadow.
//
//   2. RemoveUnreachableNodes - Remove shadow and allocation nodes that are not
//      reachable from some other node in the graph.  Unreachable nodes are left
//      lying around often because a method only refers to some allocations with
//      scalar values or an alloca, then when it is inlined, these references
//      disappear and the nodes become homeless and prunable.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructureGraph.h"
#include "llvm/Value.h"
#include "Support/STLExtras.h"
#include <algorithm>
using std::vector;

//#define DEBUG_NODE_ELIMINATE 1

static void DestroyFirstNodeOfPair(DSNode *N1, DSNode *N2) {
#ifdef DEBUG_NODE_ELIMINATE
  std::cerr << "Found Indistinguishable Node:\n";
  N1->print(std::cerr);
#endif

  // The nodes can be merged.  Make sure that N2 contains all of the
  // outgoing edges (fields) that N1 does...
  //
  assert(N1->getNumLinks() == N2->getNumLinks() &&
         "Same type, diff # fields?");
  for (unsigned i = 0, e = N1->getNumLinks(); i != e; ++i)
    N2->getLink(i).add(N1->getLink(i));
  
  // Now make sure that all of the nodes that point to N1 also point to the node
  // that we are merging it with...
  //
  const vector<PointerValSet*> &Refs = N1->getReferrers();
  for (unsigned i = 0, e = Refs.size(); i != e; ++i) {
    PointerValSet &PVS = *Refs[i];

    bool RanOnce = false;
    for (unsigned j = 0, je = PVS.size(); j != je; ++j)
      if (PVS[j].Node == N1) {
        RanOnce = true;
        PVS.add(PointerVal(N2, PVS[j].Index));
      }

    assert(RanOnce && "Node on user set but cannot find the use!");
  }

  N1->mergeInto(N2);
  N1->removeAllIncomingEdges();
  delete N1;
}

// isIndistinguishableNode - A node is indistinguishable if some other node
// has exactly the same incoming links to it and if the node considers itself
// to be the same as the other node...
//
static bool isIndistinguishableNode(DSNode *DN) {
  if (DN->getReferrers().empty()) {       // No referrers...
    if (isa<ShadowDSNode>(DN) || isa<AllocDSNode>(DN)) {
      delete DN;
      return true;  // Node is trivially dead
    } else
      return false;
  }
  
  // Pick a random referrer... Ptr is the things that the referrer points to.
  // Since DN is in the Ptr set, look through the set seeing if there are any
  // other nodes that are exactly equilivant to DN (with the exception of node
  // type), but are not DN.  If anything exists, then DN is indistinguishable.
  //

  DSNode *IndFrom = 0;
  const vector<PointerValSet*> &Refs = DN->getReferrers();
  for (unsigned R = 0, RE = Refs.size(); R != RE; ++R) {
    const PointerValSet &Ptr = *Refs[R];

    for (unsigned i = 0, e = Ptr.size(); i != e; ++i) {
      DSNode *N2 = Ptr[i].Node;
      if (Ptr[i].Index == 0 && N2 != cast<DSNode>(DN) &&
          DN->getType() == N2->getType() && DN->isEquivalentTo(N2)) {

        IndFrom = N2;
        R = RE-1;
        break;
      }
    }
  }

  // If we haven't found an equivalent node to merge with, see if one of the
  // nodes pointed to by this node is equivalent to this one...
  //
  if (IndFrom == 0) {
    unsigned NumOutgoing = DN->getNumOutgoingLinks();
    for (DSNode::iterator I = DN->begin(), E = DN->end(); I != E; ++I) {
      DSNode *Linked = *I;
      if (Linked != DN && Linked->getNumOutgoingLinks() == NumOutgoing &&
          DN->getType() == Linked->getType() && DN->isEquivalentTo(Linked)) {
#if 0
        // Make sure the leftover node contains links to everything we do...
        for (unsigned i = 0, e = DN->getNumLinks(); i != e; ++i)
          Linked->getLink(i).add(DN->getLink(i));
#endif

        IndFrom = Linked;
        break;
      }
    }
  }


  // If DN is indistinguishable from some other node, merge them now...
  if (IndFrom == 0)
    return false;     // Otherwise, nothing found, perhaps next time....

  DestroyFirstNodeOfPair(DN, IndFrom);
  return true;
}

template<typename NodeTy>
static bool removeIndistinguishableNodes(vector<NodeTy*> &Nodes) {
  bool Changed = false;
  vector<NodeTy*>::iterator I = Nodes.begin();
  while (I != Nodes.end()) {
    if (isIndistinguishableNode(*I)) {
      I = Nodes.erase(I);
      Changed = true;
    } else {
      ++I;
    }
  }
  return Changed;
}

template<typename NodeTy>
static bool removeIndistinguishableNodePairs(vector<NodeTy*> &Nodes) {
  bool Changed = false;
  vector<NodeTy*>::iterator I = Nodes.begin();
  while (I != Nodes.end()) {
    NodeTy *N1 = *I++;
    for (vector<NodeTy*>::iterator I2 = I, I2E = Nodes.end();
         I2 != I2E; ++I2) {
      NodeTy *N2 = *I2;
      if (N1->isEquivalentTo(N2)) {
        DestroyFirstNodeOfPair(N1, N2);
        --I;
        I = Nodes.erase(I);
        Changed = true;
        break;
      }
    }
  }
  return Changed;
}



// UnlinkUndistinguishableNodes - Eliminate shadow nodes that are not
// distinguishable from some other node in the graph...
//
bool FunctionDSGraph::UnlinkUndistinguishableNodes() {
  // Loop over all of the shadow nodes, checking to see if they are
  // indistinguishable from some other node.  If so, eliminate the node!
  //
  return
    removeIndistinguishableNodes(AllocNodes) |
    removeIndistinguishableNodes(ShadowNodes) |
    removeIndistinguishableNodePairs(CallNodes) |
    removeIndistinguishableNodePairs(GlobalNodes);
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
    if (isa<ShadowDSNode>(PVS[i].Node) || isa<AllocDSNode>(PVS[i].Node))
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

  const vector<PointerValSet> *Links = N->getAuxLinks();
  if (Links)
    for (unsigned i = 0, e = Links->size(); i != e; ++i)
      MarkReferredNodeSetReachable((*Links)[i],
                                   ShadowNodes, ReachableShadowNodes,
                                   AllocNodes, ReachableAllocNodes);
}

void FunctionDSGraph::MarkEscapeableNodesReachable(
                                       vector<bool> &ReachableShadowNodes,
                                       vector<bool> &ReachableAllocNodes) {
  // Mark all shadow nodes that have edges from other nodes as reachable.  
  // Recursively mark any shadow nodes pointed to by the newly live shadow
  // nodes as also alive.
  //
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
}

bool FunctionDSGraph::RemoveUnreachableNodes() {
  bool Changed = false;
  bool LocalChange = true;
  
  while (LocalChange) {
    LocalChange = false;
    // Reachable*Nodes - Contains true if there is an edge from a reachable
    // node to the numbered node...
    //
    vector<bool> ReachableShadowNodes(ShadowNodes.size());
    vector<bool> ReachableAllocNodes (AllocNodes.size());
    
    MarkEscapeableNodesReachable(ReachableShadowNodes, ReachableAllocNodes);

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
    for (unsigned i = 0; i != ShadowNodes.size(); ++i)
      if (!ReachableShadowNodes[i]) {
        // Track all unreachable nodes...
#if DEBUG_NODE_ELIMINATE
        std::cerr << "Unreachable node eliminated:\n";
        ShadowNodes[i]->print(std::cerr);
#endif
        ShadowNodes[i]->removeAllIncomingEdges();
        delete ShadowNodes[i];

        // Remove from reachable...
        ReachableShadowNodes.erase(ReachableShadowNodes.begin()+i);
        ShadowNodes.erase(ShadowNodes.begin()+i);   // Remove node entry
        --i;  // Don't skip the next node.
        LocalChange = Changed = true;
      }

    for (unsigned i = 0; i != AllocNodes.size(); ++i)
      if (!ReachableAllocNodes[i]) {
        // Track all unreachable nodes...
#if DEBUG_NODE_ELIMINATE
        std::cerr << "Unreachable node eliminated:\n";
        AllocNodes[i]->print(std::cerr);
#endif
        AllocNodes[i]->removeAllIncomingEdges();
        delete AllocNodes[i];

        // Remove from reachable...
        ReachableAllocNodes.erase(ReachableAllocNodes.begin()+i);
        AllocNodes.erase(AllocNodes.begin()+i);   // Remove node entry
        --i;  // Don't skip the next node.
        LocalChange = Changed = true;
      }
  }

  // Loop over the global nodes, removing nodes that have no edges into them or
  // out of them.
  // 
  for (vector<GlobalDSNode*>::iterator I = GlobalNodes.begin();
       I != GlobalNodes.end(); )
    if ((*I)->getReferrers().empty()) {
      GlobalDSNode *GDN = *I;
      bool NoLinks = true;    // Make sure there are no outgoing links...
      for (unsigned i = 0, e = GDN->getNumLinks(); i != e; ++i)
        if (!GDN->getLink(i).empty()) {
          NoLinks = false;
          break;
        }
      if (NoLinks) {
        delete GDN;
        I = GlobalNodes.erase(I);                     // Remove the node...
        Changed = true;
      } else {
        ++I;
      }
    } else {
      ++I;
    }
  
  return Changed;
}




// getEscapingAllocations - Add all allocations that escape the current
// function to the specified vector.
//
void FunctionDSGraph::getEscapingAllocations(vector<AllocDSNode*> &Allocs) {
  vector<bool> ReachableShadowNodes(ShadowNodes.size());
  vector<bool> ReachableAllocNodes (AllocNodes.size());
    
  MarkEscapeableNodesReachable(ReachableShadowNodes, ReachableAllocNodes);

  for (unsigned i = 0, e = AllocNodes.size(); i != e; ++i)
    if (ReachableAllocNodes[i])
      Allocs.push_back(AllocNodes[i]);
}

// getNonEscapingAllocations - Add all allocations that do not escape the
// current function to the specified vector.
//
void FunctionDSGraph::getNonEscapingAllocations(vector<AllocDSNode*> &Allocs) {
  vector<bool> ReachableShadowNodes(ShadowNodes.size());
  vector<bool> ReachableAllocNodes (AllocNodes.size());
    
  MarkEscapeableNodesReachable(ReachableShadowNodes, ReachableAllocNodes);

  for (unsigned i = 0, e = AllocNodes.size(); i != e; ++i)
    if (!ReachableAllocNodes[i])
      Allocs.push_back(AllocNodes[i]);
}
