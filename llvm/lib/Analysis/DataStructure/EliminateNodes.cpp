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

// removeEdgesTo - Erase all edges in the graph that point to the specified node
static void removeEdgesTo(DSNode *Node) {
  while (!Node->getReferrers().empty()) {
    PointerValSet *PVS = Node->getReferrers().back();
    PVS->removePointerTo(Node);
  }
}

// UnlinkUndistinguishableShadowNodes - Eliminate shadow nodes that are not
// distinguishable from some other node in the graph...
//
void FunctionDSGraph::UnlinkUndistinguishableShadowNodes() {
  // TODO:
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

void FunctionDSGraph::RemoveUnreachableShadowNodes() {
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
#if 0
        cerr << "Unreachable node eliminated:\n";
        ShadowNodes[i]->print(cerr);
#endif
        DeadNodes.push_back(ShadowNodes[i]);
        ShadowNodes[i]->dropAllReferences();  // Drop references to other nodes
        Reachable.erase(Reachable.begin()+i); // Remove from reachable...
        ShadowNodes.erase(ShadowNodes.begin()+i);   // Remove node entry
        --i;  // Don't skip the next node.
      }

    if (DeadNodes.empty()) return;      // No more dead nodes...

    // All dead nodes are in the DeadNodes vector... delete them now.
    for_each(DeadNodes.begin(), DeadNodes.end(), deleter<DSNode>);
  }
}
