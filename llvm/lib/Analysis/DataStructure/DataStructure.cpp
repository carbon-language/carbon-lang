//===- DataStructure.cpp - Implement the core data structure analysis -----===//
//
// This file implements the core data structure functionality.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include <algorithm>
#include "Support/STLExtras.h"

AnalysisID LocalDataStructures::ID(AnalysisID::create<LocalDataStructures>());

//===----------------------------------------------------------------------===//
// DSNode Implementation
//===----------------------------------------------------------------------===//

DSNode::DSNode(enum NodeTy NT, const Type *T) : Ty(T), NodeType(NT) {
  // If this node has any fields, allocate them now, but leave them null.
  switch (T->getPrimitiveID()) {
  case Type::PointerTyID: Links.resize(1); break;
  case Type::ArrayTyID:   Links.resize(1); break;
  case Type::StructTyID:
    Links.resize(cast<StructType>(T)->getNumContainedTypes());
    break;
  default: break;
  }
}

void DSNode::removeReferrer(DSNodeHandle *H) {
  // Search backwards, because we depopulate the list from the back for
  // efficiency (because it's a vector).
  std::vector<DSNodeHandle*>::reverse_iterator I =
    std::find(Referrers.rbegin(), Referrers.rend(), H);
  assert(I != Referrers.rend() && "Referrer not pointing to node!");
  Referrers.erase(I.base()-1);
}

// addGlobal - Add an entry for a global value to the Globals list.  This also
// marks the node with the 'G' flag if it does not already have it.
//
void DSNode::addGlobal(GlobalValue *GV) {
  assert(GV->getType()->getElementType() == Ty);
  Globals.push_back(GV);
  NodeType |= GlobalNode;
}


// addEdgeTo - Add an edge from the current node to the specified node.  This
// can cause merging of nodes in the graph.
//
void DSNode::addEdgeTo(unsigned LinkNo, DSNode *N) {
  assert(LinkNo < Links.size() && "LinkNo out of range!");
  if (N == 0 || Links[LinkNo] == N) return;  // Nothing to do
  if (Links[LinkNo] == 0) {                  // No merging to perform
    Links[LinkNo] = N;
    return;
  }

  // Merge the two nodes...
  Links[LinkNo]->mergeWith(N);
}


// mergeWith - Merge this node into the specified node, moving all links to and
// from the argument node into the current node.  The specified node may be a
// null pointer (in which case, nothing happens).
//
void DSNode::mergeWith(DSNode *N) {
  if (N == 0 || N == this) return;  // Noop
  assert(N->Ty == Ty && N->Links.size() == Links.size() &&
         "Cannot merge nodes of two different types!");

  // Remove all edges pointing at N, causing them to point to 'this' instead.
  while (!N->Referrers.empty())
    *N->Referrers.back() = this;

  // Make all of the outgoing links of N now be outgoing links of this.  This
  // can cause recursive merging!
  //
  for (unsigned i = 0, e = Links.size(); i != e; ++i) {
    addEdgeTo(i, N->Links[i]);
    N->Links[i] = 0;  // Reduce unneccesary edges in graph. N is dead
  }

  // Merge the node types
  NodeType |= N->NodeType;
  N->NodeType = 0;   // N is now a dead node.

  // Merge the globals list...
  Globals.insert(Globals.end(), N->Globals.begin(), N->Globals.end());
  N->Globals.clear();
}

//===----------------------------------------------------------------------===//
// DSGraph Implementation
//===----------------------------------------------------------------------===//

DSGraph::~DSGraph() {
  FunctionCalls.clear();
  ValueMap.clear();
  RetNode = 0;

#ifndef NDEBUG
  // Drop all intra-node references, so that assertions don't fail...
  std::for_each(Nodes.begin(), Nodes.end(),
                std::mem_fun(&DSNode::dropAllReferences));
#endif

  // Delete all of the nodes themselves...
  std::for_each(Nodes.begin(), Nodes.end(), deleter<DSNode>);
}

//===----------------------------------------------------------------------===//
// LocalDataStructures Implementation
//===----------------------------------------------------------------------===//

// releaseMemory - If the pass pipeline is done with this pass, we can release
// our memory... here...
//
void LocalDataStructures::releaseMemory() {
  for (std::map<Function*, DSGraph*>::iterator I = DSInfo.begin(),
         E = DSInfo.end(); I != E; ++I)
    delete I->second;

  // Empty map so next time memory is released, data structures are not
  // re-deleted.
  DSInfo.clear();
}

bool LocalDataStructures::run(Module &M) {
  // Calculate all of the graphs...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal()) {
      std::map<Function*, DSGraph*>::iterator DI = DSInfo.find(I);
      if (DI == DSInfo.end() || DI->second == 0)
        DSInfo.insert(std::make_pair(&*I, new DSGraph(*I)));
    }

  return false;
}
