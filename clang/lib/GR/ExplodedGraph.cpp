//=-- ExplodedGraph.cpp - Local, Path-Sens. "Exploded Graph" -*- C++ -*------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the template classes ExplodedNode and ExplodedGraph,
//  which represent a path-sensitive, intra-procedural "exploded graph."
//
//===----------------------------------------------------------------------===//

#include "clang/GR/PathSensitive/ExplodedGraph.h"
#include "clang/GR/PathSensitive/GRState.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

using namespace clang;
using namespace ento;

//===----------------------------------------------------------------------===//
// Node auditing.
//===----------------------------------------------------------------------===//

// An out of line virtual method to provide a home for the class vtable.
ExplodedNode::Auditor::~Auditor() {}

#ifndef NDEBUG
static ExplodedNode::Auditor* NodeAuditor = 0;
#endif

void ExplodedNode::SetAuditor(ExplodedNode::Auditor* A) {
#ifndef NDEBUG
  NodeAuditor = A;
#endif
}

//===----------------------------------------------------------------------===//
// ExplodedNode.
//===----------------------------------------------------------------------===//

static inline BumpVector<ExplodedNode*>& getVector(void* P) {
  return *reinterpret_cast<BumpVector<ExplodedNode*>*>(P);
}

void ExplodedNode::addPredecessor(ExplodedNode* V, ExplodedGraph &G) {
  assert (!V->isSink());
  Preds.addNode(V, G);
  V->Succs.addNode(this, G);
#ifndef NDEBUG
  if (NodeAuditor) NodeAuditor->AddEdge(V, this);
#endif
}

void ExplodedNode::NodeGroup::addNode(ExplodedNode* N, ExplodedGraph &G) {
  assert((reinterpret_cast<uintptr_t>(N) & Mask) == 0x0);
  assert(!getFlag());

  if (getKind() == Size1) {
    if (ExplodedNode* NOld = getNode()) {
      BumpVectorContext &Ctx = G.getNodeAllocator();
      BumpVector<ExplodedNode*> *V = 
        G.getAllocator().Allocate<BumpVector<ExplodedNode*> >();
      new (V) BumpVector<ExplodedNode*>(Ctx, 4);
      
      assert((reinterpret_cast<uintptr_t>(V) & Mask) == 0x0);
      V->push_back(NOld, Ctx);
      V->push_back(N, Ctx);
      P = reinterpret_cast<uintptr_t>(V) | SizeOther;
      assert(getPtr() == (void*) V);
      assert(getKind() == SizeOther);
    }
    else {
      P = reinterpret_cast<uintptr_t>(N);
      assert(getKind() == Size1);
    }
  }
  else {
    assert(getKind() == SizeOther);
    getVector(getPtr()).push_back(N, G.getNodeAllocator());
  }
}

unsigned ExplodedNode::NodeGroup::size() const {
  if (getFlag())
    return 0;

  if (getKind() == Size1)
    return getNode() ? 1 : 0;
  else
    return getVector(getPtr()).size();
}

ExplodedNode **ExplodedNode::NodeGroup::begin() const {
  if (getFlag())
    return NULL;

  if (getKind() == Size1)
    return (ExplodedNode**) (getPtr() ? &P : NULL);
  else
    return const_cast<ExplodedNode**>(&*(getVector(getPtr()).begin()));
}

ExplodedNode** ExplodedNode::NodeGroup::end() const {
  if (getFlag())
    return NULL;

  if (getKind() == Size1)
    return (ExplodedNode**) (getPtr() ? &P+1 : NULL);
  else {
    // Dereferencing end() is undefined behaviour. The vector is not empty, so
    // we can dereference the last elem and then add 1 to the result.
    return const_cast<ExplodedNode**>(getVector(getPtr()).end());
  }
}

ExplodedNode *ExplodedGraph::getNode(const ProgramPoint& L,
                                     const GRState* State, bool* IsNew) {
  // Profile 'State' to determine if we already have an existing node.
  llvm::FoldingSetNodeID profile;
  void* InsertPos = 0;

  NodeTy::Profile(profile, L, State);
  NodeTy* V = Nodes.FindNodeOrInsertPos(profile, InsertPos);

  if (!V) {
    // Allocate a new node.
    V = (NodeTy*) getAllocator().Allocate<NodeTy>();
    new (V) NodeTy(L, State);

    // Insert the node into the node set and return it.
    Nodes.InsertNode(V, InsertPos);

    ++NumNodes;

    if (IsNew) *IsNew = true;
  }
  else
    if (IsNew) *IsNew = false;

  return V;
}

std::pair<ExplodedGraph*, InterExplodedGraphMap*>
ExplodedGraph::Trim(const NodeTy* const* NBeg, const NodeTy* const* NEnd,
               llvm::DenseMap<const void*, const void*> *InverseMap) const {

  if (NBeg == NEnd)
    return std::make_pair((ExplodedGraph*) 0,
                          (InterExplodedGraphMap*) 0);

  assert (NBeg < NEnd);

  llvm::OwningPtr<InterExplodedGraphMap> M(new InterExplodedGraphMap());

  ExplodedGraph* G = TrimInternal(NBeg, NEnd, M.get(), InverseMap);

  return std::make_pair(static_cast<ExplodedGraph*>(G), M.take());
}

ExplodedGraph*
ExplodedGraph::TrimInternal(const ExplodedNode* const* BeginSources,
                            const ExplodedNode* const* EndSources,
                            InterExplodedGraphMap* M,
                   llvm::DenseMap<const void*, const void*> *InverseMap) const {

  typedef llvm::DenseSet<const ExplodedNode*> Pass1Ty;
  Pass1Ty Pass1;

  typedef llvm::DenseMap<const ExplodedNode*, ExplodedNode*> Pass2Ty;
  Pass2Ty& Pass2 = M->M;

  llvm::SmallVector<const ExplodedNode*, 10> WL1, WL2;

  // ===- Pass 1 (reverse DFS) -===
  for (const ExplodedNode* const* I = BeginSources; I != EndSources; ++I) {
    assert(*I);
    WL1.push_back(*I);
  }

  // Process the first worklist until it is empty.  Because it is a std::list
  // it acts like a FIFO queue.
  while (!WL1.empty()) {
    const ExplodedNode *N = WL1.back();
    WL1.pop_back();

    // Have we already visited this node?  If so, continue to the next one.
    if (Pass1.count(N))
      continue;

    // Otherwise, mark this node as visited.
    Pass1.insert(N);

    // If this is a root enqueue it to the second worklist.
    if (N->Preds.empty()) {
      WL2.push_back(N);
      continue;
    }

    // Visit our predecessors and enqueue them.
    for (ExplodedNode** I=N->Preds.begin(), **E=N->Preds.end(); I!=E; ++I)
      WL1.push_back(*I);
  }

  // We didn't hit a root? Return with a null pointer for the new graph.
  if (WL2.empty())
    return 0;

  // Create an empty graph.
  ExplodedGraph* G = MakeEmptyGraph();

  // ===- Pass 2 (forward DFS to construct the new graph) -===
  while (!WL2.empty()) {
    const ExplodedNode* N = WL2.back();
    WL2.pop_back();

    // Skip this node if we have already processed it.
    if (Pass2.find(N) != Pass2.end())
      continue;

    // Create the corresponding node in the new graph and record the mapping
    // from the old node to the new node.
    ExplodedNode* NewN = G->getNode(N->getLocation(), N->State, NULL);
    Pass2[N] = NewN;

    // Also record the reverse mapping from the new node to the old node.
    if (InverseMap) (*InverseMap)[NewN] = N;

    // If this node is a root, designate it as such in the graph.
    if (N->Preds.empty())
      G->addRoot(NewN);

    // In the case that some of the intended predecessors of NewN have already
    // been created, we should hook them up as predecessors.

    // Walk through the predecessors of 'N' and hook up their corresponding
    // nodes in the new graph (if any) to the freshly created node.
    for (ExplodedNode **I=N->Preds.begin(), **E=N->Preds.end(); I!=E; ++I) {
      Pass2Ty::iterator PI = Pass2.find(*I);
      if (PI == Pass2.end())
        continue;

      NewN->addPredecessor(PI->second, *G);
    }

    // In the case that some of the intended successors of NewN have already
    // been created, we should hook them up as successors.  Otherwise, enqueue
    // the new nodes from the original graph that should have nodes created
    // in the new graph.
    for (ExplodedNode **I=N->Succs.begin(), **E=N->Succs.end(); I!=E; ++I) {
      Pass2Ty::iterator PI = Pass2.find(*I);
      if (PI != Pass2.end()) {
        PI->second->addPredecessor(NewN, *G);
        continue;
      }

      // Enqueue nodes to the worklist that were marked during pass 1.
      if (Pass1.count(*I))
        WL2.push_back(*I);
    }

    // Finally, explictly mark all nodes without any successors as sinks.
    if (N->isSink())
      NewN->markAsSink();
  }

  return G;
}

ExplodedNode*
InterExplodedGraphMap::getMappedNode(const ExplodedNode* N) const {
  llvm::DenseMap<const ExplodedNode*, ExplodedNode*>::const_iterator I =
    M.find(N);

  return I == M.end() ? 0 : I->second;
}

