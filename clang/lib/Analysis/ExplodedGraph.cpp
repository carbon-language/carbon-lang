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

#include "clang/Analysis/PathSensitive/ExplodedGraph.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

using namespace clang;

//===----------------------------------------------------------------------===//
// Node auditing.
//===----------------------------------------------------------------------===//

// An out of line virtual method to provide a home for the class vtable.
ExplodedNodeImpl::Auditor::~Auditor() {}

#ifndef NDEBUG
static ExplodedNodeImpl::Auditor* NodeAuditor = 0;
#endif

void ExplodedNodeImpl::SetAuditor(ExplodedNodeImpl::Auditor* A) {
#ifndef NDEBUG
  NodeAuditor = A;
#endif
}

//===----------------------------------------------------------------------===//
// ExplodedNodeImpl.
//===----------------------------------------------------------------------===//

static inline std::vector<ExplodedNodeImpl*>& getVector(void* P) {
  return *reinterpret_cast<std::vector<ExplodedNodeImpl*>*>(P);
}

void ExplodedNodeImpl::addPredecessor(ExplodedNodeImpl* V) {
  assert (!V->isSink());
  Preds.addNode(V);
  V->Succs.addNode(this);
#ifndef NDEBUG
  if (NodeAuditor) NodeAuditor->AddEdge(V, this);
#endif
}

void ExplodedNodeImpl::NodeGroup::addNode(ExplodedNodeImpl* N) {
  
  assert ((reinterpret_cast<uintptr_t>(N) & Mask) == 0x0);
  assert (!getFlag());
  
  if (getKind() == Size1) {
    if (ExplodedNodeImpl* NOld = getNode()) {
      std::vector<ExplodedNodeImpl*>* V = new std::vector<ExplodedNodeImpl*>();
      assert ((reinterpret_cast<uintptr_t>(V) & Mask) == 0x0);
      V->push_back(NOld);
      V->push_back(N);
      P = reinterpret_cast<uintptr_t>(V) | SizeOther;
      assert (getPtr() == (void*) V);
      assert (getKind() == SizeOther);
    }
    else {
      P = reinterpret_cast<uintptr_t>(N);
      assert (getKind() == Size1);
    }
  }
  else {
    assert (getKind() == SizeOther);
    getVector(getPtr()).push_back(N);
  }
}


unsigned ExplodedNodeImpl::NodeGroup::size() const {
  if (getFlag())
    return 0;
  
  if (getKind() == Size1)
    return getNode() ? 1 : 0;
  else
    return getVector(getPtr()).size();
}

ExplodedNodeImpl** ExplodedNodeImpl::NodeGroup::begin() const {
  if (getFlag())
    return NULL;
  
  if (getKind() == Size1)
    return (ExplodedNodeImpl**) (getPtr() ? &P : NULL);
  else
    return const_cast<ExplodedNodeImpl**>(&*(getVector(getPtr()).begin()));
}

ExplodedNodeImpl** ExplodedNodeImpl::NodeGroup::end() const {
  if (getFlag())
    return NULL;
  
  if (getKind() == Size1)
    return (ExplodedNodeImpl**) (getPtr() ? &P+1 : NULL);
  else {
    // Dereferencing end() is undefined behaviour. The vector is not empty, so
    // we can dereference the last elem and then add 1 to the result.
    return const_cast<ExplodedNodeImpl**>(&getVector(getPtr()).back()) + 1;
  }
}

ExplodedNodeImpl::NodeGroup::~NodeGroup() {
  if (getKind() == SizeOther) delete &getVector(getPtr());
}

ExplodedGraphImpl*
ExplodedGraphImpl::Trim(const ExplodedNodeImpl* const* BeginSources,
                        const ExplodedNodeImpl* const* EndSources,
                        InterExplodedGraphMapImpl* M,
                        llvm::DenseMap<const void*, const void*> *InverseMap) 
const {
  
  typedef llvm::DenseSet<const ExplodedNodeImpl*> Pass1Ty;
  Pass1Ty Pass1;
  
  typedef llvm::DenseMap<const ExplodedNodeImpl*, ExplodedNodeImpl*> Pass2Ty;
  Pass2Ty& Pass2 = M->M;
  
  llvm::SmallVector<const ExplodedNodeImpl*, 10> WL1, WL2;

  // ===- Pass 1 (reverse DFS) -===
  for (const ExplodedNodeImpl* const* I = BeginSources; I != EndSources; ++I) {
    assert(*I);
    WL1.push_back(*I);
  }
    
  // Process the first worklist until it is empty.  Because it is a std::list
  // it acts like a FIFO queue.
  while (!WL1.empty()) {
    const ExplodedNodeImpl *N = WL1.back();
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
    for (ExplodedNodeImpl** I=N->Preds.begin(), **E=N->Preds.end(); I!=E; ++I)
      WL1.push_back(*I);
  }
  
  // We didn't hit a root? Return with a null pointer for the new graph.
  if (WL2.empty())
    return 0;

  // Create an empty graph.
  ExplodedGraphImpl* G = MakeEmptyGraph();
  
  // ===- Pass 2 (forward DFS to construct the new graph) -===  
  while (!WL2.empty()) {
    const ExplodedNodeImpl* N = WL2.back();
    WL2.pop_back();
    
    // Skip this node if we have already processed it.
    if (Pass2.find(N) != Pass2.end())
      continue;
    
    // Create the corresponding node in the new graph and record the mapping
    // from the old node to the new node.
    ExplodedNodeImpl* NewN = G->getNodeImpl(N->getLocation(), N->State, NULL);
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
    for (ExplodedNodeImpl **I=N->Preds.begin(), **E=N->Preds.end(); I!=E; ++I) {
      Pass2Ty::iterator PI = Pass2.find(*I);
      if (PI == Pass2.end())
        continue;
      
      NewN->addPredecessor(PI->second);
    }

    // In the case that some of the intended successors of NewN have already
    // been created, we should hook them up as successors.  Otherwise, enqueue
    // the new nodes from the original graph that should have nodes created
    // in the new graph.
    for (ExplodedNodeImpl **I=N->Succs.begin(), **E=N->Succs.end(); I!=E; ++I) {
      Pass2Ty::iterator PI = Pass2.find(*I);      
      if (PI != Pass2.end()) {
        PI->second->addPredecessor(NewN);
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

ExplodedNodeImpl*
InterExplodedGraphMapImpl::getMappedImplNode(const ExplodedNodeImpl* N) const {
  llvm::DenseMap<const ExplodedNodeImpl*, ExplodedNodeImpl*>::iterator I =
    M.find(N);

  return I == M.end() ? 0 : I->second;
}

InterExplodedGraphMapImpl::InterExplodedGraphMapImpl() {}

