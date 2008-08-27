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
#include <list>

using namespace clang;


static inline std::vector<ExplodedNodeImpl*>& getVector(void* P) {
  return *reinterpret_cast<std::vector<ExplodedNodeImpl*>*>(P);
}

void ExplodedNodeImpl::addPredecessor(ExplodedNodeImpl* V) {
  assert (!V->isSink());
  Preds.addNode(V);
  V->Succs.addNode(this);
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

ExplodedGraphImpl* ExplodedGraphImpl::Trim(ExplodedNodeImpl** BeginSources,
                                           ExplodedNodeImpl** EndSources) const{
  
  typedef llvm::DenseMap<ExplodedNodeImpl*, ExplodedNodeImpl*> Pass1Ty;
  typedef llvm::DenseMap<ExplodedNodeImpl*, ExplodedNodeImpl*> Pass2Ty;
  
  Pass1Ty Pass1;
  Pass2Ty Pass2;
  
  llvm::SmallVector<ExplodedNodeImpl*, 10> WL2;

  { // ===- Pass 1 (reverse BFS) -===
    
    // Enqueue the source nodes to the first worklist. 
    
    std::list<std::pair<ExplodedNodeImpl*, ExplodedNodeImpl*> > WL1;
    std::list<std::pair<ExplodedNodeImpl*, ExplodedNodeImpl*> > WL1_Loops;
  
    for (ExplodedNodeImpl** I = BeginSources; I != EndSources; ++I)
      WL1.push_back(std::make_pair(*I, *I));
    
    // Process the worklist.

    while (! (WL1.empty() && WL1_Loops.empty())) {
      
      ExplodedNodeImpl *N, *Src;

      // Only dequeue from the "loops" worklist if WL1 has no items.
      // Thus we prioritize for paths that don't span loop boundaries.
      
      if (WL1.empty()) {
        N = WL1_Loops.back().first;
        Src = WL1_Loops.back().second;
        WL1_Loops.pop_back();
      }
      else {
        N = WL1.back().first;
        Src = WL1.back().second;
        WL1.pop_back();
      }      
      
      if (Pass1.find(N) != Pass1.end())
        continue;
      
      bool PredHasSameSource = false;
      bool VisitPreds = true;
            
      for (ExplodedNodeImpl** I=N->Preds.begin(), **E=N->Preds.end();
            I!=E; ++I) {
        
        Pass1Ty::iterator pi = Pass1.find(*I);
        
        if (pi == Pass1.end())
          continue;
        
        VisitPreds = false;
        
        if (pi->second == Src) {
          PredHasSameSource = true;
          break;
        }
      }
      
      if (VisitPreds || !PredHasSameSource) {
        
        Pass1[N] = Src;
      
        if (N->Preds.empty()) {
          WL2.push_back(N);
          continue;      
        }
      }
      else
        Pass1[N] = NULL;
      
      if (VisitPreds)
        for (ExplodedNodeImpl** I=N->Preds.begin(), **E=N->Preds.end();
             I!=E; ++I) {
          
          ProgramPoint P = Src->getLocation();
          
          if (const BlockEdge *BE = dyn_cast<BlockEdge>(&P))
            if (Stmt* T = BE->getSrc()->getTerminator())
              switch (T->getStmtClass()) {
                default: break;
                case Stmt::ForStmtClass:
                case Stmt::WhileStmtClass:
                case Stmt::DoStmtClass:
                  WL1_Loops.push_front(std::make_pair(*I, Src));
                  continue;
                  
              }
          
          WL1.push_front(std::make_pair(*I, Src));
        }
    }
  }
  
  if (WL2.empty())
    return NULL;
    
  ExplodedGraphImpl* G = MakeEmptyGraph();
  
  // ===- Pass 2 (forward DFS to construct the new graph) -===
  
  while (!WL2.empty()) {
    
    ExplodedNodeImpl* N = WL2.back();
    WL2.pop_back();
    
    // Skip this node if we have already processed it.
    
    if (Pass2.find(N) != Pass2.end())
      continue;
    
    // Create the corresponding node in the new graph.
    
    ExplodedNodeImpl* NewN = G->getNodeImpl(N->getLocation(), N->State, NULL);
    Pass2[N] = NewN;
    
    if (N->Preds.empty())
      G->addRoot(NewN);
    
    // In the case that some of the intended predecessors of NewN have already
    // been created, we should hook them up as predecessors.
    
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
      
      Pass1Ty::iterator pi = Pass1.find(*I);
      
      if (pi == Pass1.end() || pi->second == NULL)
        continue;
            
      WL2.push_back(*I);
    }
    
    if (N->isSink())
      NewN->markAsSink();
  }
    
  return G;
}
