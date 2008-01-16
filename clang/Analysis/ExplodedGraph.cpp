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
#include <vector>

using namespace clang;


static inline std::vector<ExplodedNodeImpl*>& getVector(void* P) {
  return *reinterpret_cast<std::vector<ExplodedNodeImpl*>*>(P);
}

void ExplodedNodeImpl::NodeGroup::addNode(ExplodedNodeImpl* N) {
  if (getKind() == Size1) {
    if (ExplodedNodeImpl* NOld = getNode()) {
      std::vector<ExplodedNodeImpl*>* V = new std::vector<ExplodedNodeImpl*>();
      V->push_back(NOld);
      V->push_back(N);
      P = reinterpret_cast<uintptr_t>(V) & SizeOther;
    }
    else
      P = reinterpret_cast<uintptr_t>(N);
  }
  else
    getVector(getPtr()).push_back(N);
}

bool ExplodedNodeImpl::NodeGroup::empty() const {
  if (getKind() == Size1)
    return getNode() ? false : true;
  else
    return getVector(getPtr()).empty();
}

unsigned ExplodedNodeImpl::NodeGroup::size() const {
  if (getKind() == Size1)
    return getNode() ? 1 : 0;
  else
    return getVector(getPtr()).size();
}

ExplodedNodeImpl** ExplodedNodeImpl::NodeGroup::begin() const {
  if (getKind() == Size1)
    return (ExplodedNodeImpl**) &P;
  else
    return const_cast<ExplodedNodeImpl**>(&*(getVector(getPtr()).begin()));
}

ExplodedNodeImpl** ExplodedNodeImpl::NodeGroup::end() const {
  if (getKind() == Size1)
    return (ExplodedNodeImpl**) (P ? &P+1 : &P);
  else
    return const_cast<ExplodedNodeImpl**>(&*(getVector(getPtr()).rbegin())+1);
}

ExplodedNodeImpl::NodeGroup::~NodeGroup() {
  if (getKind() == SizeOther) delete &getVector(getPtr());
}


ExplodedGraphImpl::~ExplodedGraphImpl() {
  // Delete the FoldingSet's in Nodes.  Note that the contents
  // of the FoldingSets are nodes allocated from the BumpPtrAllocator,
  // so all of those will get nuked when that object is destroyed.
  for (EdgeNodeSetMap::iterator I=Nodes.begin(), E=Nodes.end(); I!=E; ++I) {
    llvm::FoldingSet<ExplodedNodeImpl>* ENodes = 
      reinterpret_cast<llvm::FoldingSet<ExplodedNodeImpl>*>(I->second);
    
    for (llvm::FoldingSet<ExplodedNodeImpl>::iterator
         I=ENodes->begin(), E=ENodes->end(); I!=E; ++I)
      (*I).~ExplodedNodeImpl();
    
    delete ENodes;
  }
}
