//===- MaximumSpanningTree.cpp - LLVM Pass to estimate profile info -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This module privides means for calculating a maximum spanning tree for the
// CFG of a function according to a given profile. The tree does not contain
// leaf edges, since they are needed for optimal edge profiling.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "maximum-spanning-tree"
#include "MaximumSpanningTree.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
using namespace llvm;

namespace {
  // compare two weighted edges
  struct VISIBILITY_HIDDEN EdgeWeightCompare {
    bool operator()(const ProfileInfo::EdgeWeight X, 
                    const ProfileInfo::EdgeWeight Y) const {
      if (X.second > Y.second) return true;
      if (X.second < Y.second) return false;

      // It would be enough to just compare the weights of the edges and be
      // done. With edges of the same weight this may lead to a different MST
      // each time the MST is created. To have more stable sorting (and thus
      // more stable MSTs) furhter sort the edges.
      if (X.first.first != 0 && Y.first.first == 0) return true;
      if (X.first.first == 0 && Y.first.first != 0) return false;
      if (X.first.first == 0 && Y.first.first == 0) return false;

      if (X.first.first->size() > Y.first.first->size()) return true;
      if (X.first.first->size() < Y.first.first->size()) return false;

      if (X.first.second != 0 && Y.first.second == 0) return true;
      if (X.first.second == 0 && Y.first.second != 0) return false;
      if (X.first.second == 0 && Y.first.second == 0) return false;

      if (X.first.second->size() > Y.first.second->size()) return true;
      if (X.first.second->size() < Y.first.second->size()) return false;

      return false;
    }
  };
}

static void inline printMSTEdge(ProfileInfo::EdgeWeight E, 
                                const char *M) {
  DEBUG(errs() << "--Edge " << E.first
               <<" (Weight "<< format("%g",E.second) << ") "
               << (M) << "\n");
}

// MaximumSpanningTree() - Takes a function and returns a spanning tree
// according to the currently active profiling information, the leaf edges are
// NOT in the MST. MaximumSpanningTree uses the algorithm of Kruskal.
MaximumSpanningTree::MaximumSpanningTree(std::vector<ProfileInfo::EdgeWeight>
                                         &EdgeVector) {

  std::sort(EdgeVector.begin(), EdgeVector.end(), EdgeWeightCompare());

  // Create spanning tree, Forest contains a special data structure
  // that makes checking if two nodes are already in a common (sub-)tree
  // fast and cheap.
  EquivalenceClasses<const BasicBlock*> Forest;
  for (std::vector<ProfileInfo::EdgeWeight>::iterator bbi = EdgeVector.begin(),
       bbe = EdgeVector.end(); bbi != bbe; ++bbi) {
    Forest.insert(bbi->first.first);
    Forest.insert(bbi->first.second);
  }
  Forest.insert(0);

  // Iterate over the sorted edges, biggest first.
  for (std::vector<ProfileInfo::EdgeWeight>::iterator bbi = EdgeVector.begin(),
       bbe = EdgeVector.end(); bbi != bbe; ++bbi) {
    ProfileInfo::Edge e = (*bbi).first;

    if (Forest.findLeader(e.first) != Forest.findLeader(e.second)) {
      Forest.unionSets(e.first, e.second);
      // So we know now that the edge is not already in a subtree (and not
      // (0,entry)), so we push the edge to the MST if it has some successors.
      MST.push_back(e);
      printMSTEdge(*bbi,"in MST");
    } else {
      // This edge is either (0,entry) or (BB,0) or would create a circle in a
      // subtree.
      printMSTEdge(*bbi,"*not* in MST");
    }
  }

  // Sort the MST edges.
  std::stable_sort(MST.begin(),MST.end());
}

MaximumSpanningTree::MaxSpanTree::iterator MaximumSpanningTree::begin() {
  return MST.begin();
}

MaximumSpanningTree::MaxSpanTree::iterator MaximumSpanningTree::end() {
  return MST.end();
}

void MaximumSpanningTree::dump() {
  errs()<<"{";
  for ( MaxSpanTree::iterator ei = MST.begin(), ee = MST.end();
        ei!=ee; ++ei ) {
    errs()<<"("<<((*ei).first?(*ei).first->getNameStr():"0")<<",";
    errs()<<(*ei).second->getNameStr()<<")";
  }
  errs()<<"}\n";
}
