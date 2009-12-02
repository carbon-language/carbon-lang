//===- llvm/Analysis/MaximumSpanningTree.h - Interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This module privides means for calculating a maximum spanning tree for a
// given set of weighted edges. The type parameter T is the type of a node.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MAXIMUMSPANNINGTREE_H
#define LLVM_ANALYSIS_MAXIMUMSPANNINGTREE_H

#include "llvm/BasicBlock.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include <vector>
#include <algorithm>

namespace llvm {

  /// MaximumSpanningTree - A MST implementation.
  /// The type parameter T determines the type of the nodes of the graph.
  template <typename T>
  class MaximumSpanningTree {

    // A comparing class for comparing weighted edges.
    template <typename CT>
    struct EdgeWeightCompare {
      bool operator()(typename MaximumSpanningTree<CT>::EdgeWeight X, 
                      typename MaximumSpanningTree<CT>::EdgeWeight Y) const {
        if (X.second > Y.second) return true;
        if (X.second < Y.second) return false;
        if (const BasicBlock *BBX = dyn_cast<BasicBlock>(X.first.first)) {
          if (const BasicBlock *BBY = dyn_cast<BasicBlock>(Y.first.first)) {
            if (BBX->size() > BBY->size()) return true;
            if (BBX->size() < BBY->size()) return false;
          }
        }
        if (const BasicBlock *BBX = dyn_cast<BasicBlock>(X.first.second)) {
          if (const BasicBlock *BBY = dyn_cast<BasicBlock>(Y.first.second)) {
            if (BBX->size() > BBY->size()) return true;
            if (BBX->size() < BBY->size()) return false;
          }
        }
        return false;
      }
    };

  public:
    typedef std::pair<const T*, const T*> Edge;
    typedef std::pair<Edge, double> EdgeWeight;
    typedef std::vector<EdgeWeight> EdgeWeights;
  protected:
    typedef std::vector<Edge> MaxSpanTree;

    MaxSpanTree MST;

  public:
    static char ID; // Class identification, replacement for typeinfo

    /// MaximumSpanningTree() - Takes a vector of weighted edges and returns a
    /// spanning tree.
    MaximumSpanningTree(EdgeWeights &EdgeVector) {

      std::stable_sort(EdgeVector.begin(), EdgeVector.end(), EdgeWeightCompare<T>());

      // Create spanning tree, Forest contains a special data structure
      // that makes checking if two nodes are already in a common (sub-)tree
      // fast and cheap.
      EquivalenceClasses<const T*> Forest;
      for (typename EdgeWeights::iterator EWi = EdgeVector.begin(),
           EWe = EdgeVector.end(); EWi != EWe; ++EWi) {
        Edge e = (*EWi).first;

        Forest.insert(e.first);
        Forest.insert(e.second);
      }

      // Iterate over the sorted edges, biggest first.
      for (typename EdgeWeights::iterator EWi = EdgeVector.begin(),
           EWe = EdgeVector.end(); EWi != EWe; ++EWi) {
        Edge e = (*EWi).first;

        if (Forest.findLeader(e.first) != Forest.findLeader(e.second)) {
          Forest.unionSets(e.first, e.second);
          // So we know now that the edge is not already in a subtree, so we push
          // the edge to the MST.
          MST.push_back(e);
        }
      }
    }

    typename MaxSpanTree::iterator begin() {
      return MST.begin();
    }

    typename MaxSpanTree::iterator end() {
      return MST.end();
    }
  };

} // End llvm namespace

#endif
