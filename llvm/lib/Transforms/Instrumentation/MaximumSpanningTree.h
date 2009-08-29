//===- llvm/Analysis/MaximumSpanningTree.h - Interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This module privides means for calculating a maximum spanning tree for the
// CFG of a function according to a given profile.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MAXIMUMSPANNINGTREE_H
#define LLVM_ANALYSIS_MAXIMUMSPANNINGTREE_H

#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

namespace llvm {
  class Function;

  class MaximumSpanningTree {
  public:
    typedef std::vector<ProfileInfo::Edge> MaxSpanTree;

  protected:
    MaxSpanTree MST;

  public:
    static char ID; // Class identification, replacement for typeinfo

    // MaxSpanTree() - Calculates a MST for a function according to a profile.
    // If inverted is true, all the edges *not* in the MST are returned. As a
    // special also all leaf edges of the MST are not included, this makes it
    // easier for the OptimalEdgeProfileInstrumentation to use this MST to do
    // an optimal profiling.
    MaximumSpanningTree(Function *F, ProfileInfo *PI, bool invert);
    virtual ~MaximumSpanningTree();

    virtual MaxSpanTree::iterator begin();
    virtual MaxSpanTree::iterator end();

    virtual void dump();
  };

} // End llvm namespace

#endif
