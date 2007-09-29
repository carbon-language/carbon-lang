//=== llvm/Analysis/DominatorInternals.h - Dominator Calculation -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines shared implementation details of dominator and
// postdominator calculation.  This file SHOULD NOT BE INCLUDED outside
// of the dominator and postdominator implementation files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMINATOR_INTERNALS_H
#define LLVM_ANALYSIS_DOMINATOR_INTERNALS_H

#include "llvm/Analysis/Dominators.h"

namespace llvm {

template<class GraphT>
unsigned DFSPass(DominatorTreeBase& DT, typename GraphT::NodeType* V,
                 unsigned N) {
  // This is more understandable as a recursive algorithm, but we can't use the
  // recursive algorithm due to stack depth issues.  Keep it here for
  // documentation purposes.
#if 0
  InfoRec &VInfo = DT.Info[DT.Roots[i]];
  VInfo.Semi = ++N;
  VInfo.Label = V;

  Vertex.push_back(V);        // Vertex[n] = V;
  //Info[V].Ancestor = 0;     // Ancestor[n] = 0
  //Info[V].Child = 0;        // Child[v] = 0
  VInfo.Size = 1;             // Size[v] = 1

  for (succ_iterator SI = succ_begin(V), E = succ_end(V); SI != E; ++SI) {
    InfoRec &SuccVInfo = DT.Info[*SI];
    if (SuccVInfo.Semi == 0) {
      SuccVInfo.Parent = V;
      N = DTDFSPass(DT, *SI, N);
    }
  }
#else
  std::vector<std::pair<typename GraphT::NodeType*,
                        typename GraphT::ChildIteratorType> > Worklist;
  Worklist.push_back(std::make_pair(V, GraphT::child_begin(V)));
  while (!Worklist.empty()) {
    typename GraphT::NodeType* BB = Worklist.back().first;
    typename GraphT::ChildIteratorType NextSucc = Worklist.back().second;

    // First time we visited this BB?
    if (NextSucc == GraphT::child_begin(BB)) {
      DominatorTree::InfoRec &BBInfo = DT.Info[BB];
      BBInfo.Semi = ++N;
      BBInfo.Label = BB;

      DT.Vertex.push_back(BB);       // Vertex[n] = V;
      //BBInfo[V].Ancestor = 0;   // Ancestor[n] = 0
      //BBInfo[V].Child = 0;      // Child[v] = 0
      BBInfo.Size = 1;            // Size[v] = 1
    }
    
    // If we are done with this block, remove it from the worklist.
    if (NextSucc == GraphT::child_end(BB)) {
      Worklist.pop_back();
      continue;
    }

    // Increment the successor number for the next time we get to it.
    ++Worklist.back().second;
    
    // Visit the successor next, if it isn't already visited.
    typename GraphT::NodeType* Succ = *NextSucc;

    DominatorTree::InfoRec &SuccVInfo = DT.Info[Succ];
    if (SuccVInfo.Semi == 0) {
      SuccVInfo.Parent = BB;
      Worklist.push_back(std::make_pair(Succ, GraphT::child_begin(Succ)));
    }
  }
#endif
    return N;
}

}

#endif
