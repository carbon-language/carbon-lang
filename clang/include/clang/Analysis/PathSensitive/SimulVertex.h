//=- SimulVertex.h - Local, Path-Sensitive Supergraph Vertices -*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the template class SimulVertex which is used to
//  represent a vertex in the location*state supergraph of an intra-procedural,
//  path-sensitive dataflow analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_PS_ANALYSISVERTEX
#define LLVM_CLANG_ANALYSIS_PS_ANALYSISVERTEX

#include "clang/Analysis/ProgramEdge.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/FoldingSet.h"

namespace clang {
 
// *** NOTE ***
//  In the future we may not use a fully templated-implementation.  We'll first
//  see how the needed functionality develops and then refactor later.
  
template <typename StateType>
class SimulVertex : public llvm::FoldingSetNode {
public:
  typedef StateType StateTy;

private:  
  /// VertexID - A unique ID for the vertex.  This number indicates the
  ///  creation order of vertices, with lower numbers being created first.
  ///  The first created vertex has VertexID == 0.
  const unsigned VertexID;
  
  /// Location - The program edge representing the location in the function body
  ///  that this vertex corresponds to.
  const ProgramEdge Location;
  
  /// State - The state associated with this vertex. Normally this value
  ///  is immutable, but we anticipate there will be times when algorithms
  ///  that directly manipulate the analysis graph will need to change it.
  StateTy State;

  /// Predecessors/Successors - Keep track of the predecessor/successor
  /// vertices.
  
  // FIXME: Preds and Succs only grows, not shrinks.  It would be nice
  //  if these lists were allocated from the same BumpPtrAllocator as
  //  the vertices themselves.
  typedef llvm::SmallVector<SimulVertex*,2> AdjacentVertices;
  AdjacentVertices Preds;
  AdjacentVertices Succs;

public:
  explicit SimulVertex(unsigned ID, const ProgramEdge& loc, StateTy state)
    : VertexID(ID), Location(loc), State(state) {}
    
  // Accessors.
  const StateTy& getState() const { return State; }
  const ProgramEdge& getLocation() const { return Location; }
  unsigned getVertexID() const { return VertexID; }
  
  // Profiling (for FoldingSet).
  void Profile(llvm::FoldingSetNodeID& ID) const {
    StateTy::Profile(ID,getState());
  }

  // Iterators over successor and predecessor vertices.
  typedef typename AdjacentVertices::iterator        succ_iterator;
  typedef typename AdjacentVertices::const_iterator  const_succ_iterator;
  typedef typename AdjacentVertices::iterator        pred_iterator;
  typedef typename AdjacentVertices::const_iterator  const_pred_iterator;
  
  pred_iterator pred_begin() { return Preds.begin(); }
  pred_iterator pred_end() { return Preds.end(); }  
  const_pred_iterator pred_begin() const { return Preds.begin(); }
  const_pred_iterator pred_end() const { return Preds.end(); }
  
  succ_iterator succ_begin() { return Succs.begin(); }
  succ_iterator succ_end() { return Succs.end(); }  
  const_succ_iterator succ_begin() const { return Succs.begin(); }
  const_succ_iterator succ_end() const { return Succs.end(); }

  unsigned succ_size() const { return Succs.size(); }
  bool succ_empty() const { return Succs.empty(); }
  
  unsigned pred_size() const { return Preds.size(); }
  unsigned pred_empty() const { return Preds.empty(); }
  
  // Manipulation of successors/predecessors.
  void addPredecessor(SimulVertex* V) {
    Preds.push_back(V);
    V->Succs.push_back(V);
  }
};
  
} // end namespace clang

#endif
