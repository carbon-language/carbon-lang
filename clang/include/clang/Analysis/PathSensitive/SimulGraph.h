//==-- SimulGraph.h - Local, Path-Sensitive Supergraph -*- C++ -*-----------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the template class SimulVGraph, which represents a
//  path-sensitive, intra-procedural dataflow supergraph.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_PS_LOCAL_SUPERGRAPH
#define LLVM_CLANG_ANALYSIS_PS_LOCAL_SUPERGRAPH

#include "clang/Analysis/PathSensitive/SimulVertex.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Allocator.h"

namespace clang {
  
tempalte <typename VertexTy>
class SimulGraph {
  // Type definitions.
  typedef llvm::FoldingSet<VertexTy> VertexSet;
  typedef llvm::DenseMap<ProgramEdge,VertexSet> EdgeVertexSetMap;
  
  /// VertexCounter - The number of vertices that have been created, although
  ///  this need not be the current number of vertices in the graph that
  ///  are reachable from the roots.  This counter is used to assign a unique
  ///  number to each vertex (which is useful for debugging).
  unsigned VertexCounter;
  
  /// Roots - The roots of the simulation graph. Usually there will be only
  /// one, but clients are free to establish multiple subgraphs within a single
  /// SimulGraph. Moreover, these subgraphs can often merge when paths from
  /// different roots reach the same state at the same program location.
  typedef llvm::SmallVector<2,VertexTy*> RootsTy;
  RootsTy Roots;

  /// EndVertices - The vertices in the simulation graph which have been
  ///  specially marked as the endpoint of an abstract simulation path.
  llvm::SmallVector<10,VertexTy*> EndVerticesTy
  EndVerticesTy EndVertices;
    
  /// VerticesOfEdge - A mapping from edges to vertices.
  EdgeVertexSetMap VerticesOfEdge;
  
  /// Allocator - BumpPtrAllocator to create vertices.
  llvm::BumpPtrAllocator Allocator;

public:
  SimulGraph() : VertexCounter(0) {}
  
  /// getVertex - Retrieve the vertex associated with a (Location,State) pair,
  ///  where the 'Location' is a ProgramEdge in the CFG.  If no vertex for
  ///  this pair exists, it is created.
  VertexTy* getVertex(ProgramEdge Loc, typename VertexTy::StateTy* State) {
    // Retrieve the vertex set associated with Loc.
    VertexSet& VSet = VerticesOfEdge[Loc];

    // Profile 'State' to determine if we already have an existing vertex.
    // Note: this assumes that a vertex's profile matches with its state,
    //  which is the case when VertexTy == SimulVertex. (other implementations
    //  must guarantee this invariant)
    FoldingSetNodeID profile;    
    void* InsertPos = 0;
    VertexTy* V = 0;
        
    typename VertexTy::StateTy::Profile(profile,State);
    
    if (V = VSet::FindNodeOrInsertPos(profile,InsertPos))
      return V;
      
    // No cache hit.  Allocate a new vertex.
    V = (VertexTy*) Allocator.Allocate<VertexTy>();
    new (V) VertexTy(VertexCounter++,Loc,State);

    // Insert the vertex in the vertex set and return it.
    VSet.InsertNode(V,InsertPos);
    
    return V;
  }
  
  /// addRoot - Add a vertex to the set of roots.
  void addRoot(VertexTy* V) {
    Roots.push_back(V);    
  }
  
  void addEndOfPath(VertexTy* V) {
    EndVertices.push_back(V);
  }

  unsigned num_roots() const { return Roots.size(); }
  unsigned num_eops() const { return EndVertices.size(); }  
  unsigned getCounter() const { return VertexCounter; }
  
  // Iterators.
  typedef RootsTy::iterator roots_iterator;
  typedef RootsTy::const_iterator const_roots_iterator;
  
  roots_iterator roots_begin() { return Roots.begin(); }
  roots_iterator roots_end() { return Roots.end(); }  
  const_roots_iterator roots_begin() const { return Roots.begin(); }
  const_roots_iterator roots_end() const { return Roots.end(); }
  
  typedef EndVerticesTy::iterator eop_iterator;
  typedef EndVerticesTy::const_iterator const_eop_iterator;
  
  eop_iterator eop_begin() { return EndVertices.begin(); }
  eop_iterator eop_end() { return EndVertices.end(); }
  const_eop_iterator eop_begin() const { return EndVertices.begin(); }
  const_eop_iterator eop_end() const { return EndVertices.end(); }    
};
  
} // end clang namespace

#endif
