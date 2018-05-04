// Passes/ReorderUtils.h - Helper methods for function and block reordering   //
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_REORDER_UTILS_H
#define LLVM_TOOLS_LLVM_BOLT_PASSES_REORDER_UTILS_H

#include <memory>
#include <vector>

#include "llvm/ADT/BitVector.h"

namespace llvm {
namespace bolt {

// This class maintains adjacency information for all Clusters being
// processed. It is used for visiting all neighbors of any given Cluster
// while merging pairs of Clusters. Every Cluster must implement the id() method
template <typename Cluster> class AdjacencyMatrix {
public:
  explicit AdjacencyMatrix(size_t Size) : Bits(Size, BitVector(Size, false)) {}

  void initialize(std::vector<Cluster *> &_Clusters) { Clusters = _Clusters; }

  template <typename F> void forAllAdjacent(const Cluster *C, F Func) const {
    const_cast<AdjacencyMatrix *>(this)->forallAdjacent(C, Func);
  }

  template <typename F> void forAllAdjacent(const Cluster *C, F Func) {
    for (auto I = Bits[C->id()].find_first(); I != -1;
         I = Bits[C->id()].find_next(I)) {
      Func(Clusters[I]);
    }
  }

  /// Merge adjacency info from cluster B into cluster A.  Info for cluster B is
  /// left in an undefined state.
  void merge(const Cluster *A, const Cluster *B) {
    Bits[A->id()] |= Bits[B->id()];
    Bits[A->id()][A->id()] = false;
    Bits[A->id()][B->id()] = false;
    Bits[B->id()][A->id()] = false;
    for (auto I = Bits[B->id()].find_first(); I != -1;
         I = Bits[B->id()].find_next(I)) {
      Bits[I][A->id()] = true;
      Bits[I][B->id()] = false;
    }
  }

  void set(const Cluster *A, const Cluster *B) { set(A, B, true); }

private:
  void set(const Cluster *A, const Cluster *B, bool Value) {
    assert(A != B);
    Bits[A->id()][B->id()] = Value;
    Bits[B->id()][A->id()] = Value;
  }

  std::vector<Cluster *> Clusters;
  std::vector<BitVector> Bits;
};

// This class holds cached results of specified type for a pair of Clusters.
// It can invalidate all cache entries associated with a given Cluster.
template <typename Cluster, typename ValueType> class ClusterPairCache {
public:
  explicit ClusterPairCache(size_t Size)
      : Size(Size), Cache(Size * Size), Valid(Size * Size, false) {}

  bool contains(const Cluster *First, const Cluster *Second) const {
    return Valid[index(First, Second)];
  }

  ValueType get(const Cluster *First, const Cluster *Second) const {
    assert(contains(First, Second));
    return Cache[index(First, Second)];
  }

  void set(const Cluster *First, const Cluster *Second, ValueType Value) {
    const auto Index = index(First, Second);
    Cache[Index] = Value;
    Valid[Index] = true;
  }

  void invalidate(const Cluster *C) {
    Valid.reset(C->id() * Size, (C->id() + 1) * Size);
    for (size_t id = 0; id < Size; id++) {
      Valid.reset((id * Size) + C->id());
    }
  }

private:
  size_t index(const Cluster *First, const Cluster *Second) const {
    return (First->id() * Size) + Second->id();
  }

  size_t Size;
  std::vector<ValueType> Cache;
  BitVector Valid;
};

} // namespace bolt
} // namespace llvm

#endif
