//===- ReorderAlgorithm.h - Interface for basic block reorderng algorithms ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to different basic block reordering algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_REORDER_ALGORITHM_H
#define LLVM_TOOLS_LLVM_BOLT_REORDER_ALGORITHM_H

#include "llvm/Support/ErrorHandling.h"
#include <unordered_map>
#include <memory>
#include <vector>


namespace llvm {
namespace bolt {


class BinaryBasicBlock;
class BinaryFunction;

/// Objects of this class implement various basic block clustering algorithms.
/// Basic block clusters are chains of basic blocks that should be laid out
/// in this order to maximize performace. These algorithms group basic blocks
/// into clusters using execution profile data and various heuristics.
class ClusterAlgorithm {
public:
  typedef std::vector<BinaryBasicBlock *> ClusterTy;
  std::vector<ClusterTy> Clusters;
  std::vector<std::unordered_map<uint32_t, uint64_t>> ClusterEdges;
  std::vector<double> AvgFreq;

  /// Group the basic blocks the given function into clusters stored in the
  /// Clusters vector. Also encode relative weights between two clusters in
  /// the ClusterEdges vector. This vector is indexed by the clusters indices
  /// in the Clusters vector.
  virtual void clusterBasicBlocks(const BinaryFunction &BF) =0;

  /// Compute for each cluster its averagae execution frequency, that is
  /// the sum of average frequencies of its blocks (execution count / # instrs).
  /// The average frequencies are stored in the AvgFreq vector, index by the
  /// cluster indices in the Clusters vector.
  void computeClusterAverageFrequency();

  /// Clear clusters and related info.
  void reset();

  void printClusters() const;

  virtual ~ClusterAlgorithm() { }
};


/// This clustering algorithm is based on a greedy heuristic suggested by
/// Pettis (PLDI '90).
class GreedyClusterAlgorithm : public ClusterAlgorithm {
public:
  void clusterBasicBlocks(const BinaryFunction &BF) override;
};

/// Objects of this class implement various basic block reordering alogrithms.
/// Most of these algorithms depend on a clustering alogrithm.
/// Here we have 3 conflicting goals as to how to layout clusters. If we want
/// to minimize jump offsets, we should put clusters with heavy inter-cluster
/// dependence as close as possible. If we want to maximize the probability
/// that all inter-cluster edges are predicted as not-taken, we should enforce
/// a topological order to make targets appear after sources, creating forward
/// branches. If we want to separate hot from cold blocks to maximize the
/// probability that unfrequently executed code doesn't pollute the cache, we
/// should put clusters in descending order of hotness.
class ReorderAlgorithm {
protected:
  std::unique_ptr<ClusterAlgorithm> CAlgo;

public:
  ReorderAlgorithm() { }
  explicit ReorderAlgorithm(std::unique_ptr<ClusterAlgorithm> CAlgo) :
    CAlgo(std::move(CAlgo)) { }

  typedef std::vector<BinaryBasicBlock *>  BasicBlockOrder;

  /// Reorder the basic blocks of the given function and store the new order in
  /// the new Clusters vector.
  virtual void reorderBasicBlocks(
      const BinaryFunction &BF, BasicBlockOrder &Order) const =0;

  void setClusterAlgorithm(ClusterAlgorithm *CAlgo) {
    this->CAlgo.reset(CAlgo);
  }

  virtual ~ReorderAlgorithm() { }
};


/// Dynamic programming implementation for the TSP, applied to BB layout. Find
/// the optimal way to maximize weight during a path traversing all BBs. In
/// this way, we will convert the hottest branches into fall-throughs.
///
/// Uses exponential amount of memory on the number of basic blocks and should
/// only be used for small functions.
class OptimalReorderAlgorithm : public ReorderAlgorithm {
public:
  void reorderBasicBlocks(
      const BinaryFunction &BF, BasicBlockOrder &Order) const override;
};


/// Simple algorithm that groups basic blocks into clusters and then
/// lays them out cluster after cluster.
class OptimizeReorderAlgorithm : public ReorderAlgorithm {
public:
  explicit OptimizeReorderAlgorithm(std::unique_ptr<ClusterAlgorithm> CAlgo) :
    ReorderAlgorithm(std::move(CAlgo)) { }

  void reorderBasicBlocks(
      const BinaryFunction &BF, BasicBlockOrder &Order) const override;
};


/// This reorder algorithm tries to ensure that all inter-cluster edges are
/// predicted as not-taken, by enforcing a topological order to make
/// targets appear after sources, creating forward branches.
class OptimizeBranchReorderAlgorithm : public ReorderAlgorithm {
public:
  explicit OptimizeBranchReorderAlgorithm(
      std::unique_ptr<ClusterAlgorithm> CAlgo) :
    ReorderAlgorithm(std::move(CAlgo)) { }

  void reorderBasicBlocks(
      const BinaryFunction &BF, BasicBlockOrder &Order) const override;
};


/// This reorder tries to separate hot from cold blocks to maximize the
/// probability that unfrequently executed code doesn't pollute the cache, by
/// putting clusters in descending order of hotness.
class OptimizeCacheReorderAlgorithm : public ReorderAlgorithm {
public:
  explicit OptimizeCacheReorderAlgorithm(
      std::unique_ptr<ClusterAlgorithm> CAlgo) :
    ReorderAlgorithm(std::move(CAlgo)) { }

  void reorderBasicBlocks(
      const BinaryFunction &BF, BasicBlockOrder &Order) const override;
};


/// Toy example that simply reverses the original basic block order.
class ReverseReorderAlgorithm : public ReorderAlgorithm {
public:
  void reorderBasicBlocks(
      const BinaryFunction &BF, BasicBlockOrder &Order) const override;
};


} // namespace bolt
} // namespace llvm

#endif

