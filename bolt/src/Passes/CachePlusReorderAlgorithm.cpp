//===--- CachePlusReorderAlgorithm.cpp - Order basic blocks ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryBasicBlock.h"
#include "BinaryFunction.h"
#include "CacheMetrics.h"
#include "ReorderAlgorithm.h"
#include "ReorderUtils.h"
#include "llvm/Support/Options.h"

using namespace llvm;
using namespace bolt;
using EdgeList = std::vector<std::pair<BinaryBasicBlock *, uint64_t>>;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

cl::opt<unsigned>
ClusterSplitThreshold("cluster-split-threshold",
  cl::desc("The maximum size of a cluster to apply splitting"),
  cl::init(128),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

}

namespace llvm {
namespace bolt {

namespace {

// A cluster (ordered sequence) of basic blocks
class Cluster {
public:
  Cluster(BinaryBasicBlock *BB, uint64_t ExecutionCount_, uint64_t Size_)
  : Id(BB->getLayoutIndex()),
    IsEntry(BB->getLayoutIndex() == 0),
    ExecutionCount(ExecutionCount_),
    Size(Size_),
    Score(0) {
    Blocks.push_back(BB);
  }

  size_t id() const {
    return Id;
  }

  uint64_t size() const {
    return Size;
  }

  double density() const {
    return static_cast<double>(ExecutionCount) / Size;
  }

  bool isCold() const {
    return ExecutionCount == 0;
  }

  uint64_t executionCount() const {
    return ExecutionCount;
  }

  bool isEntryPoint() const {
    return IsEntry;
  }

  double score() const {
    return Score;
  }

  const std::vector<BinaryBasicBlock *> &blocks() const {
    return Blocks;
  }

  /// Update the list of basic blocks and aggregated cluster data
  void merge(const Cluster *Other,
             const std::vector<BinaryBasicBlock *> &MergedBlocks,
             double MergedScore) {
    Blocks = MergedBlocks;
    IsEntry |= Other->IsEntry;
    ExecutionCount += Other->ExecutionCount;
    Size += Other->Size;
    Score = MergedScore;
  }

  void clear() {
    Blocks.clear();
  }

private:
  std::vector<BinaryBasicBlock *> Blocks;
  size_t Id;
  bool IsEntry;
  uint64_t ExecutionCount;
  uint64_t Size;
  double Score;
};

using ClusterIter = std::vector<BinaryBasicBlock *>::const_iterator;

// A wrapper around three clusters of basic blocks; it is used to avoid extra
// instantiation of the vectors.
class MergedCluster {
public:
  MergedCluster(ClusterIter Begin1,
                ClusterIter End1,
                ClusterIter Begin2,
                ClusterIter End2,
                ClusterIter Begin3,
                ClusterIter End3)
  : Begin1(Begin1),
    End1(End1),
    Begin2(Begin2),
    End2(End2),
    Begin3(Begin3),
    End3(End3) {}

  template<typename F>
  void forEach(const F &Func) const {
    for (auto It = Begin1; It != End1; It++)
      Func(*It);
    for (auto It = Begin2; It != End2; It++)
      Func(*It);
    for (auto It = Begin3; It != End3; It++)
      Func(*It);
  }

  std::vector<BinaryBasicBlock *> getBlocks() const {
    std::vector<BinaryBasicBlock *> Result;
    Result.reserve(std::distance(Begin1, End1) +
                   std::distance(Begin2, End2) +
                   std::distance(Begin3, End3));
    Result.insert(Result.end(), Begin1, End1);
    Result.insert(Result.end(), Begin2, End2);
    Result.insert(Result.end(), Begin3, End3);
    return Result;
  }

  const BinaryBasicBlock *getFirstBlock() const {
    return *Begin1;
  }

private:
  ClusterIter Begin1;
  ClusterIter End1;
  ClusterIter Begin2;
  ClusterIter End2;
  ClusterIter Begin3;
  ClusterIter End3;
};

/// Deterministically compare clusters by their density in decreasing order
bool compareClusters(const Cluster *C1, const Cluster *C2) {
  // Original entry point to the front
  if (C1->isEntryPoint())
    return true;
  if (C2->isEntryPoint())
    return false;

  const double D1 = C1->density();
  const double D2 = C2->density();
  if (D1 != D2)
    return D1 > D2;

  // Making the order deterministic
  return C1->id() < C2->id();
}

/// Deterministically compare pairs of clusters
bool compareClusterPairs(const Cluster *A1, const Cluster *B1,
                         const Cluster *A2, const Cluster *B2) {
  const auto Samples1 = A1->executionCount() + B1->executionCount();
  const auto Samples2 = A2->executionCount() + B2->executionCount();
  if (Samples1 != Samples2)
    return Samples1 < Samples2;

  // Making the order deterministic
  if (A1 != A2)
    return A1->id() < A2->id();
  return B1->id() < B2->id();
}

} // end namespace anonymous

/// CachePlus - layout of basic blocks with i-cache optimization.
///
/// Similarly to OptimizeCacheReorderAlgorithm, this algorithm is a greedy
/// heuristic that works with clusters (ordered sequences) of basic blocks.
/// Initially all clusters are isolated basic blocks. On every iteration,
/// we pick a pair of clusters whose merging yields the biggest increase in
/// the ExtTSP metric (see CacheMetrics.cpp for exact implementation), which
/// models how i-cache "friendly" a specific cluster is. A pair of clusters
/// giving the maximum gain is merged into a new cluster. The procedure stops
/// when there is only one cluster left, or when merging does not increase
/// ExtTSP. In the latter case, the remaining clusters are sorted by density.
///
/// An important aspect is the way two clusters are merged. Unlike earlier
/// algorithms (e.g., OptimizeCacheReorderAlgorithm or Pettis-Hansen), two
/// clusters, X and Y, are first split into three, X1, X2, and Y. Then we
/// consider all possible ways of gluing the three clusters (e.g., X1YX2, X1X2Y,
/// X2X1Y, X2YX1, YX1X2, YX2X1) and choose the one producing the largest score.
/// This improves the quality of the final result (the search space is larger)
/// while keeping the implementation sufficiently fast.
class CachePlus {
public:
  CachePlus(const BinaryFunction &BF)
  : BF(BF),
    Adjacent(BF.layout_size()),
    Cache(BF.layout_size()) {
    initialize();
  }

  /// Run cache+ algorithm and return a basic block ordering
  std::vector<BinaryBasicBlock *> run() {
    // Pass 1: Merge blocks with their fallthrough successors
    mergeFallthroughs();

    // Pass 2: Merge pairs of clusters while improving the ExtTSP metric
    mergeClusterPairs();

    // Pass 3: Merge cold blocks to reduce code size
    mergeColdClusters();

    // Sorting clusters by density
    std::stable_sort(Clusters.begin(), Clusters.end(), compareClusters);

    // Collect the basic blocks in the order specified by their clusters
    std::vector<BinaryBasicBlock *> Result;
    Result.reserve(BF.layout_size());
    for (auto Cluster : Clusters) {
      Result.insert(Result.end(),
                    Cluster->blocks().begin(),
                    Cluster->blocks().end());
    }

    return Result;
  }

private:
  /// Initialize the set of active clusters, edges between blocks, and
  /// adjacency matrix.
  void initialize() {
    // Initialize indices of basic blocks
    size_t LayoutIndex = 0;
    for (auto BB : BF.layout()) {
      BB->setLayoutIndex(LayoutIndex);
      LayoutIndex++;
    }

    // Initialize edges for the blocks and compute their total in/out weights
    OutEdges = std::vector<EdgeList>(BF.layout_size());
    auto InWeight = std::vector<uint64_t>(BF.layout_size(), 0);
    auto OutWeight = std::vector<uint64_t>(BF.layout_size(), 0);
    for (auto BB : BF.layout()) {
      auto BI = BB->branch_info_begin();
      for (auto I : BB->successors()) {
        assert(BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
               "missing profile for a jump");
        if (I != BB && BI->Count > 0) {
          InWeight[I->getLayoutIndex()] += BI->Count;
          OutEdges[BB->getLayoutIndex()].push_back(std::make_pair(I, BI->Count));
          OutWeight[BB->getLayoutIndex()] += BI->Count;
        }
        ++BI;
      }
    }

    // Initialize execution count for every basic block, which is the
    // maximum over the sums of all in and out edge weights.
    // Also execution count of the entry point is set to at least 1
    auto ExecutionCounts = std::vector<uint64_t>(BF.layout_size(), 0);
    for (auto BB : BF.layout()) {
      uint64_t EC = BB->getKnownExecutionCount();
      EC = std::max(EC, InWeight[BB->getLayoutIndex()]);
      EC = std::max(EC, OutWeight[BB->getLayoutIndex()]);
      if (BB->getLayoutIndex() == 0)
        EC = std::max(EC, uint64_t(1));
      ExecutionCounts[BB->getLayoutIndex()] = EC;
    }

    // Initialize clusters
    Clusters.reserve(BF.layout_size());
    AllClusters.reserve(BF.layout_size());
    CurCluster.reserve(BF.layout_size());
    Size.reserve(BF.layout_size());
    for (auto BB : BF.layout()) {
      size_t Index = BB->getLayoutIndex();
      Size.push_back(std::max<uint64_t>(BB->estimateSize(), 1));
      AllClusters.emplace_back(BB, ExecutionCounts[Index], Size[Index]);
      Clusters.push_back(&AllClusters[Index]);
      CurCluster.push_back(&AllClusters[Index]);
    }

    // Initialize adjacency matrix
    Adjacent.initialize(Clusters);
    for (auto BB : BF.layout()) {
      auto BI = BB->branch_info_begin();
      for (auto I : BB->successors()) {
        if (BB != I && BI->Count > 0) {
          Adjacent.set(Clusters[BB->getLayoutIndex()],
                       Clusters[I->getLayoutIndex()]);
        }
        ++BI;
      }
    }

    // Initialize fallthrough successors
    findFallthroughBlocks(InWeight, OutWeight);
  }

  /// Merge blocks with their fallthrough successors.
  void mergeFallthroughs() {
    for (auto BB : BF.layout()) {
      if (FallthroughPred[BB->getLayoutIndex()] == nullptr &&
          FallthroughSucc[BB->getLayoutIndex()] != nullptr) {
        auto CurBB = BB;
        while (FallthroughSucc[CurBB->getLayoutIndex()] != nullptr) {
          const auto NextBB = FallthroughSucc[CurBB->getLayoutIndex()];
          mergeClusters(&AllClusters[BB->getLayoutIndex()],
                        &AllClusters[NextBB->getLayoutIndex()],
                        0);
          CurBB = NextBB;
        }
      }
    }
  }

  /// Merge pairs of clusters while improving the ExtTSP metric
  void mergeClusterPairs() {
    while (Clusters.size() > 1) {
      Cluster *BestClusterPred = nullptr;
      Cluster *BestClusterSucc = nullptr;
      std::pair<double, size_t> BestGain(-1, 0);
      for (auto ClusterPred : Clusters) {
        // Do not merge cold blocks
        if (ClusterPred->isCold())
          continue;

        // Get candidates for merging with the current cluster
        Adjacent.forAllAdjacent(
          ClusterPred,
          // Find the best candidate
          [&](Cluster *ClusterSucc) {
            assert(ClusterPred != ClusterSucc && "loop edges are not supported");
            assert(!ClusterSucc->isCold() && "cannot merge cold clusters");

            // Compute the gain of merging two clusters
            auto Gain = mergeGain(ClusterPred, ClusterSucc);
            if (Gain.first <= 0.0)
              return;

            // Breaking ties by density to make the hottest clusters be merged first
            if (Gain.first > BestGain.first ||
                (std::abs(Gain.first - BestGain.first) < 1e-8 &&
                 compareClusterPairs(ClusterPred,
                                     ClusterSucc,
                                     BestClusterPred,
                                     BestClusterSucc))) {
              BestGain = Gain;
              BestClusterPred = ClusterPred;
              BestClusterSucc = ClusterSucc;
            }
          });
      }

      // Stop merging when there is no improvement
      if (BestGain.first <= 0.0)
        break;

      // Merge the best pair of clusters
      mergeClusters(BestClusterPred, BestClusterSucc, BestGain.second);
    }
  }

  /// Merge cold blocks to reduce code size
  void mergeColdClusters() {
    for (auto SrcBB : BF.layout()) {
      // Iterating in reverse order to make sure original fall-trough jumps are
      // merged first
      for (auto Itr = SrcBB->succ_rbegin(); Itr != SrcBB->succ_rend(); ++Itr) {
        BinaryBasicBlock *DstBB = *Itr;
        auto SrcCluster = CurCluster[SrcBB->getLayoutIndex()];
        auto DstCluster = CurCluster[DstBB->getLayoutIndex()];
        if (SrcCluster != DstCluster && !DstCluster->isEntryPoint() &&
            SrcCluster->blocks().back() == SrcBB &&
            DstCluster->blocks().front() == DstBB) {
          mergeClusters(SrcCluster, DstCluster, 0);
        }
      }
    }
  }

  /// For a pair of blocks, A and B, block B is the fallthrough successor of A,
  /// if (i) all jumps (based on profile) from A goes to B and (ii) all jumps
  /// to B are from A. Such blocks should be adjacent in an optimal ordering,
  /// and the method finds such pairs of blocks.
  void findFallthroughBlocks(const std::vector<uint64_t> &InWeight,
                             const std::vector<uint64_t> &OutWeight) {
    FallthroughSucc = std::vector<BinaryBasicBlock *>(BF.size(), nullptr);
    FallthroughPred = std::vector<BinaryBasicBlock *>(BF.size(), nullptr);
    // Find fallthroughs based on edge weights
    for (auto BB : BF.layout()) {
      if (BB->succ_size() == 1 &&
          BB->getSuccessor()->pred_size() == 1 &&
          BB->getSuccessor()->getLayoutIndex() != 0) {
        FallthroughSucc[BB->getLayoutIndex()] = BB->getSuccessor();
        FallthroughPred[BB->getSuccessor()->getLayoutIndex()] = BB;
        continue;
      }

      if (OutWeight[BB->getLayoutIndex()] == 0)
        continue;
      for (auto Edge : OutEdges[BB->getLayoutIndex()]) {
        const auto SuccBB = Edge.first;
        // Successor cannot be the first BB, which is pinned
        if (OutWeight[BB->getLayoutIndex()] == Edge.second &&
            InWeight[SuccBB->getLayoutIndex()] == Edge.second &&
            SuccBB->getLayoutIndex() != 0) {
          FallthroughSucc[BB->getLayoutIndex()] = SuccBB;
          FallthroughPred[SuccBB->getLayoutIndex()] = BB;
          break;
        }
      }
    }

    // There might be 'cycles' in the fallthrough dependencies (since profile
    // data isn't 100% accurate).
    // Break the cycles by choosing the block with smallest index as the tail
    for (auto BB : BF.layout()) {
      const auto Idx = BB->getLayoutIndex();
      if (FallthroughSucc[Idx] == nullptr || FallthroughPred[Idx] == nullptr)
        continue;

      auto SuccBB = FallthroughSucc[Idx];
      while (SuccBB != nullptr && SuccBB != BB) {
        SuccBB = FallthroughSucc[SuccBB->getLayoutIndex()];
      }
      if (SuccBB == nullptr)
        continue;
      // break the cycle
      FallthroughSucc[FallthroughPred[Idx]->getLayoutIndex()] = nullptr;
      FallthroughPred[Idx] = nullptr;
    }
  }

  /// Compute ExtTSP score for a given order of basic blocks
  double score(const MergedCluster& MergedBlocks) const {
    uint64_t NotSet = static_cast<uint64_t>(-1);
    EstimatedAddr.assign(BF.layout_size(), NotSet);

    uint64_t CurAddr = 0;
    MergedBlocks.forEach(
      [&](const BinaryBasicBlock *BB) {
        size_t Index = BB->getLayoutIndex();
        EstimatedAddr[Index] = CurAddr;
        CurAddr += Size[Index];
      }
    );

    double Score = 0;
    MergedBlocks.forEach(
      [&](const BinaryBasicBlock *BB) {
        size_t Index = BB->getLayoutIndex();
        for (auto Edge : OutEdges[Index]) {
          auto SuccBB = Edge.first;
          size_t SuccIndex = SuccBB->getLayoutIndex();

          if (EstimatedAddr[SuccIndex] != NotSet) {
            Score += CacheMetrics::extTSPScore(EstimatedAddr[Index],
                                               Size[Index],
                                               EstimatedAddr[SuccIndex],
                                               Edge.second);
          }
        }
      }
    );

    return Score;
  }

  /// Verify if it is valid to merge two clusters into the new one
  bool isValidMerge(const Cluster *ClusterPred,
                    const Cluster *ClusterSucc,
                    size_t MergeType,
                    const MergedCluster& MergedBlocks) const {
    // Does the new cluster preserve the original entry point?
    if ((ClusterPred->isEntryPoint() || ClusterSucc->isEntryPoint()) &&
        MergedBlocks.getFirstBlock()->getLayoutIndex() != 0)
      return false;

    // This corresponds to a concatentation of clusters w/o splitting, which is
    // always safe
    if (MergeType == 0)
      return true;

    size_t Offset = MergeType / 5;
    // The basic blocks on the boundary of a split of ClusterPred
    auto BB1 = ClusterPred->blocks()[Offset - 1];
    auto BB2 = ClusterPred->blocks()[Offset];
    // Does the splitting break FT successors?
    if (FallthroughSucc[BB1->getLayoutIndex()] != nullptr) {
      assert(FallthroughSucc[BB1->getLayoutIndex()] == BB2 &&
             "Fallthrough successor is not preserved");
      return false;
    }

    // Do not split large clusters to reduce computation time
    if (ClusterPred->blocks().size() > opts::ClusterSplitThreshold) {
      return false;
    }

    return true;
  }

  /// The gain of merging two clusters.
  ///
  /// The function considers all possible ways of merging two clusters and
  /// computes the one having the largest increase in ExtTSP metric. The result
  /// is a pair with the first element being the gain and the second element being
  /// the corresponding merging type (encoded as an integer).
  std::pair<double, size_t> mergeGain(const Cluster *ClusterPred,
                                      const Cluster *ClusterSucc) const {
    if (Cache.contains(ClusterPred, ClusterSucc)) {
      return Cache.get(ClusterPred, ClusterSucc);
    }

    // The current score of two separate clusters
    const auto CurScore = ClusterPred->score() + ClusterSucc->score();

    // Merge two clusters and update the best Gain
    auto computeMergeGain = [&](const std::pair<double, size_t> &CurGain,
                                const Cluster *ClusterPred,
                                const Cluster *ClusterSucc,
                                size_t MergeType) {
      auto MergedBlocks = mergeBlocks(ClusterPred->blocks(),
                                      ClusterSucc->blocks(),
                                      MergeType);

      if (!isValidMerge(ClusterPred, ClusterSucc, MergeType, MergedBlocks))
        return CurGain;

      // The score of the new cluster
      const auto NewScore = score(MergedBlocks);
      if (NewScore > CurScore && NewScore - CurScore > CurGain.first)
        return std::make_pair(NewScore - CurScore, MergeType);
      else
        return CurGain;
    };

    std::pair<double, size_t> Gain = std::make_pair(-1, 0);
    // Try to concatenate two clusters w/o splitting
    Gain = computeMergeGain(Gain, ClusterPred, ClusterSucc, 0);
    // Try to split ClusterPred into two sub-clusters in various ways and then
    // merge it with ClusterSucc
    for (size_t Offset = 1; Offset < ClusterPred->blocks().size(); Offset++) {
      for (size_t Type = 1; Type <= 4; Type++) {
        size_t MergeType = Type + Offset * 5;
        Gain = computeMergeGain(Gain, ClusterPred, ClusterSucc, MergeType);
      }
    }

    Cache.set(ClusterPred, ClusterSucc, Gain);
    return Gain;
  }

  /// Merge two clusters of blocks respecting a given merge 'type' and 'offset'.
  ///
  /// If MergeType == 0, then the result is a concatentation of two clusters.
  /// Otherwise, the first cluster is cut into two sub-clusters at the offset,
  /// and merged using all possible ways of concatenating three clusters.
  MergedCluster mergeBlocks(const std::vector<BinaryBasicBlock *> &X,
                            const std::vector<BinaryBasicBlock *> &Y,
                            size_t MergeType) const {
    // Merging w/o splitting existing clusters
    if (MergeType == 0) {
      ClusterIter Empty;
      return MergedCluster(X.begin(), X.end(), Y.begin(), Y.end(), Empty, Empty);
    }

    size_t Type = MergeType % 5;
    size_t Offset = MergeType / 5;
    assert(0 < Offset && Offset < X.size() &&
           "Invalid offset while merging clusters");
    // Split the first cluster, X, into X1 and X2
    ClusterIter BeginX1 = X.begin();
    ClusterIter EndX1 = X.begin() + Offset;
    ClusterIter BeginX2 = X.begin() + Offset;
    ClusterIter EndX2 = X.end();
    ClusterIter BeginY = Y.begin();
    ClusterIter EndY = Y.end();

    // Construct a new cluster from three existing ones
    switch(Type) {
    case 1: return MergedCluster(BeginX1, EndX1, BeginY, EndY, BeginX2, EndX2);
    case 2: return MergedCluster(BeginY, EndY, BeginX2, EndX2, BeginX1, EndX1);
    case 3: return MergedCluster(BeginX2, EndX2, BeginY, EndY, BeginX1, EndX1);
    case 4: return MergedCluster(BeginX2, EndX2, BeginX1, EndX1, BeginY, EndY);
    default:
      llvm_unreachable("unexpected merge type");
    }
  }

  /// Merge cluster From into cluster Into, update the list of active clusters,
  /// adjacency information, and the corresponding cache.
  void mergeClusters(Cluster *Into, Cluster *From, size_t MergeType) {
    assert(Into != From && "Cluster cannot be merged with itself");

    // Merge the blocks of clusters
    auto MergedBlocks = mergeBlocks(Into->blocks(), From->blocks(), MergeType);
    Into->merge(From, MergedBlocks.getBlocks(), score(MergedBlocks));
    From->clear();

    // Remove cluster From from the list of active clusters
    auto Iter = std::remove(Clusters.begin(), Clusters.end(), From);
    Clusters.erase(Iter, Clusters.end());

    // Update block clusters
    for (auto BB : Into->blocks()) {
      CurCluster[BB->getLayoutIndex()] = Into;
    }

    // Invalidate caches
    Cache.invalidate(Into);

    // Update the adjacency matrix
    Adjacent.merge(Into, From);
  }

  // The binary function
  const BinaryFunction &BF;

  // All clusters
  std::vector<Cluster> AllClusters;

  // Active clusters. The vector gets udpated at runtime when clusters are merged
  std::vector<Cluster *> Clusters;

  // Current cluster of a basic block
  std::vector<Cluster *> CurCluster;

  // Size of the block
  std::vector<uint64_t> Size;

  // Outgoing edges of the block
  std::vector<EdgeList> OutEdges;

  // Cluster adjacency matrix
  AdjacencyMatrix<Cluster> Adjacent;

  // Fallthrough successor of the block
  std::vector<BinaryBasicBlock *> FallthroughSucc;
  // Fallthrough predecessor of the block
  std::vector<BinaryBasicBlock *> FallthroughPred;

  // A cache that keeps precomputed values of mergeGain for pairs of clusters;
  // when a pair of clusters (x,y) gets merged, we invalidate the pairs
  // containing both x and y and all clusters adjacent to x and y (and recompute
  // them on the next iteration).
  mutable ClusterPairCache<Cluster, std::pair<double, size_t>> Cache;

  // A reusable vector used within score() method
  mutable std::vector<uint64_t> EstimatedAddr;
};

void CachePlusReorderAlgorithm::reorderBasicBlocks(
      const BinaryFunction &BF, BasicBlockOrder &Order) const {
  if (BF.layout_empty())
    return;

  // Are there jumps with positive execution count?
  size_t NumHotBlocks = 0;
  for (auto BB : BF.layout()) {
    if (BB->getKnownExecutionCount() > 0)
      NumHotBlocks++;
  }

  // Do not change layout of functions w/o profile information
  if (NumHotBlocks == 0 || BF.layout_size() <= 1) {
    for (auto BB : BF.layout()) {
      Order.push_back(BB);
    }
    return;
  }

  // Apply the algorithm
  Order = CachePlus(BF).run();

  // Verify correctness
  assert(Order[0]->isEntryPoint() && "Original entry point is not preserved");
  assert(Order.size() == BF.layout_size() && "Wrong size of reordered layout");
}

} // namespace bolt
} // namespace llvm
