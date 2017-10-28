//===--- HFSortPlus.cpp - Cluster functions by hotness --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

// TODO: copyright/license msg.

/*
   +----------------------------------------------------------------------+
   | HipHop for PHP                                                       |
   +----------------------------------------------------------------------+
   | Copyright (c) 2010-present Facebook, Inc. (http://www.facebook.com)  |
   +----------------------------------------------------------------------+
   | This source file is subject to version 3.01 of the PHP license,      |
   | that is bundled with this package in the file LICENSE, and is        |
   | available through the world-wide-web at the following url:           |
   | http://www.php.net/license/3_01.txt                                  |
   | If you did not receive a copy of the PHP license and are unable to   |
   | obtain it through the world-wide-web, please send a note to          |
   | license@php.net so we can mail you a copy immediately.               |
   +----------------------------------------------------------------------+
*/

#include "BinaryFunction.h"
#include "HFSort.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Options.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "hfsort"

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::OptionCategory BoltCategory;
extern cl::OptionCategory BoltOptCategory;
extern cl::opt<bool> Verbosity;

cl::opt<unsigned>
ITLBPageSizeParam("itlb-page-size",
  cl::desc("The size of i-tlb cache page"),
  cl::init(4096),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
ITLBEntriesParam("itlb-entries",
  cl::desc("The number of entries in i-tlb cache"),
  cl::init(16),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<double>
MergeProbability("merge-probability",
  cl::desc("The minimum probability of a call for merging two clusters"),
  cl::init(0.99),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

}

namespace llvm {
namespace bolt {

using NodeId = CallGraph::NodeId;
using Arc = CallGraph::Arc;
using Node = CallGraph::Node;

namespace {

constexpr size_t InvalidAddr = -1;

// The size of a cache page: Since we optimize both for i-TLB cache (2MB pages)
// and i-cache (64b pages), using a value that fits both
int32_t ITLBPageSize;

// Capacity of the iTLB cache: Larger values yield more iTLB-friendly result,
// while smaller values result in better i-cache performance
int32_t ITLBEntries;

const char* cacheKindString(bool UseGainCache, bool UseShortCallCache) {
  if (UseGainCache && UseShortCallCache)
    return "gain + short call cache";
  else if (UseGainCache)
    return "gain cache";
  else if (UseShortCallCache)
    return "short call cache";
  else
    return "no cache";
}

// This class maintains adjacency information for all Clusters being
// processed.  It is used to invalidate cache entries when merging
// Clusters and for visiting all neighbors of any given Cluster.
class AdjacencyMatrix {
 public:
  AdjacencyMatrix(const CallGraph &Cg,
                  std::vector<Cluster *> &Clusters,
                  const std::vector<Cluster *> &FuncCluster)
  : Clusters(Clusters),
    Bits(Cg.numNodes(), BitVector(Cg.numNodes(), false)) {
    initialize(Cg, FuncCluster);
  }

  template <typename F>
  void forallAdjacent(const Cluster *C, F Func) const {
    const_cast<AdjacencyMatrix *>(this)->forallAdjacent(C, Func);
  }

  template <typename F>
  void forallAdjacent(const Cluster *C, F Func) {
    for (auto I = Bits[C->id()].find_first(); I != -1; I = Bits[C->id()].find_next(I)) {
      Func(Clusters[I]);
    }
  }

  // Merge adjacency info from cluster B into cluster A.  Info for cluster B is left
  // in an undefined state.
  void merge(const Cluster *A, const Cluster *B) {
    Bits[A->id()] |= Bits[B->id()];
    Bits[A->id()][A->id()] = false;
    Bits[A->id()][B->id()] = false;
    Bits[B->id()][A->id()] = false;
    for (auto I = Bits[B->id()].find_first(); I != -1; I = Bits[B->id()].find_next(I)) {
      Bits[I][A->id()] = true;
      Bits[I][B->id()] = false;
    }
  }

  void dump(const Cluster *A) const {
    outs() << "Cluster " << A->id() << ":";
    forallAdjacent(A, [](const Cluster *B) { outs() << " " << B->id(); });
  }

  void dump() const {
    for (auto *A : Clusters) {
      if (!A) continue;
      dump(A);
      outs() << "\n";
    }
  }
 private:
  void set(const Cluster *A, const Cluster *B, bool Value) {
    assert(A != B);
    Bits[A->id()][B->id()] = Value;
    Bits[B->id()][A->id()] = Value;
  }

  void initialize(const CallGraph &Cg, const std::vector<Cluster *> &FuncCluster) {
    for (auto *A : Clusters) {
      for (auto TargetId : A->targets()) {
        for (auto Succ : Cg.successors(TargetId)) {
          auto *B = FuncCluster[Succ];
          if (!B || B == A) continue;
          const auto &Arc = *Cg.findArc(TargetId, Succ);
          if (Arc.weight() <= 0.0) continue;

          set(A, B, true);
        }
        for (auto Pred : Cg.predecessors(TargetId)) {
          auto *B = FuncCluster[Pred];
          if (!B || B == A) continue;
          const auto &Arc = *Cg.findArc(Pred, TargetId);
          if (Arc.weight() <= 0.0) continue;

          set(A, B, true);
        }
      }
    }
  }

  std::vector<Cluster *> Clusters;
  std::vector<BitVector> Bits;
};

// A cache of precomputed results for a pair of clusters
class PrecomputedResults {
 public:
  explicit PrecomputedResults(size_t Size)
  : Size(Size),
    Cache(new double[Size*Size]),
    Valid(Size * Size, false) {
    memset(Cache, 0, sizeof(double)*Size*Size);
  }
  ~PrecomputedResults() {
    delete[] Cache;
  }

  bool contains(const Cluster *First, const Cluster *Second) const {
    return Valid[index(First, Second)];
  }

  double get(const Cluster *First, const Cluster *Second) const {
    assert(contains(First, Second));
    return Cache[index(First, Second)];
  }

  void set(const Cluster *First, const Cluster *Second, double Value) {
    const auto Index = index(First, Second);
    Cache[Index] = Value;
    Valid[Index] = true;
  }

  void invalidate(const AdjacencyMatrix &Adjacent, const Cluster *C) {
    invalidate(C);
    Adjacent.forallAdjacent(C, [&](const Cluster *A) { invalidate(A); });
  }
 private:
  void invalidate(const Cluster *C) {
    Valid.reset(C->id() * Size, (C->id() + 1) * Size);
  }

  size_t index(const Cluster *First, const Cluster *Second) const {
    return (First->id() * Size) + Second->id();
  }

  size_t Size;
  double *Cache;
  BitVector Valid;
};

/*
 * Erase an element from a container if it is present.  Otherwise, do nothing.
 */
template <typename C, typename V>
void maybeErase(C &Container, const V& Value) {
  auto Itr = Container.find(Value);
  if (Itr != Container.end())
    Container.erase(Itr);
}

/*
 * Density of a cluster formed by merging a given pair of clusters
 */
double density(const Cluster *ClusterPred, const Cluster *ClusterSucc) {
  const double CombinedSamples = ClusterPred->samples() + ClusterSucc->samples();
  const double CombinedSize = ClusterPred->size() + ClusterSucc->size();
  return CombinedSamples / CombinedSize;
}

/*
 * Deterministically compare clusters by their density in decreasing order.
 */
bool compareClusters(const Cluster *C1, const Cluster *C2) {
  const double D1 = C1->density();
  const double D2 = C2->density();
  if (D1 != D2) return D1 > D2;
  // making sure the sorting is deterministic
  if (C1->size() != C2->size()) return C1->size() < C2->size();
  if (C1->samples() != C2->samples()) return C1->samples() > C2->samples();
  return C1->target(0) < C2->target(0);
}

/*
 * Deterministically compare pairs of clusters by their density
 * in decreasing order.
 */
bool compareClusterPairs(const Cluster *A1, const Cluster *B1,
                         const Cluster *A2, const Cluster *B2) {
  const auto D1 = density(A1, B1);
  const auto D2 = density(A2, B2);
  if (D1 != D2) return D1 > D2;
  // making sure the sorting is deterministic
  const auto Size1 = A1->size() + B1->size();
  const auto Size2 = A2->size() + B2->size();
  if (Size1 != Size2) return Size1 < Size2;
  const auto Samples1 = A1->samples() + B1->samples();
  const auto Samples2 = A2->samples() + B2->samples();
  if (Samples1 != Samples2) return Samples1 > Samples2;
  return A1->target(0) < A2->target(0);
}

/*
 * Sorting clusters by their density in decreasing order
 */
template <typename C>
std::vector<Cluster *> sortByDensity(const C &Clusters_) {
  std::vector<Cluster *> Clusters(Clusters_.begin(), Clusters_.end());
  std::stable_sort(Clusters.begin(), Clusters.end(), compareClusters);
  return Clusters;
}

/*
 * HFSortPlus - layout of hot functions with iTLB cache optimization
 *
 * Given an ordering of hot functions (and hence, their assignment to the
 * iTLB pages), we can divide all functions calls into two categories:
 * - 'short' ones that have a caller-callee distance less than a page;
 * - 'long' ones where the distance exceeds a page.
 * The short calls are likely to result in a iTLB cache hit. For the long ones,
 * the hit/miss result depends on the 'hotness' of the page (i.e., how often
 * the page is accessed). Assuming that functions are sent to the iTLB cache
 * in a random order, the probability that a page is present in the cache is
 * proportional to the number of samples corresponding to the functions on the
 * page. The following algorithm detects short and long calls, and optimizes
 * the expected number of cache misses for the long ones.
 */
class HFSortPlus {
public:
  /*
   * The expected number of calls on different i-TLB pages for an arc of the
   * call graph with a specified weight
   */
  double expectedCalls(int64_t SrcAddr, int64_t DstAddr, double Weight) const {
    const auto Dist = std::abs(SrcAddr - DstAddr);
    if (Dist > ITLBPageSize)
      return 0;

    double X = double(Dist) / double(ITLBPageSize);
    // Increasing the importance of shorter calls
    return (1.0 - X * X) * Weight;
  }

  /*
   * The probability that a page with a given weight is not present in the cache
   *
   * Assume that the hot functions are called in a random order; then the
   * probability of a i-TLB page being accessed after a function call is
   * p=pageSamples/totalSamples. The probability that the page is not accessed
   * is (1-p), and the probability that it is not in the cache (i.e. not accessed
   * during the last ITLBEntries function calls) is (1-p)^ITLBEntries
   */
  double missProbability(double PageSamples) const {
    double P = PageSamples / TotalSamples;
    double X = ITLBEntries;
    // avoiding precision issues for small values
    if (P < 0.0001) return (1.0 - X * P + X * (X - 1.0) * P * P / 2.0);
    return pow(1.0 - P, X);
  }

  /*
   * The expected number of calls within a given cluster with both endpoints on
   * the same cache page
   */
  double shortCalls(const Cluster *Cluster) const {
    if (UseShortCallCache) {
      auto Itr = ShortCallCache.find(Cluster);
      if (Itr != ShortCallCache.end())
        return Itr->second;
    }

    double Calls = 0;
    for (auto TargetId : Cluster->targets()) {
      for (auto Succ : Cg.successors(TargetId)) {
        if (FuncCluster[Succ] == Cluster) {
          const auto &Arc = *Cg.findArc(TargetId, Succ);

          auto SrcAddr = Addr[TargetId] + Arc.avgCallOffset();
          auto DstAddr = Addr[Succ];

          Calls += expectedCalls(SrcAddr, DstAddr, Arc.weight());
        }
      }
    }

    if (UseShortCallCache) {
      ShortCallCache[Cluster] = Calls;
    }

    return Calls;
  }

  /*
   * The number of calls between the two clusters with both endpoints on
   * the same i-TLB page, assuming that a given pair of clusters gets merged
   */
  double shortCalls(const Cluster *ClusterPred,
                    const Cluster *ClusterSucc) const {
    if (UseShortCallCache &&
        ShortCallPairCache.contains(ClusterPred, ClusterSucc)) {
      return ShortCallPairCache.get(ClusterPred, ClusterSucc);
    }

    double Calls = 0;
    for (auto TargetId : ClusterPred->targets()) {
      for (auto Succ : Cg.successors(TargetId)) {
        if (FuncCluster[Succ] == ClusterSucc) {
          const auto &Arc = *Cg.findArc(TargetId, Succ);

          auto SrcAddr = Addr[TargetId] + Arc.avgCallOffset();
          auto DstAddr = Addr[Succ] + ClusterPred->size();

          Calls += expectedCalls(SrcAddr, DstAddr, Arc.weight());
        }
      }
    }

    for (auto TargetId : ClusterPred->targets()) {
      for (auto Pred : Cg.predecessors(TargetId)) {
        if (FuncCluster[Pred] == ClusterSucc) {
          const auto &Arc = *Cg.findArc(Pred, TargetId);

          auto SrcAddr = Addr[Pred] + Arc.avgCallOffset() +
            ClusterPred->size();
          auto DstAddr = Addr[TargetId];

          Calls += expectedCalls(SrcAddr, DstAddr, Arc.weight());
        }
      }
    }

    if (UseShortCallCache) {
      ShortCallPairCache.set(ClusterPred, ClusterSucc, Calls);
    }

    return Calls;
  }

  /*
   * The gain of merging two clusters.
   *
   * We assume that the final clusters are sorted by their density, and hence
   * every cluster is likely to be adjacent with clusters of the same density.
   * Thus, the 'hotness' of every cluster can be estimated by density*pageSize,
   * which is used to compute the probability of cache misses for long calls
   * of a given cluster.
   * The result is also scaled by the size of the resulting cluster in order to
   * increse the chance of merging short clusters, which is helpful for
   * the i-cache performance.
   */
  double mergeGain(const Cluster *ClusterPred,
                   const Cluster *ClusterSucc) const {
    if (UseGainCache && Cache.contains(ClusterPred, ClusterSucc)) {
      return Cache.get(ClusterPred, ClusterSucc);
    }

    // cache misses on the first cluster
    double LongCallsPred = ClusterPred->samples() - shortCalls(ClusterPred);
    double ProbPred = missProbability(ClusterPred->density() * ITLBPageSize);
    double ExpectedMissesPred = LongCallsPred * ProbPred;

    // cache misses on the second cluster
    double LongCallsSucc = ClusterSucc->samples() - shortCalls(ClusterSucc);
    double ProbSucc = missProbability(ClusterSucc->density() * ITLBPageSize);
    double ExpectedMissesSucc = LongCallsSucc * ProbSucc;

    // cache misses on the merged cluster
    double LongCallsNew = LongCallsPred + LongCallsSucc -
                          shortCalls(ClusterPred, ClusterSucc);
    double NewDensity = density(ClusterPred, ClusterSucc);
    double ProbNew = missProbability(NewDensity * ITLBPageSize);
    double MissesNew = LongCallsNew * ProbNew;

    double Gain = ExpectedMissesPred + ExpectedMissesSucc - MissesNew;
    // scaling the result to increase the importance of merging short clusters
    Gain /= std::min(ClusterPred->size(), ClusterSucc->size());

    if (UseGainCache) {
      Cache.set(ClusterPred, ClusterSucc, Gain);
    }

    return Gain;
  }

  /*
   * For every active cluster, compute its total weight of outgoing edges
   */
  std::unordered_map<Cluster *, double> computeOutgoingWeight() {
    std::unordered_map<Cluster *, double> OutWeight;
    for (auto ClusterPred : Clusters) {
      double Weight = 0;
      for (auto TargetId : ClusterPred->targets()) {
        for (auto Succ : Cg.successors(TargetId)) {
          auto *ClusterSucc = FuncCluster[Succ];
          if (!ClusterSucc || ClusterSucc == ClusterPred)
            continue;
          const auto &Arc = *Cg.findArc(TargetId, Succ);
          Weight += Arc.weight();
        }
      }
      OutWeight[ClusterPred] += Weight;
    }
    return OutWeight;
  }

  /*
   * Find pairs of clusters that call each other with high probability
   */
  std::vector<std::pair<Cluster *, Cluster *>> findClustersToMerge() {
    // compute total weight of outgoing edges for every cluster
    auto OutWeight = computeOutgoingWeight();

    std::vector<std::pair<Cluster *, Cluster *>> PairsToMerge;
    std::unordered_set<Cluster *> ClustersToMerge;
    for (auto ClusterPred : Clusters) {
      for (auto TargetId : ClusterPred->targets()) {
        for (auto Succ : Cg.successors(TargetId)) {
          auto *ClusterSucc = FuncCluster[Succ];
          if (!ClusterSucc || ClusterSucc == ClusterPred)
            continue;

          const auto &Arc = *Cg.findArc(TargetId, Succ);

          const double CallsFromPred = OutWeight[ClusterPred];
          const double CallsToSucc = ClusterSucc->samples();
          const double CallsPredSucc = Arc.weight();

          // probability that the first cluster is calling the second one
          const double ProbOut =
            CallsFromPred > 0 ? CallsPredSucc / CallsFromPred : 0;
          assert(0.0 <= ProbOut && ProbOut <= 1.0 && "incorrect probability");
          
          // probability that the second cluster is called from the first one
          const double ProbIn =
            CallsToSucc > 0 ? CallsPredSucc / CallsToSucc : 0;
          assert(0.0 <= ProbIn && ProbIn <= 1.0 && "incorrect probability");

          if (std::min(ProbOut, ProbIn) >= opts::MergeProbability) {
            if (ClustersToMerge.count(ClusterPred) == 0 &&
                ClustersToMerge.count(ClusterSucc) == 0) {
              PairsToMerge.push_back(std::make_pair(ClusterPred, ClusterSucc));
              ClustersToMerge.insert(ClusterPred);
              ClustersToMerge.insert(ClusterSucc);
            }
          }
        }
      }
    }

    return PairsToMerge;
  }

  /*
   * Run the first optimization pass of the hfsort+ algorithm:
   * Merge clusters that call each other with high probability
   */
  void runPassOne() {
    while (Clusters.size() > 1) {
      // pairs of clusters that will be merged on this iteration
      auto PairsToMerge = findClustersToMerge();

      // stop the pass when there are no pairs to merge
      if (PairsToMerge.empty())
        break;

      // merge the pairs of clusters
      for (auto &Pair : PairsToMerge) {
        mergeClusters(Pair.first, Pair.second);
      }
    }
  }

  /*
   * Run the second optimization pass of the hfsort+ algorithm:
   * Merge pairs of clusters while there is an improvement in the
   * expected cache miss ratio
   */
  void runPassTwo() {
    while (Clusters.size() > 1) {
      Cluster *BestClusterPred = nullptr;
      Cluster *BestClusterSucc = nullptr;
      double BestGain = -1;
      for (auto ClusterPred : Clusters) {
        // get candidates for merging with the current cluster
        Adjacent.forallAdjacent(
          ClusterPred,
          // find the best candidate
          [&](Cluster *ClusterSucc) {
            assert(ClusterPred != ClusterSucc && "loop edges are not supported");
            // compute the gain of merging two clusters
            const double Gain = mergeGain(ClusterPred, ClusterSucc);

            // breaking ties by density to make the hottest clusters be merged first
            if (Gain > BestGain || (std::abs(Gain - BestGain) < 1e-8 &&
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

      // stop merging when there is no improvement
      if (BestGain <= 0.0)
        break;

      // merge the best pair of clusters
      mergeClusters(BestClusterPred, BestClusterSucc);
    }
  }

  /*
   * Run hfsort+ algorithm and return ordered set of function clusters.
   */
  std::vector<Cluster> run() {
    DEBUG(dbgs() << "Starting hfsort+ w/"
                 << cacheKindString(UseGainCache, UseShortCallCache)
                 << " for " << Clusters.size() << " clusters "
                 << "with ITLBPageSize = " << ITLBPageSize << ", "
                 << "ITLBEntries = " << ITLBEntries << ", "
                 << "and MergeProbability = " << opts::MergeProbability << "\n");


    // Pass 1
    runPassOne();

    // Pass 2
    runPassTwo();

    DEBUG(dbgs() << "Completed hfsort+ with " << Clusters.size() << " clusters\n");

    // Return the set of clusters that are left, which are the ones that
    // didn't get merged (so their first func is its original func)
    std::vector<Cluster> Result;
    for (auto Cluster : sortByDensity(Clusters)) {
      Result.emplace_back(std::move(*Cluster));
    }

    assert(std::is_sorted(Result.begin(), Result.end(), compareClustersDensity));

    return Result;
  }

  HFSortPlus(const CallGraph &Cg,
             bool UseGainCache,
             bool UseShortCallCache)
  : Cg(Cg),
    FuncCluster(Cg.numNodes(), nullptr),
    Addr(Cg.numNodes(), InvalidAddr),
    TotalSamples(0.0),
    Clusters(initializeClusters()),
    Adjacent(Cg, Clusters, FuncCluster),
    UseGainCache(UseGainCache),
    UseShortCallCache(UseShortCallCache),
    Cache(Clusters.size()),
    ShortCallPairCache(Clusters.size()) {
  }
private:

  /*
   * Initialize the set of active clusters, function id to cluster mapping,
   * total number of samples and function addresses.
   */
  std::vector<Cluster *> initializeClusters() {
    ITLBPageSize = opts::ITLBPageSizeParam;
    ITLBEntries = opts::ITLBEntriesParam;

    // Initialize clusters
    std::vector<Cluster *> Clusters;
    Clusters.reserve(Cg.numNodes());
    AllClusters.reserve(Cg.numNodes());
    for (NodeId F = 0; F < Cg.numNodes(); ++F) {
      AllClusters.emplace_back(F, Cg.getNode(F));
      Clusters.emplace_back(&AllClusters[F]);
      Clusters.back()->setId(Clusters.size() - 1);
      FuncCluster[F] = &AllClusters[F];
      Addr[F] = 0;
      TotalSamples += Cg.samples(F);
    }

    return Clusters;
  }

  /*
   * Merge cluster From into cluster Into and update the list of active clusters
   */
  void mergeClusters(Cluster *Into, Cluster *From) {
    DEBUG(
      if (opts::Verbosity > 0) {
        dbgs() << "Merging cluster " << From->id()
               << " into cluster " << Into->id() << "\n";
      });

    // The adjacency merge must happen before the Cluster::merge since that
    // clobbers the contents of From.
    Adjacent.merge(Into, From);

    Into->merge(*From);

    // Update the clusters and addresses for functions merged from From.
    size_t CurAddr = 0;
    for (auto TargetId : Into->targets()) {
      FuncCluster[TargetId] = Into;
      Addr[TargetId] = CurAddr;
      CurAddr += Cg.size(TargetId);
      // Functions are aligned in the output binary,
      // replicating the effect here using BinaryFunction::MinAlign
      const auto Align = BinaryFunction::MinAlign;
      CurAddr = ((CurAddr + Align - 1) / Align) * Align;
    }

    // Update caches
    invalidateCaches(Into);

    // Remove cluster From from the list of active clusters
    auto Iter = std::remove(Clusters.begin(), Clusters.end(), From);
    Clusters.erase(Iter, Clusters.end());
  }

  /*
   * Invalidate all cache entries associated with cluster C and its neighbors.
   */
  void invalidateCaches(const Cluster *C) {
    if (UseShortCallCache) {
      maybeErase(ShortCallCache, C);
      Adjacent.forallAdjacent(C,
        [this](const Cluster *A) {
          maybeErase(ShortCallCache, A);
        });
      ShortCallPairCache.invalidate(Adjacent, C);
    }
    if (UseGainCache) {
      Cache.invalidate(Adjacent, C);
    }
  }

  // The call graph
  const CallGraph &Cg;

  // All clusters
  std::vector<Cluster> AllClusters;

  // Target_id => cluster
  std::vector<Cluster *> FuncCluster;

  // current address of the function from the beginning of its cluster
  std::vector<size_t> Addr;

  // the total number of samples in the graph
  double TotalSamples;

  // All clusters with non-zero number of samples. This vector gets
  // udpated at runtime when clusters are merged.
  std::vector<Cluster *> Clusters;

  // Cluster adjacency matrix
  AdjacencyMatrix Adjacent;

  // Use cache for mergeGain results
  bool UseGainCache;

  // Use caches for shortCalls results
  bool UseShortCallCache;

  // A cache that keeps precomputed values of mergeGain for pairs of clusters;
  // when a pair of clusters (x,y) gets merged, we need to invalidate the pairs
  // containing both x and y and all clusters adjacent to x and y (and recompute
  // them on the next iteration).
  mutable PrecomputedResults Cache;

  // Cache for shortCalls for a single cluster.
  mutable std::unordered_map<const Cluster *, double> ShortCallCache;

  // Cache for shortCalls for a pair of Clusters
  mutable PrecomputedResults ShortCallPairCache;
};

}

std::vector<Cluster> hfsortPlus(CallGraph &Cg,
                                bool UseGainCache,
                                bool UseShortCallCache) {
  // It is required that the sum of incoming arc weights is not greater
  // than the number of samples for every function.
  // Ensuring the call graph obeys the property before running the algorithm.
  Cg.adjustArcWeights();
  return HFSortPlus(Cg, UseGainCache, UseShortCallCache).run();
}

}}
