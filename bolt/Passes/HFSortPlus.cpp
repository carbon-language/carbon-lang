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

namespace opts {
extern llvm::cl::opt<bool> Verbosity;
}

namespace llvm {
namespace bolt {

using NodeId = CallGraph::NodeId;
using Arc = CallGraph::Arc;
using Node = CallGraph::Node;  

namespace {

// The size of a cache page
// Since we optimize both for iTLB cache (2MB pages) and i-cache (64b pages),
// using a value that fits both
constexpr uint32_t PageSize = uint32_t(1) << 12;

// Capacity of the iTLB cache: larger values yield more iTLB-friendly result,
// while smaller values result in better i-cache performance
constexpr uint32_t ITLBEntries = 16;

constexpr size_t InvalidAddr = -1;

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
          set(A, B, true);
        }
        for (auto Pred : Cg.predecessors(TargetId)) {
          auto *B = FuncCluster[Pred];
          if (!B || B == A) continue;
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
  // making sure the sorting is deterministic
  if (D1 != D2) return D1 > D2;
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
 * The probability that a page with a given weight is not present in the cache.
 *
 * Assume that the hot functions are called in a random order; then the
 * probability of a TLB page being accessed after a function call is
 * p=pageSamples/totalSamples. The probability that the page is not accessed
 * is (1-p), and the probability that it is not in the cache (i.e. not accessed
 * during the last kITLBEntries function calls) is (1-p)^kITLBEntries
 */
double expectedCalls(int64_t SrcAddr, int64_t DstAddr, double EdgeWeight) {
  const auto Dist = std::abs(SrcAddr - DstAddr);
  if (Dist > PageSize) {
    return 0;
  }
  return (double(PageSize - Dist) / PageSize) * EdgeWeight;
}

/*
 * HFSortPlus - layout of hot functions with iTLB cache optimization
 */
class HFSortPlus {
public:
  /*
   * The probability that a page with a given weight is not present in the cache.
   *
   * Assume that the hot functions are called in a random order; then the
   * probability of a TLB page being accessed after a function call is
   * p=pageSamples/totalSamples. The probability that the page is not accessed
   * is (1-p), and the probability that it is not in the cache (i.e. not accessed
   * during the last kITLBEntries function calls) is (1-p)^kITLBEntries
   */
  double missProbability(double PageSamples) const {
    double P = PageSamples / TotalSamples;
    double X = ITLBEntries;
    // avoiding precision issues for small values
    if (P < 0.0001) return (1.0 - X * P + X * (X - 1.0) * P * P / 2.0);
    return pow(1.0 - P, X);
  }

  /*
   * Expected hit ratio of the iTLB cache under the given order of clusters
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
   * page. The following procedure detects short and long calls, and estimates
   * the expected number of cache misses for the long ones.
   */
  template <typename C>
  double expectedCacheHitRatio(const C &Clusters_) const {
    // sort by density
    std::vector<Cluster *> Clusters(sortByDensity(Clusters_));

    // generate function addresses with an alignment
    std::vector<size_t> Addr(Cg.numNodes(), InvalidAddr);
    size_t CurAddr = 0;
    // 'hotness' of the pages
    std::vector<double> PageSamples;
    for (auto Cluster : Clusters) {
      for (auto TargetId : Cluster->targets()) {
        if (CurAddr & 0xf) CurAddr = (CurAddr & ~0xf) + 16;
        Addr[TargetId] = CurAddr;
        CurAddr += Cg.size(TargetId);
        // update page weight
        size_t Page = Addr[TargetId] / PageSize;
        while (PageSamples.size() <= Page) PageSamples.push_back(0.0);
        PageSamples[Page] += Cg.samples(TargetId);
      }
    }

    // computing expected number of misses for every function
    double Misses = 0;
    for (auto Cluster : Clusters) {
      for (auto TargetId : Cluster->targets()) {
        size_t Page = Addr[TargetId] / PageSize;
        double Samples = Cg.samples(TargetId);
        // probability that the page is not present in the cache
        double MissProb = missProbability(PageSamples[Page]);

        for (auto Pred : Cg.predecessors(TargetId)) {
          if (Cg.samples(Pred) == 0) continue;
          const auto &Arc = *Cg.findArc(Pred, TargetId);

          // the source page
          size_t SrcPage = (Addr[Pred] + (size_t)Arc.avgCallOffset()) / PageSize;
          if (Page != SrcPage) {
            // this is a miss
            Misses += Arc.weight() * MissProb;
          }
          Samples -= Arc.weight();
        }

        // the remaining samples come from the jitted code
        Misses += Samples * MissProb;
      }
    }

    return 100.0 * (1.0 - Misses / TotalSamples);
  }

  /*
   * The expected number of calls within a given cluster with both endpoints on
   * the same TLB cache page
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
   * the same TLB page, assuming that a given pair of clusters gets merged
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
    double ProbPred = missProbability(ClusterPred->density() * PageSize);
    double ExpectedMissesPred = LongCallsPred * ProbPred;

    // cache misses on the second cluster
    double LongCallsSucc = ClusterSucc->samples() - shortCalls(ClusterSucc);
    double ProbSucc = missProbability(ClusterSucc->density() * PageSize);
    double ExpectedMissesSucc = LongCallsSucc * ProbSucc;

    // cache misses on the merged cluster
    double LongCallsNew = LongCallsPred + LongCallsSucc -
                          shortCalls(ClusterPred, ClusterSucc);
    double NewDensity = density(ClusterPred, ClusterSucc);
    double ProbNew = missProbability(NewDensity * PageSize);
    double MissesNew = LongCallsNew * ProbNew;

    double Gain = ExpectedMissesPred + ExpectedMissesSucc - MissesNew;
    // scaling the result to increase the importance of merging short clusters
    Gain /= (ClusterPred->size() + ClusterSucc->size());

    if (UseGainCache) {
      Cache.set(ClusterPred, ClusterSucc, Gain);
    }

    return Gain;
  }

  /*
   * Run hfsort+ algorithm and return ordered set of function clusters.
   */
  std::vector<Cluster> run() {
    DEBUG(dbgs() << "Starting hfsort+ w/"
                 << cacheKindString(UseGainCache, UseShortCallCache)
                 << " for " << Clusters.size() << " clusters\n"
                 << format("Initial expected iTLB cache hit ratio: %.4lf\n",
                           expectedCacheHitRatio(Clusters)));

    int Steps = 0;
    // merge pairs of clusters while there is an improvement
    while (Clusters.size() > 1) {
      DEBUG(
        if (Steps % 500 == 0) {
          dbgs() << format("step = %d  clusters = %lu  expected_hit_rate = %.4lf\n",
                           Steps, Clusters.size(),
                           expectedCacheHitRatio(Clusters));
        });
      ++Steps;

      Cluster *BestClusterPred = nullptr;
      Cluster *BestClusterSucc = nullptr;
      double BestGain = -1;
      for (auto ClusterPred : Clusters) {
        // get candidates for merging with the current cluster
        Adjacent.forallAdjacent(
          ClusterPred,
          // find the best candidate
          [&](Cluster *ClusterSucc) {
            assert(ClusterPred != ClusterSucc);
            // get a cost of merging two clusters
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

      if (BestGain <= 0.0) break;

      // merge the best pair of clusters
      mergeClusters(BestClusterPred, BestClusterSucc);

      // remove BestClusterSucc from the list of active clusters
      auto Iter = std::remove(Clusters.begin(), Clusters.end(), BestClusterSucc);
      Clusters.erase(Iter, Clusters.end());
    }

    DEBUG(dbgs() << "Completed hfsort+ with " << Clusters.size() << " clusters\n"
                 << format("Final expected iTLB cache hit ratio: %.4lf\n",
                           expectedCacheHitRatio(Clusters)));

    // Return the set of clusters that are left, which are the ones that
    // didn't get merged (so their first func is its original func).
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
  // Initialize the set of active clusters, function id to cluster mapping,
  // total number of samples and function addresses.
  std::vector<Cluster *> initializeClusters() {
    std::vector<Cluster *> Clusters;

    Clusters.reserve(Cg.numNodes());
    AllClusters.reserve(Cg.numNodes());

    for (NodeId F = 0; F < Cg.numNodes(); F++) {
      AllClusters.emplace_back(F, Cg.getNode(F));
      if (Cg.samples(F) == 0) continue;
      Clusters.emplace_back(&AllClusters[F]);
      Clusters.back()->setId(Clusters.size() - 1);
      FuncCluster[F] = &AllClusters[F];
      Addr[F] = 0;
      TotalSamples += Cg.samples(F);
    }

    return Clusters;
  }

  /*
   * Merge cluster From into cluster Into.
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
    }

    invalidateCaches(Into);
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

  // the call graph
  const CallGraph &Cg;

  // All clusters.
  std::vector<Cluster> AllClusters;

  // target_id => cluster
  std::vector<Cluster *> FuncCluster;

  // current address of the function from the beginning of its cluster
  std::vector<size_t> Addr;

  // the total number of samples in the graph
  double TotalSamples;

  // All clusters with non-zero number of samples.  This vector gets
  // udpated at runtime when clusters are merged.
  std::vector<Cluster *> Clusters;

  // Cluster adjacency matrix.
  AdjacencyMatrix Adjacent;

  // Use cache for mergeGain results.
  bool UseGainCache;

  // Use caches for shortCalls results.
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

std::vector<Cluster> hfsortPlus(const CallGraph &Cg,
                                bool UseGainCache,
                                bool UseShortCallCache) {
  return HFSortPlus(Cg, UseGainCache, UseShortCallCache).run();
}

}}
