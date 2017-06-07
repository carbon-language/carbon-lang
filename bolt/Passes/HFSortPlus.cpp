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

extern llvm::cl::OptionCategory BoltOptCategory;
extern llvm::cl::opt<bool> Verbosity;

static llvm::cl::opt<bool>
UseGainCache("hfsort+-use-cache",
  llvm::cl::desc("Use a cache for mergeGain results when computing hfsort+."),
  llvm::cl::ZeroOrMore,
  llvm::cl::init(true),
  llvm::cl::Hidden,
  llvm::cl::cat(BoltOptCategory));

static llvm::cl::opt<bool>
UseShortCallCache("hfsort+-use-short-call-cache",
  llvm::cl::desc("Use a cache for shortCall results when computing hfsort+."),
  llvm::cl::ZeroOrMore,
  llvm::cl::init(true),
  llvm::cl::Hidden,
  llvm::cl::cat(BoltOptCategory));

const char* cacheKindString() {
  if (opts::UseGainCache && opts::UseShortCallCache)
    return "gain + short call cache";
  else if (opts::UseGainCache)
    return "gain cache";
  else if (opts::UseShortCallCache)
    return "short call cache";
  else
    return "no cache";
}

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
    forallAdjacent(A,
                   [this,A](const Cluster *B) {
                     outs() << " " << B->id();
                   });
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

// A wrapper for algorithm-wide variables
struct AlgoState {
  explicit AlgoState(size_t Size)
    : Cache(Size), ShortCallPairCache(Size) { }

  // the call graph
  const CallGraph *Cg;
  // the total number of samples in the graph
  double TotalSamples;
  // target_id => cluster
  std::vector<Cluster *> FuncCluster;
  // current address of the function from the beginning of its cluster
  std::vector<size_t> Addr;
  // maximum cluster id.
  size_t MaxClusterId{0};
  // A cache that keeps precomputed values of mergeGain for pairs of clusters;
  // when a pair of clusters (x,y) gets merged, we need to invalidate the pairs
  // containing both x and y (and recompute them on the next iteration)
  PrecomputedResults Cache;
  // Cache for shortCalls for a single cluster.
  std::unordered_map<const Cluster *, double> ShortCallCache;
  // Cache for shortCalls for a pair of Clusters
  PrecomputedResults ShortCallPairCache;
};

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
double missProbability(const AlgoState &State, double PageSamples) {
  double P = PageSamples / State.TotalSamples;
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
double expectedCacheHitRatio(const AlgoState &State, const C &Clusters_) {
  // sort by density
  std::vector<Cluster *> Clusters(sortByDensity(Clusters_));

  // generate function addresses with an alignment
  std::vector<size_t> Addr(State.Cg->numNodes(), InvalidAddr);
  size_t CurAddr = 0;
  // 'hotness' of the pages
  std::vector<double> PageSamples;
  for (auto Cluster : Clusters) {
    for (auto TargetId : Cluster->targets()) {
      if (CurAddr & 0xf) CurAddr = (CurAddr & ~0xf) + 16;
      Addr[TargetId] = CurAddr;
      CurAddr += State.Cg->size(TargetId);
      // update page weight
      size_t Page = Addr[TargetId] / PageSize;
      while (PageSamples.size() <= Page) PageSamples.push_back(0.0);
      PageSamples[Page] += State.Cg->samples(TargetId);
    }
  }

  // computing expected number of misses for every function
  double Misses = 0;
  for (auto Cluster : Clusters) {
    for (auto TargetId : Cluster->targets()) {
      size_t Page = Addr[TargetId] / PageSize;
      double Samples = State.Cg->samples(TargetId);
      // probability that the page is not present in the cache
      double MissProb = missProbability(State, PageSamples[Page]);

      for (auto Pred : State.Cg->predecessors(TargetId)) {
        if (State.Cg->samples(Pred) == 0) continue;
        const auto &Arc = *State.Cg->findArc(Pred, TargetId);

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

  return 100.0 * (1.0 - Misses / State.TotalSamples);
}

/*
 * The expected number of calls for an edge withing the same TLB page
 */
double expectedCalls(int64_t SrcAddr, int64_t DstAddr, double EdgeWeight) {
  auto Dist = std::abs(SrcAddr - DstAddr);
  if (Dist > PageSize) {
    return 0;
  }
  return (double(PageSize - Dist) / PageSize) * EdgeWeight;
}

/*
 * The expected number of calls within a given cluster with both endpoints on
 * the same TLB cache page
 */
double shortCalls(AlgoState &State, const Cluster *Cluster) {
  if (opts::UseShortCallCache) {
    auto Itr = State.ShortCallCache.find(Cluster);
    if (Itr != State.ShortCallCache.end())
      return Itr->second;
  }

  double Calls = 0;
  for (auto TargetId : Cluster->targets()) {
    for (auto Succ : State.Cg->successors(TargetId)) {
      if (State.FuncCluster[Succ] == Cluster) {
        const auto &Arc = *State.Cg->findArc(TargetId, Succ);

        auto SrcAddr = State.Addr[TargetId] + Arc.avgCallOffset();
        auto DstAddr = State.Addr[Succ];

        Calls += expectedCalls(SrcAddr, DstAddr, Arc.weight());
      }
    }
  }

  if (opts::UseShortCallCache) {
    State.ShortCallCache[Cluster] = Calls;
  }

  return Calls;
}

/*
 * The number of calls between the two clusters with both endpoints on
 * the same TLB page, assuming that a given pair of clusters gets merged
 */
double shortCalls(AlgoState &State,
                  const Cluster *ClusterPred,
                  const Cluster *ClusterSucc) {
  if (opts::UseShortCallCache &&
      State.ShortCallPairCache.contains(ClusterPred, ClusterSucc)) {
    return State.ShortCallPairCache.get(ClusterPred, ClusterSucc);
  }

  double Calls = 0;
  for (auto TargetId : ClusterPred->targets()) {
    for (auto Succ : State.Cg->successors(TargetId)) {
      if (State.FuncCluster[Succ] == ClusterSucc) {
        const auto &Arc = *State.Cg->findArc(TargetId, Succ);

        auto SrcAddr = State.Addr[TargetId] + Arc.avgCallOffset();
        auto DstAddr = State.Addr[Succ] + ClusterPred->size();

        Calls += expectedCalls(SrcAddr, DstAddr, Arc.weight());
      }
    }
  }

  for (auto TargetId : ClusterPred->targets()) {
    for (auto Pred : State.Cg->predecessors(TargetId)) {
      if (State.FuncCluster[Pred] == ClusterSucc) {
        const auto &Arc = *State.Cg->findArc(Pred, TargetId);

        auto SrcAddr = State.Addr[Pred] + Arc.avgCallOffset() +
          ClusterPred->size();
        auto DstAddr = State.Addr[TargetId];

        Calls += expectedCalls(SrcAddr, DstAddr, Arc.weight());
      }
    }
  }

  if (opts::UseShortCallCache) {
    State.ShortCallPairCache.set(ClusterPred, ClusterSucc, Calls);
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
double mergeGain(AlgoState &State,
                 const Cluster *ClusterPred,
                 const Cluster *ClusterSucc) {
  if (opts::UseGainCache && State.Cache.contains(ClusterPred, ClusterSucc)) {
    return State.Cache.get(ClusterPred, ClusterSucc);
  }

  // cache misses on the first cluster
  double LongCallsPred = ClusterPred->samples() - shortCalls(State, ClusterPred);
  double ProbPred = missProbability(State, ClusterPred->density() * PageSize);
  double ExpectedMissesPred = LongCallsPred * ProbPred;

  // cache misses on the second cluster
  double LongCallsSucc = ClusterSucc->samples() - shortCalls(State, ClusterSucc);
  double ProbSucc = missProbability(State, ClusterSucc->density() * PageSize);
  double ExpectedMissesSucc = LongCallsSucc * ProbSucc;

  // cache misses on the merged cluster
  double LongCallsNew = LongCallsPred + LongCallsSucc -
                        shortCalls(State, ClusterPred, ClusterSucc);
  double NewDensity = density(ClusterPred, ClusterSucc);
  double ProbNew = missProbability(State, NewDensity * PageSize);
  double MissesNew = LongCallsNew * ProbNew;

  double Gain = ExpectedMissesPred + ExpectedMissesSucc - MissesNew;
  // scaling the result to increase the importance of merging short clusters
  Gain /= (ClusterPred->size() + ClusterSucc->size());

  if (opts::UseGainCache) {
    State.Cache.set(ClusterPred, ClusterSucc, Gain);
  }

  return Gain;
}

template <typename C, typename V>
void maybeErase(C &Container, const V& Value) {
  auto Itr = Container.find(Value);
  if (Itr != Container.end())
    Container.erase(Itr);
}

/*
 * HFSortPlus - layout of hot functions with iTLB cache optimization
 */
std::vector<Cluster> hfsortPlus(const CallGraph &Cg) {
  // create a cluster for every function
  std::vector<Cluster> AllClusters;
  AllClusters.reserve(Cg.numNodes());
  for (NodeId F = 0; F < Cg.numNodes(); F++) {
    AllClusters.emplace_back(F, Cg.getNode(F));
  }

  // initialize objects used by the algorithm
  std::vector<Cluster *> Clusters;
  Clusters.reserve(Cg.numNodes());
  AlgoState State(AllClusters.size()); // TODO: should use final Clusters.size()
  State.Cg = &Cg;
  State.TotalSamples = 0;
  State.FuncCluster = std::vector<Cluster *>(Cg.numNodes(), nullptr);
  State.Addr = std::vector<size_t>(Cg.numNodes(), InvalidAddr);
  uint32_t Id = 0;
  for (NodeId F = 0; F < Cg.numNodes(); F++) {
    if (Cg.samples(F) == 0) continue;
    Clusters.push_back(&AllClusters[F]);
    Clusters.back()->setId(Id);
    State.FuncCluster[F] = &AllClusters[F];
    State.Addr[F] = 0;
    State.TotalSamples += Cg.samples(F);
    ++Id;
  }
  State.MaxClusterId = Id;

  AdjacencyMatrix Adjacent(Cg, Clusters, State.FuncCluster);

  DEBUG(dbgs() << "Starting hfsort+ w/" << opts::cacheKindString() << " for "
               << Clusters.size() << " clusters\n"
               << format("Initial expected iTLB cache hit ratio: %.4lf\n",
                         expectedCacheHitRatio(State, Clusters)));

  int Steps = 0;
  // merge pairs of clusters while there is an improvement
  while (Clusters.size() > 1) {
    DEBUG(
      if (Steps % 500 == 0) {
        dbgs() << format("step = %d  clusters = %lu  expected_hit_rate = %.4lf\n",
                         Steps,
                         Clusters.size(),
                         expectedCacheHitRatio(State, Clusters));
      }
    );
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
          const double Gain = mergeGain(State, ClusterPred, ClusterSucc);

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
        }
      );
    }

    if (BestGain <= 0.0) break;

    // merge the best pair of clusters
    DEBUG(
      if (opts::Verbosity > 0) {
        dbgs() << "Merging cluster " << BestClusterSucc->id()
               << " into cluster " << BestClusterPred->id() << "\n";
      });

    Adjacent.merge(BestClusterPred, BestClusterSucc);
    BestClusterPred->merge(*BestClusterSucc);

    size_t CurAddr = 0;
    for (auto TargetId : BestClusterPred->targets()) {
      State.FuncCluster[TargetId] = BestClusterPred;
      State.Addr[TargetId] = CurAddr;
      CurAddr += State.Cg->size(TargetId);
    }

    if (opts::UseShortCallCache) {
      maybeErase(State.ShortCallCache, BestClusterPred);
      Adjacent.forallAdjacent(BestClusterPred,
                              [&State](const Cluster *C) {
                                maybeErase(State.ShortCallCache, C);
                              });
      State.ShortCallPairCache.invalidate(Adjacent, BestClusterPred);
    }
    if (opts::UseGainCache) {
      State.Cache.invalidate(Adjacent, BestClusterPred);
    }

    // remove BestClusterSucc from the list of active clusters
    auto Iter = std::remove(Clusters.begin(), Clusters.end(), BestClusterSucc);
    Clusters.erase(Iter, Clusters.end());
  }

  DEBUG(dbgs() << "Completed hfsort+ with " << Clusters.size() << " clusters\n"
               << format("Final expected iTLB cache hit ratio: %.4lf\n",
                         expectedCacheHitRatio(State, Clusters)));

  // Return the set of clusters that are left, which are the ones that
  // didn't get merged (so their first func is its original func).
  std::vector<Cluster> Result;
  for (auto Cluster : sortByDensity(Clusters)) {
    Result.emplace_back(std::move(*Cluster));
  }

  assert(std::is_sorted(Result.begin(), Result.end(), compareClustersDensity));

  return Result;
}

}}
