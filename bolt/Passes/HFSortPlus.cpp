//===--- HFSort.cpp - Cluster functions by hotness ------------------------===//
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
#include "llvm/Support/Format.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "hfsort"

namespace llvm {
namespace bolt {

namespace {

// The size of a cache page
// Since we optimize both for iTLB cache (2MB pages) and i-cache (64b pages),
// using a value that fits both
constexpr uint32_t PageSize = uint32_t(1) << 12;

// Capacity of the iTLB cache: larger values yield more iTLB-friendly result,
// while smaller values result in better i-cache performance
constexpr uint32_t ITLBEntries = 16;

constexpr size_t InvalidAddr = -1;

template <typename A, typename B>
class HashPair {
public:
  size_t operator()(const std::pair<A, B> &P) const {
    size_t Seed(0);
    Seed = hashCombine(Seed, (int64_t)P.first);
    Seed = hashCombine(Seed, (int64_t)P.second);
    return Seed;
  }
};

// A cache of precomputed results for a pair of clusters
class PrecomputedResults {
 public:
  PrecomputedResults() {}

  bool contains(Cluster *First, Cluster *Second) const {
    if (InvalidKeys.count(First) || InvalidKeys.count(Second)) {
      return false;
    }
    const auto Key = std::make_pair(First, Second);
    return Cache.find(Key) != Cache.end();
  }

  double get(Cluster *First, Cluster *Second) const {
    assert(contains(First, Second));
    const auto Key = std::make_pair(First, Second); // TODO: use min/max?
    return Cache.find(Key)->second;
  }

  void set(Cluster *First, Cluster *Second, double Value) {
    const auto Key = std::make_pair(First, Second);
    Cache[Key] = Value;
    validate(First);
    validate(Second);
  }

  void validate(Cluster *C) {
    auto Itr = InvalidKeys.find(C);
    if (Itr != InvalidKeys.end())
      InvalidKeys.erase(Itr);
  }

  void validateAll() {
    InvalidKeys.clear();
  }

  void invalidate(Cluster *Cluster) {
    InvalidKeys.insert(Cluster);
  }

 private:
  std::unordered_map<std::pair<Cluster *, Cluster *>,
                     double,
                     HashPair<Cluster *,Cluster *>> Cache;
  std::unordered_set<Cluster *> InvalidKeys;
};

// A wrapper for algorthm-wide variables
struct AlgoState {
  // the call graph
  const TargetGraph *Cg;
  // the total number of samples in the graph
  double TotalSamples;
  // target_id => cluster
  std::vector<Cluster *> FuncCluster;
  // current address of the function from the beginning of its cluster
  std::vector<size_t> Addr;
};

bool compareClustersDensity(const Cluster &C1, const Cluster &C2) {
  return C1.density() > C2.density();
}

}

/*
 * Sorting clusters by their density in decreasing order
 */
void sortByDensity(std::vector<Cluster *> &Clusters) {
  std::sort(
    Clusters.begin(),
    Clusters.end(),
    [&] (const Cluster *C1, const Cluster *C2) {
      const double D1 = C1->density();
      const double D2 = C2->density();
      // making sure the sorting is deterministic
      if (D1 != D2) return D1 > D2;
      if (C1->Size != C2->Size) return C1->Size < C2->Size;
      if (C1->Samples != C2->Samples) return C1->Samples > C2->Samples;
      return C1->Targets[0] < C2->Targets[0];
    }
  );
}

/*
 * Density of a cluster formed by merging a given pair of clusters
 */
double density(Cluster *ClusterPred, Cluster *ClusterSucc) {
  const double CombinedSamples = ClusterPred->Samples + ClusterSucc->Samples;
  const double CombinedSize = ClusterPred->Size + ClusterSucc->Size;
  return CombinedSamples / CombinedSize;
}

/*
 * The probability that a page with a given weight is not present in the cache.
 *
 * Assume that the hot function are called in a random order; then the
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
double expectedCacheHitRatio(const AlgoState &State,
                             const std::vector<Cluster *> &Clusters_) {
  // copy and sort by density
  std::vector<Cluster *> Clusters(Clusters_);
  sortByDensity(Clusters);

  // generate function addresses with an alignment
  std::vector<size_t> Addr(State.Cg->Targets.size(), InvalidAddr);
  size_t CurAddr = 0;
  // 'hotness' of the pages
  std::vector<double> PageSamples;
  for (auto Cluster : Clusters) {
    for (auto TargetId : Cluster->Targets) {
      if (CurAddr & 0xf) CurAddr = (CurAddr & ~0xf) + 16;
      Addr[TargetId] = CurAddr;
      CurAddr += State.Cg->Targets[TargetId].Size;
      // update page weight
      size_t Page = Addr[TargetId] / PageSize;
      while (PageSamples.size() <= Page) PageSamples.push_back(0.0);
      PageSamples[Page] += State.Cg->Targets[TargetId].Samples;
    }
  }

  // computing expected number of misses for every function
  double Misses = 0;
  for (auto Cluster : Clusters) {
    for (auto TargetId : Cluster->Targets) {
      size_t Page = Addr[TargetId] / PageSize;
      double Samples = State.Cg->Targets[TargetId].Samples;
      // probability that the page is not present in the cache
      double MissProb = missProbability(State, PageSamples[Page]);

      for (auto Pred : State.Cg->Targets[TargetId].Preds) {
        if (State.Cg->Targets[Pred].Samples == 0) continue;
        auto A = State.Cg->Arcs.find(Arc(Pred, TargetId));

        // the source page
        size_t SrcPage = (Addr[Pred] + (size_t)A->AvgCallOffset) / PageSize;
        if (Page != SrcPage) {
          // this is a miss
          Misses += A->Weight * MissProb;
        }
        Samples -= A->Weight;
      }

      // the remaining samples come from the jitted code
      Misses += Samples * MissProb;
    }
  }

  return 100.0 * (1.0 - Misses / State.TotalSamples);
}

/*
 * Get adjacent clusters (the ones that share an arc) with the given one
 */
std::unordered_set<Cluster *> adjacentClusters(const AlgoState &State,
                                              Cluster *C) {
  std::unordered_set<Cluster *> Result;
  for (auto TargetId : C->Targets) {
    for (auto Succ : State.Cg->Targets[TargetId].Succs) {
      auto SuccCluster = State.FuncCluster[Succ];
      if (SuccCluster != nullptr && SuccCluster != C) {
        Result.insert(SuccCluster);
      }
    }
    for (auto Pred : State.Cg->Targets[TargetId].Preds) {
      auto PredCluster = State.FuncCluster[Pred];
      if (PredCluster != nullptr && PredCluster != C) {
        Result.insert(PredCluster);
      }
    }
  }
  return Result;
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
double shortCalls(const AlgoState &State, Cluster *Cluster) {
  double Calls = 0;
  for (auto TargetId : Cluster->Targets) {
    for (auto Succ : State.Cg->Targets[TargetId].Succs) {
      if (State.FuncCluster[Succ] == Cluster) {
        auto A = State.Cg->Arcs.find(Arc(TargetId, Succ));

        auto SrcAddr = State.Addr[TargetId] + A->AvgCallOffset;
        auto DstAddr = State.Addr[Succ];

        Calls += expectedCalls(SrcAddr, DstAddr, A->Weight);
      }
    }
  }

  return Calls;
}

/*
 * The number of calls between the two clusters with both endpoints on
 * the same TLB page, assuming that a given pair of clusters gets merged
 */
double shortCalls(const AlgoState &State,
                  Cluster *ClusterPred,
                  Cluster *ClusterSucc) {
  double Calls = 0;
  for (auto TargetId : ClusterPred->Targets) {
    for (auto Succ : State.Cg->Targets[TargetId].Succs) {
      if (State.FuncCluster[Succ] == ClusterSucc) {
        auto A = State.Cg->Arcs.find(Arc(TargetId, Succ));

        auto SrcAddr = State.Addr[TargetId] + A->AvgCallOffset;
        auto DstAddr = State.Addr[Succ] + ClusterPred->Size;

        Calls += expectedCalls(SrcAddr, DstAddr, A->Weight);
      }
    }
  }

  for (auto TargetId : ClusterPred->Targets) {
    for (auto Pred : State.Cg->Targets[TargetId].Preds) {
      if (State.FuncCluster[Pred] == ClusterSucc) {
        auto A = State.Cg->Arcs.find(Arc(Pred, TargetId));

        auto SrcAddr = State.Addr[Pred] + A->AvgCallOffset +
          ClusterPred->Size;
        auto DstAddr = State.Addr[TargetId];

        Calls += expectedCalls(SrcAddr, DstAddr, A->Weight);
      }
    }
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
double mergeGain(const AlgoState &State,
                 Cluster *ClusterPred,
                 Cluster *ClusterSucc) {
  // cache misses on the first cluster
  double LongCallsPred = ClusterPred->Samples - shortCalls(State, ClusterPred);
  double ProbPred = missProbability(State, ClusterPred->density() * PageSize);
  double ExpectedMissesPred = LongCallsPred * ProbPred;

  // cache misses on the second cluster
  double LongCallsSucc = ClusterSucc->Samples - shortCalls(State, ClusterSucc);
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
  return Gain / (ClusterPred->Size + ClusterSucc->Size);
}

 /*
  * Merge two clusters
  */
void mergeInto(AlgoState &State, Cluster *Into, Cluster *Other) {
  auto &Targets = Other->Targets;
  Into->Targets.insert(Into->Targets.end(), Targets.begin(), Targets.end());
  Into->Size += Other->Size;
  Into->Samples += Other->Samples;

  size_t CurAddr = 0;
  for (auto TargetId : Into->Targets) {
    State.FuncCluster[TargetId] = Into;
    State.Addr[TargetId] = CurAddr;
    CurAddr += State.Cg->Targets[TargetId].Size;
  }

  Other->Size = 0;
  Other->Samples = 0;
  Other->Targets.clear();
}

/*
 * HFSortPlus - layout of hot functions with iTLB cache optimization
 */
std::vector<Cluster> hfsortPlus(const TargetGraph &Cg) {
  // create a cluster for every function
  std::vector<Cluster> AllClusters;
  AllClusters.reserve(Cg.Targets.size());
  for (TargetId F = 0; F < Cg.Targets.size(); F++) {
    AllClusters.emplace_back(F, Cg.Targets[F]);
  }

  // initialize objects used by the algorithm
  std::vector<Cluster *> Clusters;
  Clusters.reserve(Cg.Targets.size());
  AlgoState State;
  State.Cg = &Cg;
  State.TotalSamples = 0;
  State.FuncCluster = std::vector<Cluster *>(Cg.Targets.size(), nullptr);
  State.Addr = std::vector<size_t>(Cg.Targets.size(), InvalidAddr);
  for (TargetId F = 0; F < Cg.Targets.size(); F++) {
    if (Cg.Targets[F].Samples == 0) continue;

    Clusters.push_back(&AllClusters[F]);
    State.FuncCluster[F] = &AllClusters[F];
    State.Addr[F] = 0;
    State.TotalSamples += Cg.Targets[F].Samples;
  }

  DEBUG(dbgs() << "Starting hfsort+ for " << Clusters.size() << " clusters\n"
               << format("Initial expected iTLB cache hit ratio: %.4lf\n",
                         expectedCacheHitRatio(State, Clusters)));

  // the cache keeps precomputed values of mergeGain for pairs of clusters;
  // when a pair of clusters (x,y) gets merged, we need to invalidate the pairs
  // containing both x and y (and recompute them on the next iteration)
  PrecomputedResults Cache;

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
    Steps++;

    Cluster *BestClusterPred = nullptr;
    Cluster *BestClusterSucc = nullptr;
    double BestGain = -1;
    for (auto ClusterPred : Clusters) {
      // get candidates for merging with the current cluster
      auto CandidateClusters = adjacentClusters(State, ClusterPred);

      // find the best candidate
      for (auto ClusterSucc : CandidateClusters) {
        // get a cost of merging two clusters
        if (!Cache.contains(ClusterPred, ClusterSucc)) {
          double Value = mergeGain(State, ClusterPred, ClusterSucc);
          Cache.set(ClusterPred, ClusterSucc, Value);
          assert(Cache.contains(ClusterPred, ClusterSucc));
        }

        double Gain = Cache.get(ClusterPred, ClusterSucc);
        // breaking ties by density to make the hottest clusters be merged first
        if (Gain > BestGain || (std::abs(Gain - BestGain) < 1e-8 &&
                                density(ClusterPred, ClusterSucc) >
                                density(BestClusterPred, BestClusterSucc))) {
          BestGain = Gain;
          BestClusterPred = ClusterPred;
          BestClusterSucc = ClusterSucc;
        }
      }
    }
    Cache.validateAll();

    if (BestGain <= 0.0) break;

    Cache.invalidate(BestClusterPred);
    Cache.invalidate(BestClusterSucc);

    // merge the best pair of clusters
    mergeInto(State, BestClusterPred, BestClusterSucc);
    // remove BestClusterSucc from the list of active clusters
    auto Iter = std::remove(Clusters.begin(), Clusters.end(), BestClusterSucc);
    Clusters.erase(Iter, Clusters.end());
  }

  DEBUG(dbgs() << "Completed hfsort+ with " << Clusters.size() << " clusters\n"
               << format("Final expected iTLB cache hit ratio: %.4lf\n",
                         expectedCacheHitRatio(State, Clusters)));

  // Return the set of clusters that are left, which are the ones that
  // didn't get merged (so their first func is its original func).
  sortByDensity(Clusters);
  std::vector<Cluster> Result;
  for (auto Cluster : Clusters) {
    Result.emplace_back(std::move(*Cluster));
  }

  std::sort(Result.begin(), Result.end(), compareClustersDensity);

  return Result;
}

}}
