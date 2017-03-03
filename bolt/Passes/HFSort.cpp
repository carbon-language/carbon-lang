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
   | Copyright (c) 2010-2016 Facebook, Inc. (http://www.facebook.com)     |
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <unordered_map>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "hfsort"

namespace llvm {
namespace bolt {

namespace {

// The number of pages to reserve for the functions with highest
// density (samples / size).  The functions put in these pages are not
// considered for clustering.
constexpr uint32_t FrozenPages = 0;

// The minimum approximate probability of a callee being called from a
// particular arc to consider merging with the caller's cluster.
constexpr double MinArcProbability = 0.1;

// This is a factor to determine by how much a caller cluster is
// willing to degrade it's density by merging a callee.
constexpr int CallerDegradeFactor = 8;

// Maximum size of a cluster, in bytes.
constexpr uint32_t MaxClusterSize = 1 << 20;

constexpr uint32_t PageSize = 2 << 20;

}
////////////////////////////////////////////////////////////////////////////////

TargetId TargetGraph::addTarget(uint32_t Size, uint32_t Samples) {
  auto Id = Targets.size();
  Targets.emplace_back(Size, Samples);
  return Id;
}

const Arc &TargetGraph::incArcWeight(TargetId Src, TargetId Dst, double W) {
  auto Res = Arcs.emplace(Src, Dst, W);
  if (!Res.second) {
    Res.first->Weight += W;
    return *Res.first;
  }
  Targets[Src].Succs.push_back(Dst);
  Targets[Dst].Preds.push_back(Src);
  return *Res.first;
}

Cluster::Cluster(TargetId Id, const TargetNode &Func) {
  Targets.push_back(Id);
  Size = Func.Size;
  Samples = Func.Samples;
  Frozen = false;
  DEBUG(dbgs() << "new Cluster: " << toString() << "\n");
}

std::string Cluster::toString() const {
  std::string Str;
  raw_string_ostream CS(Str);
  bool PrintComma = false;
  CS << "funcs = [";
  for (auto &Target : Targets) {
    if (PrintComma) CS << ", ";
    CS << Target;
    PrintComma = true;
  }
  CS << "]";
  return CS.str();
}

namespace {
////////////////////////////////////////////////////////////////////////////////

bool compareClustersDensity(const Cluster &C1, const Cluster &C2) {
  return C1.density() > C2.density();
}

////////////////////////////////////////////////////////////////////////////////

void freezeClusters(const TargetGraph &Cg, std::vector<Cluster> &Clusters) {
  uint32_t TotalSize = 0;
  std::sort(Clusters.begin(), Clusters.end(), compareClustersDensity);
  for (auto &C : Clusters) {
    uint32_t NewSize = TotalSize + C.Size;
    if (NewSize > FrozenPages * PageSize) break;
    C.Frozen = true;
    TotalSize = NewSize;
    auto Fid = C.Targets[0];
    DEBUG(dbgs() <<
          format("freezing cluster for func %d, size = %u, samples = %u)\n",
                 Fid, Cg.Targets[Fid].Size, Cg.Targets[Fid].Samples););
  }
}

void mergeInto(Cluster &Into, Cluster&& Other, const double Aw = 0) {
  Into.Targets.insert(Into.Targets.end(),
                      Other.Targets.begin(),
                      Other.Targets.end());
  Into.Size += Other.Size;
  Into.Samples += Other.Samples;

  Other.Size = 0;
  Other.Samples = 0;
  Other.Targets.clear();
}
}

std::vector<Cluster> clusterize(const TargetGraph &Cg) {
  std::vector<TargetId> SortedFuncs;

  // indexed by TargetId, keeps it's current cluster
  std::vector<Cluster*> FuncCluster(Cg.Targets.size(), nullptr);
  std::vector<Cluster> Clusters;
  Clusters.reserve(Cg.Targets.size());

  for (TargetId F = 0; F < Cg.Targets.size(); F++) {
    if (Cg.Targets[F].Samples == 0) continue;
    Clusters.emplace_back(F, Cg.Targets[F]);
    SortedFuncs.push_back(F);
  }

  freezeClusters(Cg, Clusters);

  // The size and order of Clusters is fixed until we reshuffle it immediately
  // before returning.
  for (auto &Cluster : Clusters) {
    FuncCluster[Cluster.Targets.front()] = &Cluster;
  }

  std::sort(
    SortedFuncs.begin(),
    SortedFuncs.end(),
    [&] (const TargetId F1, const TargetId F2) {
      const auto &Func1 = Cg.Targets[F1];
      const auto &Func2 = Cg.Targets[F2];
      return
        (uint64_t)Func1.Samples * Func2.Size >  // TODO: is this correct?
        (uint64_t)Func2.Samples * Func1.Size;
    }
  );

  // Process each function, and consider merging its cluster with the
  // one containing its most likely predecessor.
  for (const auto Fid : SortedFuncs) {
    auto Cluster = FuncCluster[Fid];
    if (Cluster->Frozen) continue;

    // Find best predecessor.
    TargetId BestPred = InvalidId;
    double BestProb = 0;

    for (const auto Src : Cg.Targets[Fid].Preds) {
      auto &A = *Cg.Arcs.find(Arc(Src, Fid));
      if (BestPred == InvalidId || A.NormalizedWeight > BestProb) {
        BestPred = A.Src;
        BestProb = A.NormalizedWeight;
      }
    }

    // Check if the merge is good for the callee.
    //   Don't merge if the probability of getting to the callee from the
    //   caller is too low.
    if (BestProb < MinArcProbability) continue;

    assert(BestPred != InvalidId);

    auto PredCluster = FuncCluster[BestPred];

    // Skip if no predCluster (predecessor w/ no samples), or if same
    // as cluster, of it's frozen.
    if (PredCluster == nullptr || PredCluster == Cluster ||
        PredCluster->Frozen) {
      continue;
    }

    // Skip if merged cluster would be bigger than the threshold.
    if (Cluster->Size + PredCluster->Size > MaxClusterSize) continue;

    // Check if the merge is good for the caller.
    //   Don't merge if the caller's density is significantly better
    //   than the density resulting from the merge.
    const double NewDensity =
      ((double)PredCluster->Samples + Cluster->Samples) /
      (PredCluster->Size + Cluster->Size);
    if (PredCluster->density() > NewDensity * CallerDegradeFactor) {
      continue;
    }

    DEBUG(dbgs() << format("merging %s -> %s: %u\n",
                           PredCluster->toString().c_str(),
                           Cluster->toString().c_str(),
                           Cg.Targets[Fid].Samples););

    for (auto F : Cluster->Targets) {
      FuncCluster[F] = PredCluster;
    }

    mergeInto(*PredCluster, std::move(*Cluster));
  }

  // Return the set of Clusters that are left, which are the ones that
  // didn't get merged (so their first func is its original func).
  std::vector<Cluster> SortedClusters;
  for (const auto Func : SortedFuncs) {
    auto Cluster = FuncCluster[Func];
    if (!Cluster || Cluster->Targets.empty()) continue;
    if (Cluster->Targets[0] != Func) continue;
    SortedClusters.emplace_back(std::move(*Cluster));
    Cluster->Targets.clear();
  }

  std::sort(SortedClusters.begin(),
            SortedClusters.end(),
            compareClustersDensity);

  return SortedClusters;
}

////////////////////////////////////////////////////////////////////////////////

namespace {
class ClusterArc {
public:
  ClusterArc(Cluster *Ca, Cluster *Cb, double W = 0)
    : C1(std::min(Ca, Cb))
    , C2(std::max(Ca, Cb))
    , Weight(W)
  {}

  friend bool operator==(const ClusterArc &Lhs, const ClusterArc &Rhs) {
    return Lhs.C1 == Rhs.C1 && Lhs.C2 == Rhs.C2;
  }

  Cluster *const C1;
  Cluster *const C2;
  mutable double Weight;
};

class ClusterArcHash {
public:
  int64_t operator()(const ClusterArc &Arc) const {
    std::hash<int64_t> Hasher;
    return hashCombine(Hasher(int64_t(Arc.C1)), int64_t(Arc.C2));
  }
};

using ClusterArcSet = std::unordered_set<ClusterArc, ClusterArcHash>;

void orderFuncs(const TargetGraph &Cg, Cluster *C1, Cluster *C2) {
  TargetId C1head = C1->Targets.front();
  TargetId C1tail = C1->Targets.back();
  TargetId C2head = C2->Targets.front();
  TargetId C2tail = C2->Targets.back();

  double C1headC2head = 0;
  double C1headC2tail = 0;
  double C1tailC2head = 0;
  double C1tailC2tail = 0;

  for (const auto &Arc : Cg.Arcs) {
    if ((Arc.Src == C1head && Arc.Dst == C2head) ||
        (Arc.Dst == C1head && Arc.Src == C2head)) {
      C1headC2head += Arc.Weight;
    } else if ((Arc.Src == C1head && Arc.Dst == C2tail) ||
               (Arc.Dst == C1head && Arc.Src == C2tail)) {
      C1headC2tail += Arc.Weight;
    } else if ((Arc.Src == C1tail && Arc.Dst == C2head) ||
               (Arc.Dst == C1tail && Arc.Src == C2head)) {
      C1tailC2head += Arc.Weight;
    } else if ((Arc.Src == C1tail && Arc.Dst == C2tail) ||
               (Arc.Dst == C1tail && Arc.Src == C2tail)) {
      C1tailC2tail += Arc.Weight;
    }
  }

  const double Max = std::max(std::max(C1headC2head, C1headC2tail),
                              std::max(C1tailC2head, C1tailC2tail));

  if (C1headC2head == Max) {
    // flip C1
    std::reverse(C1->Targets.begin(), C1->Targets.end());
  } else if (C1headC2tail == Max) {
    // flip C1 C2
    std::reverse(C1->Targets.begin(), C1->Targets.end());
    std::reverse(C2->Targets.begin(), C2->Targets.end());
  } else if (C1tailC2tail == Max) {
    // flip C2
    std::reverse(C2->Targets.begin(), C2->Targets.end());
  }
}
}

std::vector<Cluster> pettisAndHansen(const TargetGraph &Cg) {
  // indexed by TargetId, keeps its current cluster
  std::vector<Cluster*> FuncCluster(Cg.Targets.size(), nullptr);
  std::vector<Cluster> Clusters;
  std::vector<TargetId> Funcs;

  Clusters.reserve(Cg.Targets.size());

  for (TargetId F = 0; F < Cg.Targets.size(); F++) {
    if (Cg.Targets[F].Samples == 0) continue;
    Clusters.emplace_back(F, Cg.Targets[F]);
    FuncCluster[F] = &Clusters.back();
    Funcs.push_back(F);
  }

  ClusterArcSet Carcs;

  auto insertOrInc = [&](Cluster *C1, Cluster *C2, double Weight) {
    auto Res = Carcs.emplace(C1, C2, Weight);
    if (!Res.second) {
      Res.first->Weight += Weight;
    }
  };

  // Create a std::vector of cluster arcs

  for (auto &Arc : Cg.Arcs) {
    if (Arc.Weight == 0) continue;

    auto const S = FuncCluster[Arc.Src];
    auto const D = FuncCluster[Arc.Dst];

    // ignore if s or d is nullptr

    if (S == nullptr || D == nullptr) continue;

    // ignore self-edges

    if (S == D) continue;

    insertOrInc(S, D, Arc.Weight);
  }

  // Find an arc with max weight and merge its nodes

  while (!Carcs.empty()) {
    auto Maxpos = std::max_element(
      Carcs.begin(),
      Carcs.end(),
      [&] (const ClusterArc &Carc1, const ClusterArc &Carc2) {
        return Carc1.Weight < Carc2.Weight;
      }
    );

    auto Max = *Maxpos;
    Carcs.erase(Maxpos);

    auto const C1 = Max.C1;
    auto const C2 = Max.C2;

    if (C1->Size + C2->Size > MaxClusterSize) continue;

    if (C1->Frozen || C2->Frozen) continue;

    // order functions and merge cluster

    orderFuncs(Cg, C1, C2);

    DEBUG(dbgs() << format("merging %s -> %s: %.1f\n", C2->toString().c_str(),
          C1->toString().c_str(), Max.Weight););

    // update carcs: merge C1arcs to C2arcs

    std::unordered_map<ClusterArc, Cluster *, ClusterArcHash> C2arcs;
    for (auto &Carc : Carcs) {
      if (Carc.C1 == C2) C2arcs.emplace(Carc, Carc.C2);
      if (Carc.C2 == C2) C2arcs.emplace(Carc, Carc.C1);
    }

    for (auto It : C2arcs) {
      auto const C = It.second;
      auto const C2arc = It.first;

      insertOrInc(C, C1, C2arc.Weight);
      Carcs.erase(C2arc);
    }

    // update FuncCluster

    for (auto F : C2->Targets) {
      FuncCluster[F] = C1;
    }
    mergeInto(*C1, std::move(*C2), Max.Weight);
  }

  // Return the set of Clusters that are left, which are the ones that
  // didn't get merged.

  std::set<Cluster*> LiveClusters;
  std::vector<Cluster> OutClusters;

  for (auto Fid : Funcs) {
    LiveClusters.insert(FuncCluster[Fid]);
  }
  for (auto C : LiveClusters) {
    OutClusters.push_back(std::move(*C));
  }

  std::sort(OutClusters.begin(),
            OutClusters.end(),
            compareClustersDensity);

  return OutClusters;
}

std::vector<Cluster> randomClusters(const TargetGraph &Cg) {
  std::vector<TargetId> FuncIds(Cg.Targets.size(), 0);
  std::vector<Cluster> Clusters;
  Clusters.reserve(Cg.Targets.size());  

  for (TargetId F = 0; F < Cg.Targets.size(); F++) {
    if (Cg.Targets[F].Samples == 0) continue;
    Clusters.emplace_back(F, Cg.Targets[F]);
  }

  std::sort(Clusters.begin(),
            Clusters.end(),
            [](const Cluster &A, const Cluster &B) {
              return A.Size < B.Size;
            });

  auto pickMergeCluster = [&Clusters](const size_t Idx) {
    size_t MaxIdx = Idx + 1;

    while (MaxIdx < Clusters.size() &&
           Clusters[Idx].Size + Clusters[MaxIdx].Size <= MaxClusterSize) {
      ++MaxIdx;
    }

    if (MaxIdx - Idx > 1) {
      size_t MergeIdx = (std::rand() % (MaxIdx - Idx - 1)) + Idx + 1;
      assert(Clusters[MergeIdx].Size + Clusters[Idx].Size <= MaxClusterSize);
      return MergeIdx;
    }
    return Clusters.size();
  };

  size_t Idx = 0;
  while (Idx < Clusters.size()) {
    auto MergeIdx = pickMergeCluster(Idx);
    if (MergeIdx == Clusters.size()) {
      ++Idx;
    } else {
      mergeInto(Clusters[Idx], std::move(Clusters[MergeIdx]));
      Clusters.erase(Clusters.begin() + MergeIdx);
    }
  }

  return Clusters;
}

}
}
