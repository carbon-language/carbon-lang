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

using NodeId = CallGraph::NodeId;
using Arc = CallGraph::Arc;
using Node = CallGraph::Node;  

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

}

////////////////////////////////////////////////////////////////////////////////

Cluster::Cluster(NodeId Id, const Node &Func) {
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

void freezeClusters(const CallGraph &Cg, std::vector<Cluster> &Clusters) {
  uint32_t TotalSize = 0;
  std::sort(Clusters.begin(), Clusters.end(), compareClustersDensity);
  for (auto &C : Clusters) {
    uint32_t NewSize = TotalSize + C.Size;
    if (NewSize > FrozenPages * HugePageSize) break;
    C.Frozen = true;
    TotalSize = NewSize;
    auto Fid = C.Targets[0];
    DEBUG(dbgs() <<
          format("freezing cluster for func %d, size = %u, samples = %u)\n",
                 Fid, Cg.Nodes[Fid].Size, Cg.Nodes[Fid].Samples););
  }
}

}

void Cluster::merge(Cluster&& Other, const double Aw) {
  Targets.insert(Targets.end(),
                 Other.Targets.begin(),
                 Other.Targets.end());
  Size += Other.Size;
  Samples += Other.Samples;

  Other.Size = 0;
  Other.Samples = 0;
  Other.Targets.clear();
}

std::vector<Cluster> clusterize(const CallGraph &Cg) {
  std::vector<NodeId> SortedFuncs;

  // indexed by NodeId, keeps it's current cluster
  std::vector<Cluster*> FuncCluster(Cg.Nodes.size(), nullptr);
  std::vector<Cluster> Clusters;
  Clusters.reserve(Cg.Nodes.size());

  for (NodeId F = 0; F < Cg.Nodes.size(); F++) {
    if (Cg.Nodes[F].Samples == 0) continue;
    Clusters.emplace_back(F, Cg.Nodes[F]);
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
    [&] (const NodeId F1, const NodeId F2) {
      const auto &Func1 = Cg.Nodes[F1];
      const auto &Func2 = Cg.Nodes[F2];
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
    NodeId BestPred = CallGraph::InvalidId;
    double BestProb = 0;

    for (const auto Src : Cg.Nodes[Fid].Preds) {
      auto &A = *Cg.Arcs.find(Arc(Src, Fid));
      if (BestPred == CallGraph::InvalidId || A.NormalizedWeight > BestProb) {
        BestPred = A.Src;
        BestProb = A.NormalizedWeight;
      }
    }

    // Check if the merge is good for the callee.
    //   Don't merge if the probability of getting to the callee from the
    //   caller is too low.
    if (BestProb < MinArcProbability) continue;

    assert(BestPred != CallGraph::InvalidId);

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
                           Cg.Nodes[Fid].Samples););

    for (auto F : Cluster->Targets) {
      FuncCluster[F] = PredCluster;
    }

    PredCluster->merge(std::move(*Cluster));
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

std::vector<Cluster> randomClusters(const CallGraph &Cg) {
  std::vector<NodeId> FuncIds(Cg.Nodes.size(), 0);
  std::vector<Cluster> Clusters;
  Clusters.reserve(Cg.Nodes.size());  

  for (NodeId F = 0; F < Cg.Nodes.size(); F++) {
    if (Cg.Nodes[F].Samples == 0) continue;
    Clusters.emplace_back(F, Cg.Nodes[F]);
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
      Clusters[Idx].merge(std::move(Clusters[MergeIdx]));
      Clusters.erase(Clusters.begin() + MergeIdx);
    }
  }

  return Clusters;
}

}
}
