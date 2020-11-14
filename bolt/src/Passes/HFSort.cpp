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
#include "llvm/Support/Options.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_set>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "hfsort"

namespace opts {
extern llvm::cl::opt<unsigned> Verbosity;
}

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

Cluster::Cluster(NodeId Id, const Node &Func)
: Samples(Func.samples()),
  Size(Func.size()),
  Density((double)Samples / Size) {
  Targets.push_back(Id);
}

Cluster::Cluster(const std::vector<NodeId> &Nodes, const CallGraph &Cg) {
  Samples = 0;
  Size = 0;
  for (auto TargetId : Nodes) {
    Targets.push_back(TargetId);
    Samples += Cg.samples(TargetId);
    Size += Cg.size(TargetId);
  }
  Density = (double)Samples / Size;
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
    uint32_t NewSize = TotalSize + C.size();
    if (NewSize > FrozenPages * HugePageSize) break;
    C.freeze();
    TotalSize = NewSize;
    DEBUG(
      auto Fid = C.target(0);
      dbgs() <<
          format("freezing cluster for func %d, size = %u, samples = %lu)\n",
                 Fid, Cg.size(Fid), Cg.samples(Fid)););
  }
}

}

void Cluster::reverseTargets() {
  std::reverse(Targets.begin(), Targets.end());
}

void Cluster::merge(const Cluster &Other, const double Aw) {
  Targets.insert(Targets.end(),
                 Other.Targets.begin(),
                 Other.Targets.end());
  Size += Other.Size;
  Samples += Other.Samples;
  Density = (double)Samples / Size;
}

void Cluster::merge(const Cluster &Other, const std::vector<CallGraph::NodeId> &Targets_) {
  Targets = Targets_;
  Size += Other.Size;
  Samples += Other.Samples;
  Density = (double)Samples / Size;
}

void Cluster::clear() {
  Id = -1u;
  Size = 0;
  Samples = 0;
  Density = 0.0;
  Targets.clear();
  Frozen = false;
}

std::vector<Cluster> clusterize(const CallGraph &Cg) {
  std::vector<NodeId> SortedFuncs;

  // indexed by NodeId, keeps it's current cluster
  std::vector<Cluster*> FuncCluster(Cg.numNodes(), nullptr);
  std::vector<Cluster> Clusters;
  Clusters.reserve(Cg.numNodes());

  for (NodeId F = 0; F < Cg.numNodes(); F++) {
    if (Cg.samples(F) == 0) continue;
    Clusters.emplace_back(F, Cg.getNode(F));
    SortedFuncs.push_back(F);
  }

  freezeClusters(Cg, Clusters);

  // The size and order of Clusters is fixed until we reshuffle it immediately
  // before returning.
  for (auto &Cluster : Clusters) {
    FuncCluster[Cluster.targets().front()] = &Cluster;
  }

  std::sort(
    SortedFuncs.begin(),
    SortedFuncs.end(),
    [&] (const NodeId F1, const NodeId F2) {
      const auto &Func1 = Cg.getNode(F1);
      const auto &Func2 = Cg.getNode(F2);
      return
        Func1.samples() * Func2.size() >  // TODO: is this correct?
        Func2.samples() * Func1.size();
    }
  );

  // Process each function, and consider merging its cluster with the
  // one containing its most likely predecessor.
  for (const auto Fid : SortedFuncs) {
    auto Cluster = FuncCluster[Fid];
    if (Cluster->frozen()) continue;

    // Find best predecessor.
    NodeId BestPred = CallGraph::InvalidId;
    double BestProb = 0;

    for (const auto Src : Cg.predecessors(Fid)) {
      const auto &Arc = *Cg.findArc(Src, Fid);
      if (BestPred == CallGraph::InvalidId || Arc.normalizedWeight() > BestProb) {
        BestPred = Arc.src();
        BestProb = Arc.normalizedWeight();
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
        PredCluster->frozen()) {
      continue;
    }

    // Skip if merged cluster would be bigger than the threshold.
    if (Cluster->size() + PredCluster->size() > MaxClusterSize) continue;

    // Check if the merge is good for the caller.
    //   Don't merge if the caller's density is significantly better
    //   than the density resulting from the merge.
    const double NewDensity =
      ((double)PredCluster->samples() + Cluster->samples()) /
      (PredCluster->size() + Cluster->size());
    if (PredCluster->density() > NewDensity * CallerDegradeFactor) {
      continue;
    }

    DEBUG(
      if (opts::Verbosity > 1) {
        dbgs() << format("merging %s -> %s: %u\n",
                         PredCluster->toString().c_str(),
                         Cluster->toString().c_str(),
                         Cg.samples(Fid));
      });

    for (auto F : Cluster->targets()) {
      FuncCluster[F] = PredCluster;
    }

    PredCluster->merge(*Cluster);
    Cluster->clear();
  }

  // Return the set of Clusters that are left, which are the ones that
  // didn't get merged (so their first func is its original func).
  std::vector<Cluster> SortedClusters;
  std::unordered_set<Cluster *> Visited;
  for (const auto Func : SortedFuncs) {
    auto Cluster = FuncCluster[Func];
    if (!Cluster ||
        Visited.count(Cluster) == 1 ||
        Cluster->target(0) != Func) {
      continue;
    }
    SortedClusters.emplace_back(std::move(*Cluster));
    Visited.insert(Cluster);
  }

  std::sort(SortedClusters.begin(),
            SortedClusters.end(),
            compareClustersDensity);

  return SortedClusters;
}

std::vector<Cluster> randomClusters(const CallGraph &Cg) {
  std::vector<NodeId> FuncIds(Cg.numNodes(), 0);
  std::vector<Cluster> Clusters;
  Clusters.reserve(Cg.numNodes());

  for (NodeId F = 0; F < Cg.numNodes(); F++) {
    if (Cg.samples(F) == 0) continue;
    Clusters.emplace_back(F, Cg.getNode(F));
  }

  std::sort(Clusters.begin(),
            Clusters.end(),
            [](const Cluster &A, const Cluster &B) {
              return A.size() < B.size();
            });

  auto pickMergeCluster = [&Clusters](const size_t Idx) {
    size_t MaxIdx = Idx + 1;

    while (MaxIdx < Clusters.size() &&
           Clusters[Idx].size() + Clusters[MaxIdx].size() <= MaxClusterSize) {
      ++MaxIdx;
    }

    if (MaxIdx - Idx > 1) {
      size_t MergeIdx = (std::rand() % (MaxIdx - Idx - 1)) + Idx + 1;
      assert(Clusters[MergeIdx].size() + Clusters[Idx].size() <= MaxClusterSize);
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
      Clusters[Idx].merge(Clusters[MergeIdx]);
      Clusters.erase(Clusters.begin() + MergeIdx);
    }
  }

  return Clusters;
}

}
}
