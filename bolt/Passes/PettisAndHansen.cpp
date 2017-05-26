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

void orderFuncs(const CallGraph &Cg, Cluster *C1, Cluster *C2) {
  auto C1head = C1->Targets.front();
  auto C1tail = C1->Targets.back();
  auto C2head = C2->Targets.front();
  auto C2tail = C2->Targets.back();

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

std::vector<Cluster> pettisAndHansen(const CallGraph &Cg) {
  // indexed by NodeId, keeps its current cluster
  std::vector<Cluster*> FuncCluster(Cg.Nodes.size(), nullptr);
  std::vector<Cluster> Clusters;
  std::vector<NodeId> Funcs;

  Clusters.reserve(Cg.Nodes.size());

  for (NodeId F = 0; F < Cg.Nodes.size(); F++) {
    if (Cg.Nodes[F].Samples == 0) continue;
    Clusters.emplace_back(F, Cg.Nodes[F]);
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
    C1->merge(std::move(*C2), Max.Weight);
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

}
}
