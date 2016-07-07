//===--- ReorderAlgorithm.cpp - Basic block reorderng algorithms ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements different basic block reordering algorithms.
//
//===----------------------------------------------------------------------===//

#include "ReorderAlgorithm.h"
#include "BinaryBasicBlock.h"
#include "BinaryFunction.h"
#include "llvm/Support/CommandLine.h"
#include <queue>
#include <functional>

using namespace llvm;
using namespace bolt;

namespace opts {

static cl::opt<bool>
PrintClusters("print-clusters", cl::desc("print clusters"), cl::Optional);

} // namespace opts

namespace {

template <class T>
inline void hashCombine(size_t &Seed, const T &Val) {
  std::hash<T> Hasher;
  Seed ^= Hasher(Val) + 0x9e3779b9 + (Seed << 6) + (Seed >> 2);
}

template <typename A, typename B>
struct HashPair {
  size_t operator()(const std::pair<A,B>& Val) const {
    std::hash<A> Hasher;
    size_t Seed = Hasher(Val.first);
    hashCombine(Seed, Val.second);
    return Seed;
  }
};

}

void ClusterAlgorithm::computeClusterAverageFrequency() {
  AvgFreq.resize(Clusters.size(), 0.0);
  for (uint32_t I = 0, E = Clusters.size(); I < E; ++I) {
    double Freq = 0.0;
    for (auto BB : Clusters[I]) {
      if (!BB->empty() && BB->size() != BB->getNumPseudos())
        Freq += ((double) BB->getExecutionCount()) /
                (BB->size() - BB->getNumPseudos());
    }
    AvgFreq[I] = Freq;
  }
}

void ClusterAlgorithm::printClusters() const {
  for (uint32_t I = 0, E = Clusters.size(); I < E; ++I) {
    errs() << "Cluster number " << I;
    if (AvgFreq.size() == Clusters.size())
      errs() << " (frequency: " << AvgFreq[I] << ")";
    errs() << " : ";
    auto Sep = "";
    for (auto BB : Clusters[I]) {
      errs() << Sep << BB->getName();
      Sep = ", ";
    }
    errs() << "\n";
  }
}

void ClusterAlgorithm::reset() {
  Clusters.clear();
  ClusterEdges.clear();
  AvgFreq.clear();
}

void GreedyClusterAlgorithm::clusterBasicBlocks(const BinaryFunction &BF) {
  reset();

  // Greedy heuristic implementation for the TSP, applied to BB layout. Try to
  // maximize weight during a path traversing all BBs. In this way, we will
  // convert the hottest branches into fall-throughs.

  // Encode an edge between two basic blocks, source and destination
  typedef std::pair<BinaryBasicBlock *, BinaryBasicBlock *> EdgeTy;
  typedef HashPair<BinaryBasicBlock *, BinaryBasicBlock *> Hasher;
  std::unordered_map<EdgeTy, uint64_t, Hasher> Weight;

  // Define a comparison function to establish SWO between edges
  auto Comp = [&] (EdgeTy A, EdgeTy B) {
    // With equal weights, prioritize branches with lower index
    // source/destination. This helps to keep original block order for blocks
    // when optimal order cannot be deducted from a profile.
    if (Weight[A] == Weight[B]) {
      uint32_t ASrcBBIndex = BF.getIndex(A.first);
      uint32_t BSrcBBIndex = BF.getIndex(B.first);
      if (ASrcBBIndex != BSrcBBIndex)
        return ASrcBBIndex > BSrcBBIndex;
      return BF.getIndex(A.second) > BF.getIndex(B.second);
    }
    return Weight[A] < Weight[B];
  };
  std::priority_queue<EdgeTy, std::vector<EdgeTy>, decltype(Comp)> Queue(Comp);

  typedef std::unordered_map<BinaryBasicBlock *, int> BBToClusterMapTy;
  BBToClusterMapTy BBToClusterMap;

  ClusterEdges.resize(BF.layout_size());

  for (auto BB : BF.layout()) {
    // Create a cluster for this BB
    uint32_t I = Clusters.size();
    Clusters.emplace_back();
    auto &Cluster = Clusters.back();
    Cluster.push_back(BB);
    BBToClusterMap[BB] = I;
    // Populate priority queue with edges
    auto BI = BB->branch_info_begin();
    for (auto &I : BB->successors()) {
      if (BI->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE)
        Weight[std::make_pair(BB, I)] = BI->Count;
      Queue.push(std::make_pair(BB, I));
      ++BI;
    }
  }

  // Grow clusters in a greedy fashion
  while (!Queue.empty()) {
    auto elmt = Queue.top();
    Queue.pop();

    BinaryBasicBlock *BBSrc = elmt.first;
    BinaryBasicBlock *BBDst = elmt.second;

    // Case 1: BBSrc and BBDst are the same. Ignore this edge
    if (BBSrc == BBDst || BBDst == *BF.layout_begin())
      continue;

    int I = BBToClusterMap[BBSrc];
    int J = BBToClusterMap[BBDst];

    // Case 2: If they are already allocated at the same cluster, just increase
    // the weight of this cluster
    if (I == J) {
      ClusterEdges[I][I] += Weight[elmt];
      continue;
    }

    auto &ClusterA = Clusters[I];
    auto &ClusterB = Clusters[J];
    if (ClusterA.back() == BBSrc && ClusterB.front() == BBDst) {
      // Case 3: BBSrc is at the end of a cluster and BBDst is at the start,
      // allowing us to merge two clusters
      for (auto BB : ClusterB)
        BBToClusterMap[BB] = I;
      ClusterA.insert(ClusterA.end(), ClusterB.begin(), ClusterB.end());
      ClusterB.clear();
      // Iterate through all inter-cluster edges and transfer edges targeting
      // cluster B to cluster A.
      // It is bad to have to iterate though all edges when we could have a list
      // of predecessors for cluster B. However, it's not clear if it is worth
      // the added code complexity to create a data structure for clusters that
      // maintains a list of predecessors. Maybe change this if it becomes a
      // deal breaker.
      for (uint32_t K = 0, E = ClusterEdges.size(); K != E; ++K)
        ClusterEdges[K][I] += ClusterEdges[K][J];
    } else {
      // Case 4: Both BBSrc and BBDst are allocated in positions we cannot
      // merge them. Annotate the weight of this edge in the weight between
      // clusters to help us decide ordering between these clusters.
      ClusterEdges[I][J] += Weight[elmt];
    }
  }
}

void OptimalReorderAlgorithm::reorderBasicBlocks(
      const BinaryFunction &BF, BasicBlockOrder &Order) const {
  std::vector<std::vector<uint64_t>> Weight;
  std::unordered_map<BinaryBasicBlock *, int> BBToIndex;
  std::vector<BinaryBasicBlock *> IndexToBB;

  unsigned N = BF.layout_size();
  // Populating weight map and index map
  for (auto BB : BF.layout()) {
    BBToIndex[BB] = IndexToBB.size();
    IndexToBB.push_back(BB);
  }
  Weight.resize(N);
  for (auto BB : BF.layout()) {
    auto BI = BB->branch_info_begin();
    Weight[BBToIndex[BB]].resize(N);
    for (auto I : BB->successors()) {
      if (BI->Count != BinaryBasicBlock::COUNT_FALLTHROUGH_EDGE)
        Weight[BBToIndex[BB]][BBToIndex[I]] = BI->Count;
      ++BI;
    }
  }

  std::vector<std::vector<int64_t>> DP;
  DP.resize(1 << N);
  for (auto &Elmt : DP) {
    Elmt.resize(N, -1);
  }
  // Start with the entry basic block being allocated with cost zero
  DP[1][0] = 0;
  // Walk through TSP solutions using a bitmask to represent state (current set
  // of BBs in the layout)
  unsigned BestSet = 1;
  unsigned BestLast = 0;
  int64_t BestWeight = 0;
  for (unsigned Set = 1; Set < (1U << N); ++Set) {
    // Traverse each possibility of Last BB visited in this layout
    for (unsigned Last = 0; Last < N; ++Last) {
      // Case 1: There is no possible layout with this BB as Last
      if (DP[Set][Last] == -1)
        continue;

      // Case 2: There is a layout with this Set and this Last, and we try
      // to expand this set with New
      for (unsigned New = 1; New < N; ++New) {
        // Case 2a: BB "New" is already in this Set
        if ((Set & (1 << New)) != 0)
          continue;

        // Case 2b: BB "New" is not in this set and we add it to this Set and
        // record total weight of this layout with "New" as the last BB.
        unsigned NewSet = (Set | (1 << New));
        if (DP[NewSet][New] == -1)
          DP[NewSet][New] = DP[Set][Last] + (int64_t)Weight[Last][New];
        DP[NewSet][New] = std::max(DP[NewSet][New],
                                   DP[Set][Last] + (int64_t)Weight[Last][New]);

        if (DP[NewSet][New] > BestWeight) {
          BestWeight = DP[NewSet][New];
          BestSet = NewSet;
          BestLast = New;
        }
      }
    }
  }

  // Define final function layout based on layout that maximizes weight
  unsigned Last = BestLast;
  unsigned Set = BestSet;
  std::vector<bool> Visited;
  Visited.resize(N);
  Visited[Last] = true;
  Order.push_back(IndexToBB[Last]);
  Set = Set & ~(1U << Last);
  while (Set != 0) {
    int64_t Best = -1;
    for (unsigned I = 0; I < N; ++I) {
      if (DP[Set][I] == -1)
        continue;
      if (DP[Set][I] > Best) {
        Last = I;
        Best = DP[Set][I];
      }
    }
    Visited[Last] = true;
    Order.push_back(IndexToBB[Last]);
    Set = Set & ~(1U << Last);
  }
  std::reverse(Order.begin(), Order.end());

  // Finalize layout with BBs that weren't assigned to the layout
  for (auto BB : BF.layout()) {
    if (Visited[BBToIndex[BB]] == false)
      Order.push_back(BB);
  }
}

void OptimizeReorderAlgorithm::reorderBasicBlocks(
      const BinaryFunction &BF, BasicBlockOrder &Order) const {
  if (BF.layout_empty())
    return;

  // Cluster basic blocks.
  CAlgo->clusterBasicBlocks(BF);

  if (opts::PrintClusters)
    CAlgo->printClusters();

  // Arrange basic blocks according to clusters.
  for (ClusterAlgorithm::ClusterTy &Cluster : CAlgo->Clusters)
    Order.insert(Order.end(),  Cluster.begin(), Cluster.end());
}

void OptimizeBranchReorderAlgorithm::reorderBasicBlocks(
      const BinaryFunction &BF, BasicBlockOrder &Order) const {
  if (BF.layout_empty())
    return;

  // Cluster basic blocks.
  CAlgo->clusterBasicBlocks(BF);
  std::vector<ClusterAlgorithm::ClusterTy> &Clusters = CAlgo->Clusters;;
  auto &ClusterEdges = CAlgo->ClusterEdges;

  // Compute clusters' average frequencies.
  CAlgo->computeClusterAverageFrequency();
  std::vector<double> &AvgFreq = CAlgo->AvgFreq;;

  if (opts::PrintClusters)
    CAlgo->printClusters();

  // Cluster layout order
  std::vector<uint32_t> ClusterOrder;

  // Do a topological sort for clusters, prioritizing frequently-executed BBs
  // during the traversal.
  std::stack<uint32_t> Stack;
  std::vector<uint32_t> Status;
  std::vector<uint32_t> Parent;
  Status.resize(Clusters.size(), 0);
  Parent.resize(Clusters.size(), 0);
  constexpr uint32_t STACKED = 1;
  constexpr uint32_t VISITED = 2;
  Status[0] = STACKED;
  Stack.push(0);
  while (!Stack.empty()) {
    uint32_t I = Stack.top();
    if (!(Status[I] & VISITED)) {
      Status[I] |= VISITED;
      // Order successors by weight
      auto ClusterComp = [&ClusterEdges, I](uint32_t A, uint32_t B) {
        return ClusterEdges[I][A] > ClusterEdges[I][B];
      };
      std::priority_queue<uint32_t, std::vector<uint32_t>,
                          decltype(ClusterComp)> SuccQueue(ClusterComp);
      for (auto &Target: ClusterEdges[I]) {
        if (Target.second > 0 && !(Status[Target.first] & STACKED) &&
            !Clusters[Target.first].empty()) {
          Parent[Target.first] = I;
          Status[Target.first] = STACKED;
          SuccQueue.push(Target.first);
        }
      }
      while (!SuccQueue.empty()) {
        Stack.push(SuccQueue.top());
        SuccQueue.pop();
      }
      continue;
    }
    // Already visited this node
    Stack.pop();
    ClusterOrder.push_back(I);
  }
  std::reverse(ClusterOrder.begin(), ClusterOrder.end());
  // Put unreachable clusters at the end
  for (uint32_t I = 0, E = Clusters.size(); I < E; ++I)
    if (!(Status[I] & VISITED) && !Clusters[I].empty())
      ClusterOrder.push_back(I);

  // Sort nodes with equal precedence
  auto Beg = ClusterOrder.begin();
  // Don't reorder the first cluster, which contains the function entry point
  ++Beg;
  std::stable_sort(Beg, ClusterOrder.end(),
                   [&AvgFreq, &Parent](uint32_t A, uint32_t B) {
                     uint32_t P = Parent[A];
                     while (Parent[P] != 0) {
                       if (Parent[P] == B)
                         return false;
                       P = Parent[P];
                     }
                     P = Parent[B];
                     while (Parent[P] != 0) {
                       if (Parent[P] == A)
                         return true;
                       P = Parent[P];
                     }
                     return AvgFreq[A] > AvgFreq[B];
                   });

  if (opts::PrintClusters) {
    errs() << "New cluster order: ";
    auto Sep = "";
    for (auto O : ClusterOrder) {
      errs() << Sep << O;
      Sep = ", ";
    }
    errs() << '\n';
  }

  // Arrange basic blocks according to cluster order.
  for (uint32_t ClusterIndex : ClusterOrder) {
    ClusterAlgorithm::ClusterTy &Cluster = Clusters[ClusterIndex];
    Order.insert(Order.end(),  Cluster.begin(), Cluster.end());
  }
}

void OptimizeCacheReorderAlgorithm::reorderBasicBlocks(
      const BinaryFunction &BF, BasicBlockOrder &Order) const {
  if (BF.layout_empty())
    return;

  // Cluster basic blocks.
  CAlgo->clusterBasicBlocks(BF);
  std::vector<ClusterAlgorithm::ClusterTy> &Clusters = CAlgo->Clusters;;

  // Compute clusters' average frequencies.
  CAlgo->computeClusterAverageFrequency();
  std::vector<double> &AvgFreq = CAlgo->AvgFreq;;

  if (opts::PrintClusters)
    CAlgo->printClusters();

  // Cluster layout order
  std::vector<uint32_t> ClusterOrder;

  // Order clusters based on average instruction execution frequency
  for (uint32_t I = 0, E = Clusters.size(); I < E; ++I)
    if (!Clusters[I].empty())
      ClusterOrder.push_back(I);
  auto Beg = ClusterOrder.begin();
  // Don't reorder the first cluster, which contains the function entry point
  ++Beg;
  std::stable_sort(Beg, ClusterOrder.end(), [&AvgFreq](uint32_t A, uint32_t B) {
    return AvgFreq[A] > AvgFreq[B];
  });

  if (opts::PrintClusters) {
    errs() << "New cluster order: ";
    auto Sep = "";
    for (auto O : ClusterOrder) {
      errs() << Sep << O;
      Sep = ", ";
    }
    errs() << '\n';
  }

  // Arrange basic blocks according to cluster order.
  for (uint32_t ClusterIndex : ClusterOrder) {
    ClusterAlgorithm::ClusterTy &Cluster = Clusters[ClusterIndex];
    Order.insert(Order.end(),  Cluster.begin(), Cluster.end());
  }
}

void ReverseReorderAlgorithm::reorderBasicBlocks(
      const BinaryFunction &BF, BasicBlockOrder &Order) const {
  if (BF.layout_empty())
    return;

  auto FirstBB = *BF.layout_begin();
  Order.push_back(FirstBB);
  for (auto RLI = BF.layout_rbegin(); *RLI != FirstBB; ++RLI)
    Order.push_back(*RLI);
}


