//===--- ExtTSPReorderAlgorithm.cpp - Order basic blocks ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// ExtTSP - layout of basic blocks with i-cache optimization.
//
// The algorithm is a greedy heuristic that works with chains (ordered lists)
// of basic blocks. Initially all chains are isolated basic blocks. On every
// iteration, we pick a pair of chains whose merging yields the biggest increase
// in the ExtTSP value, which models how i-cache "friendly" a specific chain is.
// A pair of chains giving the maximum gain is merged into a new chain. The
// procedure stops when there is only one chain left, or when merging does not
// increase ExtTSP. In the latter case, the remaining chains are sorted by
// density in decreasing order.
//
// An important aspect is the way two chains are merged. Unlike earlier
// algorithms (e.g., OptimizeCacheReorderAlgorithm or Pettis-Hansen), two
// chains, X and Y, are first split into three, X1, X2, and Y. Then we
// consider all possible ways of gluing the three chains (e.g., X1YX2, X1X2Y,
// X2X1Y, X2YX1, YX1X2, YX2X1) and choose the one producing the largest score.
// This improves the quality of the final result (the search space is larger)
// while keeping the implementation sufficiently fast.
//
// Reference:
//   * A. Newell and S. Pupyrev, Improved Basic Block Reordering, available
//         at https://arxiv.org/abs/1809.04676
//===----------------------------------------------------------------------===//
#include "BinaryBasicBlock.h"
#include "BinaryFunction.h"
#include "CacheMetrics.h"
#include "ReorderAlgorithm.h"
#include "llvm/Support/Options.h"

using namespace llvm;
using namespace bolt;
namespace opts {

extern cl::OptionCategory BoltOptCategory;
extern cl::opt<bool> NoThreads;

cl::opt<unsigned>
ChainSplitThreshold("chain-split-threshold",
  cl::desc("The maximum size of a chain to apply splitting"),
  cl::init(128),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<double>
FallthroughWeight("fallthrough-weight",
  cl::desc("The weight of forward jumps for ExtTSP value"),
  cl::init(1),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<double>
ForwardWeight("forward-weight",
  cl::desc("The weight of forward jumps for ExtTSP value"),
  cl::init(0.1),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<double>
BackwardWeight("backward-weight",
  cl::desc("The weight of backward jumps for ExtTSP value"),
  cl::init(0.1),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
ForwardDistance("forward-distance",
  cl::desc("The maximum distance (in bytes) of forward jumps for ExtTSP value"),
  cl::init(1024),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<unsigned>
BackwardDistance("backward-distance",
  cl::desc("The maximum distance (in bytes) of backward jumps for ExtTSP value"),
  cl::init(640),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

}

namespace llvm {
namespace bolt {

namespace {

class Block;
class Chain;
class Edge;

const double EPS = 1e-8;

// Calculate Ext-TSP value, which quantifies the expected number of i-cache
// misses for a given ordering of basic blocks
double extTSPScore(uint64_t SrcAddr,
                   uint64_t SrcSize,
                   uint64_t DstAddr,
                   uint64_t Count) {
  assert(Count != BinaryBasicBlock::COUNT_NO_PROFILE);

  // Fallthrough
  if (SrcAddr + SrcSize == DstAddr) {
    return opts::FallthroughWeight * Count;
  }
  // Forward
  if (SrcAddr + SrcSize < DstAddr) {
    const auto Dist = DstAddr - (SrcAddr + SrcSize);
    if (Dist <= opts::ForwardDistance) {
      double Prob = 1.0 - static_cast<double>(Dist) / opts::ForwardDistance;
      return opts::ForwardWeight * Prob * Count;
    }
    return 0;
  }
  // Backward
  const auto Dist = SrcAddr + SrcSize - DstAddr;
  if (Dist <= opts::BackwardDistance) {
    double Prob = 1.0 - static_cast<double>(Dist) / opts::BackwardDistance;
    return opts::BackwardWeight * Prob * Count;
  }
  return 0;
}

using BlockPair = std::pair<Block *, Block *>;
using JumpList = std::vector<std::pair<BlockPair, uint64_t>>;
using MergeGainTy = std::pair<double, size_t>;
using BlockIter = std::vector<Block *>::const_iterator;

// A node in CFG corresponding to a BinaryBasicBlock.
// The class wraps several mutable fields utilized in the ExtTSP algorithm
class Block {
public:
  // Delete copy constructor to make sure objects are moved rather than copied
  Block() {}
  Block(const Block&) = delete;
  Block(Block&&) = default;
  Block& operator=(const Block&) = delete;
  Block& operator=(Block&&) = default;

  // Corresponding basic block
  BinaryBasicBlock *BB{nullptr};
  // Current chain of the basic block
  Chain *CurChain{nullptr};
  // (Estimated) size of the block in the binary
  uint64_t Size{0};
  // Execution count of the block in the binary
  uint64_t ExecutionCount{0};
  // An original index of the node in CFG
  size_t Index{0};
  // An offset of the block in the current chain
  mutable uint64_t EstimatedAddr{0};
  // Fallthrough successor of the node in CFG
  Block *FallthroughSucc{nullptr};
  // Fallthrough predecessor of the node in CFG
  Block *FallthroughPred{nullptr};
  // Outgoing jumps from the block
  std::vector<std::pair<Block *, uint64_t>> OutJumps;
  // Total execution count of incoming jumps
  uint64_t InWeight{0};
  // Total execution count of outgoing jumps
  uint64_t OutWeight{0};

  explicit Block(BinaryBasicBlock *BB_, uint64_t Size_)
    : BB(BB_),
      Size(Size_),
      ExecutionCount(BB_->getKnownExecutionCount()),
      Index(BB->getLayoutIndex()) {}
};

// A chain (ordered sequence) of CFG nodes (basic blocks)
class Chain {
public:
  Chain() {}
  Chain(const Chain&) = delete;
  Chain(Chain&&) = default;
  Chain& operator=(const Chain&) = delete;
  Chain& operator=(Chain&&) = default;

  explicit Chain(size_t Id_, Block *Block)
    : Id(Id_),
      IsEntry(Block->Index == 0),
      ExecutionCount(Block->ExecutionCount),
      Size(Block->Size),
      Score(0),
      Blocks(1, Block) {}

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

  void setScore(double NewScore) {
    Score = NewScore;
  }

  const std::vector<Block *> &blocks() const {
    return Blocks;
  }

  const std::unordered_map<Chain *, Edge *> &edges() const {
    return Edges;
  }

  Edge *getEdge(Chain *Other) const {
    auto It = Edges.find(Other);
    return It != Edges.end() ? It->second : nullptr;
  }

  void addEdge(Chain *Other, Edge *Edge) {
    assert(Edges.find(Other) == Edges.end());
    Edges.insert(std::make_pair(Other, Edge));
  }

  /// Update the list of basic blocks and aggregated chain data
  void merge(Chain *Other, const std::vector<Block *> &MergedBlocks) {
    Blocks = MergedBlocks;
    IsEntry |= Other->IsEntry;
    ExecutionCount += Other->ExecutionCount;
    Size += Other->Size;
    // Update block's chains
    for (auto Block : Other->blocks()) {
      Block->CurChain = this;
    }
    // Merge edges
    mergeEdges(Other);
  }

  void mergeEdges(Chain *Other);

  void clear() {
    Blocks.clear();
    Edges.clear();
  }

private:
  size_t Id;
  bool IsEntry;
  uint64_t ExecutionCount;
  uint64_t Size;
  // Cached ext-tsp score for the chain
  double Score;
  // Blocks of the chain
  std::vector<Block *> Blocks;
  // Adjacent chains and corresponding edges (lists of jumps)
  std::unordered_map<Chain *, Edge *> Edges;
};

// An edge in CFG reprsenting jumps between chains of BinaryBasicBlocks.
// When blocks are merged into chains, the edges are combined too so that
// there is always at most one edge between a pair of chains
class Edge {
public:
  Edge() {}
  Edge(const Edge&) = delete;
  Edge(Edge&&) = default;
  Edge& operator=(const Edge&) = delete;
  Edge& operator=(Edge&&) = default;

  explicit Edge(Block *SrcBlock, Block *DstBlock, uint64_t EC)
    : SrcChain(SrcBlock->CurChain),
      DstChain(DstBlock->CurChain),
      Jumps(1, std::make_pair(std::make_pair(SrcBlock, DstBlock), EC)) {}

  const JumpList &jumps() const {
    return Jumps;
  }

  void changeEndpoint(Chain *From, Chain *To) {
    assert(From == SrcChain || From == DstChain);
    if (From == SrcChain)
      SrcChain = To;
    if (From == DstChain)
      DstChain = To;
  }

  void appendJump(Block *SrcBlock, Block *DstBlock, uint64_t EC) {
    Jumps.push_back(std::make_pair(std::make_pair(SrcBlock, DstBlock), EC));
  }

  void moveJumps(Edge *Other) {
    Jumps.insert(Jumps.end(), Other->Jumps.begin(), Other->Jumps.end());
    Other->Jumps.clear();
  }

  bool hasCachedMergeGain(Chain *Src, Chain *Dst) const {
    assert(Src == SrcChain || Src == DstChain);
    return Src == SrcChain ? CacheIsValidF : CacheIsValidB;
  }

  MergeGainTy getCachedMergeGain(Chain *Src, Chain *Dst) const {
    assert(Src == SrcChain || Src == DstChain);
    return Src == SrcChain ? CachedMergeGainF : CachedMergeGainB;
  }

  void setCachedMergeGain(Chain *Src, Chain *Dst, MergeGainTy MergeGain) {
    assert(Src == SrcChain || Src == DstChain);
    if (Src == SrcChain) {
      CachedMergeGainF = MergeGain;
      CacheIsValidF = true;
    } else {
      CachedMergeGainB = MergeGain;
      CacheIsValidB = true;
    }
  }

  void invalidateCache() {
    CacheIsValidF = false;
    CacheIsValidB = false;
  }

private:
  Chain *SrcChain{nullptr};
  Chain *DstChain{nullptr};
  // Original jumps in the binary with correspinding execution counts
  JumpList Jumps;
  // Cached ext-tsp value for merging the pair of chains
  // Since the gain of merging (Src, Dst) and (Dst, Src) might be different,
  // we store both values here
  MergeGainTy CachedMergeGainF;
  MergeGainTy CachedMergeGainB;
  // Whether the cached value must be recomputed
  bool CacheIsValidF{false};
  bool CacheIsValidB{false};
};

void Chain::mergeEdges(Chain *Other) {
  assert(this != Other && "cannot merge a cluster with itself");

  // update edges adjacent to Other
  for (auto EdgeIter : Other->edges()) {
    const auto DstChain = EdgeIter.first;
    auto DstEdge = EdgeIter.second;

    if (DstChain == Other) {
      // processing self-edge (Other, Other)
      auto It = Edges.find(this);
      if (It == Edges.end()) {
        DstEdge->changeEndpoint(Other, this);
        Edges.insert(std::make_pair(this, DstEdge));
      } else {
        auto CurEdge = It->second;
        CurEdge->moveJumps(DstEdge);
      }
    } else {
      // processing edge (Other, DstChain)
      auto It = Edges.find(DstChain);
      if (It == Edges.end()) {
        DstEdge->changeEndpoint(Other, this);
        Edges.insert(std::make_pair(DstChain, DstEdge));
        DstChain->Edges.insert(std::make_pair(this, DstEdge));
      } else {
        auto CurEdge = It->second;
        CurEdge->moveJumps(DstEdge);
      }
      // cleanup leftover edge
      DstChain->Edges.erase(Other);
    }
  }
}

// A wrapper around three chains of basic blocks; it is used to avoid extra
// instantiation of the vectors.
class MergedChain {
public:
  MergedChain(BlockIter Begin1,
              BlockIter End1,
              BlockIter Begin2 = BlockIter(),
              BlockIter End2 = BlockIter(),
              BlockIter Begin3 = BlockIter(),
              BlockIter End3 = BlockIter())
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

  std::vector<Block *> getBlocks() const {
    std::vector<Block *> Result;
    Result.reserve(std::distance(Begin1, End1) +
                   std::distance(Begin2, End2) +
                   std::distance(Begin3, End3));
    Result.insert(Result.end(), Begin1, End1);
    Result.insert(Result.end(), Begin2, End2);
    Result.insert(Result.end(), Begin3, End3);
    return Result;
  }

  const Block *getFirstBlock() const {
    return *Begin1;
  }

private:
  BlockIter Begin1;
  BlockIter End1;
  BlockIter Begin2;
  BlockIter End2;
  BlockIter Begin3;
  BlockIter End3;
};

/// Deterministically compare pairs of chains
bool compareChainPairs(const Chain *A1, const Chain *B1,
                       const Chain *A2, const Chain *B2) {
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

class ExtTSP {
public:
  ExtTSP(const BinaryFunction &BF) : BF(BF) {
    initialize();
  }

  /// Run the algorithm and return an ordering of basic block
  void run(std::vector<BinaryBasicBlock *> &Order) {
    // Pass 1: Merge blocks with their fallthrough successors
    mergeFallthroughs();

    // Pass 2: Merge pairs of chains while improving the ExtTSP metric
    mergeChainPairs();

    // Pass 3: Merge cold blocks to reduce code size
    mergeColdChains();

    // Collect blocks from all chains
    concatChains(Order);
  }

private:
  /// Initialize algorithm's data structures
  void initialize() {
    // Create a separate MCCodeEmitter to allow lock-free execution
    BinaryContext::IndependentCodeEmitter Emitter;
    if (!opts::NoThreads) {
      Emitter = BF.getBinaryContext().createIndependentMCCodeEmitter();
    }

    // Initialize CFG nodes
    AllBlocks.reserve(BF.layout_size());
    size_t LayoutIndex = 0;
    for (auto BB : BF.layout()) {
      BB->setLayoutIndex(LayoutIndex++);
      auto Size = std::max<uint64_t>(BB->estimateSize(Emitter.MCE.get()), 1);
      AllBlocks.emplace_back(BB, Size);
    }

    // Initialize edges for the blocks and compute their total in/out weights
    size_t NumEdges = 0;
    for (auto &Block : AllBlocks) {
      auto BI = Block.BB->branch_info_begin();
      for (auto SuccBB : Block.BB->successors()) {
        assert(BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
               "missing profile for a jump");
        if (SuccBB != Block.BB && BI->Count > 0) {
          auto &SuccBlock = AllBlocks[SuccBB->getLayoutIndex()];
          SuccBlock.InWeight += BI->Count;
          Block.OutWeight += BI->Count;
          Block.OutJumps.push_back(std::make_pair(&SuccBlock, BI->Count));
          NumEdges++;
        }
        ++BI;
      }
    }

    // Initialize execution count for every basic block, which is the
    // maximum over the sums of all in and out edge weights.
    // Also execution count of the entry point is set to at least 1
    for (auto &Block : AllBlocks) {
      size_t Index = Block.Index;
      Block.ExecutionCount = std::max(Block.ExecutionCount, Block.InWeight);
      Block.ExecutionCount = std::max(Block.ExecutionCount, Block.OutWeight);
      if (Index == 0 && Block.ExecutionCount == 0)
        Block.ExecutionCount = 1;
    }

    // Initialize chains
    AllChains.reserve(BF.layout_size());
    HotChains.reserve(BF.layout_size());
    for (auto &Block : AllBlocks) {
      AllChains.emplace_back(Block.Index, &Block);
      Block.CurChain = &AllChains.back();
      if (Block.ExecutionCount > 0) {
        HotChains.push_back(&AllChains.back());
      }
    }

    // Initialize edges
    AllEdges.reserve(NumEdges);
    for (auto &Block : AllBlocks) {
      for (auto &Jump : Block.OutJumps) {
        const auto SuccBlock = Jump.first;
        assert(Block.CurChain != SuccBlock->CurChain);
        auto CurEdge = Block.CurChain->getEdge(SuccBlock->CurChain);
        // this edge is already present in the graph
        if (CurEdge != nullptr) {
          assert(SuccBlock->CurChain->getEdge(Block.CurChain) != nullptr);
          CurEdge->appendJump(&Block, SuccBlock, Jump.second);
          continue;
        }
        // this is a new edge
        AllEdges.emplace_back(&Block, SuccBlock, Jump.second);
        Block.CurChain->addEdge(SuccBlock->CurChain, &AllEdges.back());
        SuccBlock->CurChain->addEdge(Block.CurChain, &AllEdges.back());
      }
    }
    assert(AllEdges.size() <= NumEdges && "Incorrect number of created edges");

    // Initialize fallthrough successors
    findFallthroughBlocks();
  }

  /// For a pair of blocks, A and B, block B is the fallthrough successor of A,
  /// if (i) all jumps (based on profile) from A goes to B and (ii) all jumps
  /// to B are from A. Such blocks should be adjacent in an optimal ordering,
  /// and the method finds such pairs of blocks
  void findFallthroughBlocks() {
    // Find fallthroughs based on edge weights
    for (auto &Block : AllBlocks) {
      if (Block.BB->succ_size() == 1 &&
          Block.BB->getSuccessor()->pred_size() == 1 &&
          Block.BB->getSuccessor()->getLayoutIndex() != 0) {
        size_t SuccIndex = Block.BB->getSuccessor()->getLayoutIndex();
        Block.FallthroughSucc = &AllBlocks[SuccIndex];
        AllBlocks[SuccIndex].FallthroughPred = &Block;
        continue;
      }

      if (Block.OutWeight == 0)
        continue;
      for (auto &Edge : Block.OutJumps) {
        const auto SuccBlock = Edge.first;
        // Successor cannot be the first BB, which is pinned
        if (Block.OutWeight == Edge.second &&
            SuccBlock->InWeight == Edge.second &&
            SuccBlock->Index != 0) {
          Block.FallthroughSucc = SuccBlock;
          SuccBlock->FallthroughPred = &Block;
          break;
        }
      }
    }

    // There might be 'cycles' in the fallthrough dependencies (since profile
    // data isn't 100% accurate).
    // Break the cycles by choosing the block with smallest index as the tail
    for (auto &Block : AllBlocks) {
      if (Block.FallthroughSucc == nullptr || Block.FallthroughPred == nullptr)
        continue;

      auto SuccBlock = Block.FallthroughSucc;
      while (SuccBlock != nullptr && SuccBlock != &Block) {
        SuccBlock = SuccBlock->FallthroughSucc;
      }
      if (SuccBlock == nullptr)
        continue;
      // break the cycle
      AllBlocks[Block.FallthroughPred->Index].FallthroughSucc = nullptr;
      Block.FallthroughPred = nullptr;
    }
  }

  /// Merge blocks with their fallthrough successors
  void mergeFallthroughs() {
    for (auto &Block : AllBlocks) {
      if (Block.FallthroughPred == nullptr &&
          Block.FallthroughSucc != nullptr) {
        auto CurBlock = &Block;
        while (CurBlock->FallthroughSucc != nullptr) {
          const auto NextBlock = CurBlock->FallthroughSucc;
          mergeChains(Block.CurChain, NextBlock->CurChain);
          CurBlock = NextBlock;
        }
      }
    }
  }

  /// Merge pairs of chains while improving the ExtTSP metric
  void mergeChainPairs() {
    while (HotChains.size() > 1) {
      Chain *BestChainPred = nullptr;
      Chain *BestChainSucc = nullptr;
      std::pair<double, size_t> BestGain(-1.0, 0);
      // Iterate over all pairs of chains
      for (auto ChainPred : HotChains) {
        // Get candidates for merging with the current chain
        for (auto EdgeIter : ChainPred->edges()) {
          auto ChainSucc = EdgeIter.first;
          // Ignore loop edges
          if (ChainPred == ChainSucc)
            continue;

          // Compute the gain of merging the two chains
          auto Gain = mergeGain(ChainPred, ChainSucc, EdgeIter.second);
          if (Gain.first <= 0.0)
            continue;

          // Breaking ties by density to make the hottest chains be merged first
          if (Gain.first > BestGain.first + EPS ||
              (std::abs(Gain.first - BestGain.first) < EPS &&
               compareChainPairs(ChainPred,
                                 ChainSucc,
                                 BestChainPred,
                                 BestChainSucc))) {
            BestGain = Gain;
            BestChainPred = ChainPred;
            BestChainSucc = ChainSucc;
          }
        }
      }

      // Stop merging when there is no improvement
      if (BestGain.first <= 0.0)
        break;

      // Merge the best pair of chains
      mergeChains(BestChainPred, BestChainSucc, BestGain.second);
    }
  }

  /// Merge cold blocks to reduce code size
  void mergeColdChains() {
    for (auto SrcBB : BF.layout()) {
      // Iterating in reverse order to make sure original fall-trough jumps are
      // merged first
      for (auto Itr = SrcBB->succ_rbegin(); Itr != SrcBB->succ_rend(); ++Itr) {
        BinaryBasicBlock *DstBB = *Itr;
        size_t SrcIndex = SrcBB->getLayoutIndex();
        size_t DstIndex = DstBB->getLayoutIndex();
        auto SrcChain = AllBlocks[SrcIndex].CurChain;
        auto DstChain = AllBlocks[DstIndex].CurChain;
        if (SrcChain != DstChain && !DstChain->isEntryPoint() &&
            SrcChain->blocks().back()->Index == SrcIndex &&
            DstChain->blocks().front()->Index == DstIndex) {
          mergeChains(SrcChain, DstChain);
        }
      }
    }
  }

  /// Compute ExtTSP score for a given order of basic blocks
  double score(const MergedChain &MergedBlocks, const JumpList &Jumps) const {
    if (Jumps.empty())
      return 0.0;
    uint64_t CurAddr = 0;
    MergedBlocks.forEach(
      [&](const Block *BB) {
        BB->EstimatedAddr = CurAddr;
        CurAddr += BB->Size;
      }
    );

    double Score = 0;
    for (auto &Jump : Jumps) {
      const auto SrcBlock = Jump.first.first;
      const auto DstBlock = Jump.first.second;
      Score += extTSPScore(SrcBlock->EstimatedAddr,
                           SrcBlock->Size,
                           DstBlock->EstimatedAddr,
                           Jump.second);
    }
    return Score;
  }

  /// Verify if it is valid to merge two chains into the new one
  bool isValidMerge(const Chain *ChainPred,
                    const Chain *ChainSucc,
                    size_t MergeType,
                    const MergedChain& MergedBlocks) const {
    // Does the new chain preserve the original entry point?
    if ((ChainPred->isEntryPoint() || ChainSucc->isEntryPoint()) &&
        MergedBlocks.getFirstBlock()->Index != 0)
      return false;

    // This corresponds to a concatentation of chains w/o splitting, which is
    // always safe
    if (MergeType == 0)
      return true;

    size_t Offset = MergeType / 5;
    // The basic blocks on the boundary of a split of ChainPred
    auto BB1 = ChainPred->blocks()[Offset - 1];
    auto BB2 = ChainPred->blocks()[Offset];
    // Does the splitting break FT successors?
    if (BB1->FallthroughSucc != nullptr) {
      assert(BB1->FallthroughSucc == BB2 &&
             "Fallthrough successor is not preserved");
      return false;
    }

    return true;
  }

  /// The gain of merging two chains
  ///
  /// The function considers all possible ways of merging two chains and
  /// computes the one having the largest increase in ExtTSP metric. The result
  /// is a pair with the first element being the gain and the second element being
  /// the corresponding merging type (encoded as an integer).
  MergeGainTy mergeGain(Chain *ChainPred, Chain *ChainSucc, Edge *Edge) const {
    if (Edge->hasCachedMergeGain(ChainPred, ChainSucc)) {
      return Edge->getCachedMergeGain(ChainPred, ChainSucc);
    }

    // Precompute jumps between ChainPred and ChainSucc
    auto Jumps = Edge->jumps();
    auto EdgePP = ChainPred->getEdge(ChainPred);
    if (EdgePP != nullptr)
      Jumps.insert(Jumps.end(), EdgePP->jumps().begin(), EdgePP->jumps().end());
    assert(Jumps.size() > 0 && "trying to merge chains w/o jumps");

    // Merge two chains and update the best Gain
    auto computeMergeGain = [&](const MergeGainTy &CurGain,
                                const Chain *ChainPred,
                                const Chain *ChainSucc,
                                size_t MergeType) {
      auto MergedBlocks = mergeBlocks(ChainPred->blocks(),
                                      ChainSucc->blocks(),
                                      MergeType);

      if (!isValidMerge(ChainPred, ChainSucc, MergeType, MergedBlocks))
        return CurGain;

      // The gain for the new chain
      const auto NewGain = score(MergedBlocks, Jumps) - ChainPred->score();
      if (NewGain > EPS && NewGain > CurGain.first + EPS)
        return std::make_pair(NewGain, MergeType);
      else
        return CurGain;
    };

    MergeGainTy Gain = std::make_pair(-1.0, 0);
    // Try to concatenate two chains w/o splitting
    Gain = computeMergeGain(Gain, ChainPred, ChainSucc, 0);

    // Do not split large chains to reduce computation time
    if (ChainPred->blocks().size() <= opts::ChainSplitThreshold) {
      // Try to split ChainPred into two sub-chains in various ways and then
      // merge it with ChainSucc
      for (size_t Offset = 1; Offset < ChainPred->blocks().size(); Offset++) {
        for (size_t Type = 1; Type <= 4; Type++) {
          size_t MergeType = Type + Offset * 5;
          Gain = computeMergeGain(Gain, ChainPred, ChainSucc, MergeType);
        }
      }
    }

    Edge->setCachedMergeGain(ChainPred, ChainSucc, Gain);
    return Gain;
  }

  /// Merge two chains of blocks respecting a given merge 'type' and 'offset'
  ///
  /// If MergeType == 0, then the result is a concatentation of two chains.
  /// Otherwise, the first chain is cut into two sub-chains at the offset,
  /// and merged using all possible ways of concatenating three chains.
  MergedChain mergeBlocks(const std::vector<Block *> &X,
                          const std::vector<Block *> &Y,
                          size_t MergeType) const {
    // Merging w/o splitting existing chains
    if (MergeType == 0)
      return MergedChain(X.begin(), X.end(), Y.begin(), Y.end());

    size_t Type = MergeType % 5;
    size_t Offset = MergeType / 5;
    assert(0 < Offset && Offset < X.size() &&
           "Invalid offset while merging chains");
    // Split the first chain, X, into X1 and X2
    BlockIter BeginX1 = X.begin();
    BlockIter EndX1 = X.begin() + Offset;
    BlockIter BeginX2 = X.begin() + Offset;
    BlockIter EndX2 = X.end();
    BlockIter BeginY = Y.begin();
    BlockIter EndY = Y.end();

    // Construct a new chain from three existing ones
    switch(Type) {
    case 1: return MergedChain(BeginX1, EndX1, BeginY, EndY, BeginX2, EndX2);
    case 2: return MergedChain(BeginY, EndY, BeginX2, EndX2, BeginX1, EndX1);
    case 3: return MergedChain(BeginX2, EndX2, BeginY, EndY, BeginX1, EndX1);
    case 4: return MergedChain(BeginX2, EndX2, BeginX1, EndX1, BeginY, EndY);
    default:
      llvm_unreachable("unexpected merge type");
    }
  }

  /// Merge chain From into chain Into, update the list of active chains,
  /// adjacency information, and the corresponding cached values
  void mergeChains(Chain *Into, Chain *From, size_t MergeType = 0) {
    assert(Into != From && "chain cannot be merged with itself");

    // Merge the blocks
    auto MergedBlocks = mergeBlocks(Into->blocks(), From->blocks(), MergeType);
    Into->merge(From, MergedBlocks.getBlocks());
    From->clear();

    // Update cached ext-tsp score for the new chain
    auto SelfEdge = Into->getEdge(Into);
    if (SelfEdge != nullptr) {
      MergedBlocks = MergedChain(Into->blocks().begin(), Into->blocks().end());
      Into->setScore(score(MergedBlocks, SelfEdge->jumps()));
    }

    // Remove chain From from the list of active chains
    auto Iter = std::remove(HotChains.begin(), HotChains.end(), From);
    HotChains.erase(Iter, HotChains.end());

    // Invalidate caches
    for (auto EdgeIter : Into->edges()) {
      EdgeIter.second->invalidateCache();
    }
  }

  /// Concatenate all chains into a final order
  void concatChains(std::vector<BinaryBasicBlock *> &Order) {
    // Collect chains
    std::vector<Chain *> SortedChains;
    for (auto &Chain : AllChains) {
      if (Chain.blocks().size() > 0) {
        SortedChains.push_back(&Chain);
      }
    }

    // Sorting chains by density in decreasing order
    std::stable_sort(
      SortedChains.begin(), SortedChains.end(),
      [](const Chain *C1, const Chain *C2) {
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
    );

    // Collect the basic blocks in the order specified by their chains
    Order.reserve(BF.layout_size());
    for (auto Chain : SortedChains) {
      for (auto Block : Chain->blocks()) {
        Order.push_back(Block->BB);
      }
    }
  }

private:
  // The binary function
  const BinaryFunction &BF;

  // All CFG nodes (basic blocks)
  std::vector<Block> AllBlocks;

  // All chains of blocks
  std::vector<Chain> AllChains;

  // Active chains. The vector gets udpated at runtime when chains are merged
  std::vector<Chain *> HotChains;

  // All edges between chains
  std::vector<Edge> AllEdges;
};

void ExtTSPReorderAlgorithm::reorderBasicBlocks(
      const BinaryFunction &BF, BasicBlockOrder &Order) const {
  if (BF.layout_empty())
    return;

  // Do not change layout of functions w/o profile information
  if (!BF.hasValidProfile() || BF.layout_size() <= 1) {
    for (auto BB : BF.layout()) {
      Order.push_back(BB);
    }
    return;
  }

  // Apply the algorithm
  ExtTSP(BF).run(Order);

  // Verify correctness
  assert(Order[0]->isEntryPoint() && "Original entry point is not preserved");
  assert(Order.size() == BF.layout_size() && "Wrong size of reordered layout");
}

} // namespace bolt
} // namespace llvm
