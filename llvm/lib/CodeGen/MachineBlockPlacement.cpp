//===-- MachineBlockPlacement.cpp - Basic Block Code Layout optimization --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements basic block placement transformations using branch
// probability estimates. It is based around "Algo2" from Profile Guided Code
// Positioning [http://portal.acm.org/citation.cfm?id=989433].
//
// We combine the BlockFrequencyInfo with BranchProbabilityInfo to simulate
// measured edge-weights. The BlockFrequencyInfo effectively summarizes the
// probability of starting from any particular block, and the
// BranchProbabilityInfo the probability of exiting the block via a particular
// edge. Combined they form a function-wide ordering of the edges.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "block-placement2"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Target/TargetInstrInfo.h"
#include <algorithm>
using namespace llvm;

namespace {
/// \brief A structure for storing a weighted edge.
///
/// This stores an edge and its weight, computed as the product of the
/// frequency that the starting block is entered with the probability of
/// a particular exit block.
struct WeightedEdge {
  BlockFrequency EdgeFrequency;
  MachineBasicBlock *From, *To;

  bool operator<(const WeightedEdge &RHS) const {
    return EdgeFrequency < RHS.EdgeFrequency;
  }
};
}

namespace {
struct BlockChain;
/// \brief Type for our function-wide basic block -> block chain mapping.
typedef DenseMap<MachineBasicBlock *, BlockChain *> BlockToChainMapType;
}

namespace {
/// \brief A chain of blocks which will be laid out contiguously.
///
/// This is the datastructure representing a chain of consecutive blocks that
/// are profitable to layout together in order to maximize fallthrough
/// probabilities. We also can use a block chain to represent a sequence of
/// basic blocks which have some external (correctness) requirement for
/// sequential layout.
///
/// Eventually, the block chains will form a directed graph over the function.
/// We provide an SCC-supporting-iterator in order to quicky build and walk the
/// SCCs of block chains within a function.
///
/// The block chains also have support for calculating and caching probability
/// information related to the chain itself versus other chains. This is used
/// for ranking during the final layout of block chains.
struct BlockChain {
  class SuccIterator;

  /// \brief The first and last basic block that from this chain.
  ///
  /// The chain is stored within the existing function ilist of basic blocks.
  /// When merging chains or otherwise manipulating them, we splice the blocks
  /// within this ilist, giving us very cheap storage here and constant time
  /// merge operations.
  ///
  /// It is extremely important to note that LastBB is the iterator pointing
  /// *at* the last basic block in the chain. That is, the chain consists of
  /// the *closed* range [FirstBB, LastBB]. We cannot use half-open ranges
  /// because the next basic block may get relocated to a different part of the
  /// function at any time during the run of this pass.
  MachineFunction::iterator FirstBB, LastBB;

  /// \brief A handle to the function-wide basic block to block chain mapping.
  ///
  /// This is retained in each block chain to simplify the computation of child
  /// block chains for SCC-formation and iteration. We store the edges to child
  /// basic blocks, and map them back to their associated chains using this
  /// structure.
  BlockToChainMapType &BlockToChain;

  /// \brief The weight used to rank two block chains in the same SCC.
  ///
  /// This is used during SCC layout of block chains to cache and rank the
  /// chains. It is supposed to represent the expected frequency with which
  /// control reaches a block within this chain, has the option of branching to
  /// a block in some other chain participating in the SCC, but instead
  /// continues within this chain. The higher this is, the more costly we
  /// expect mis-predicted branches between this chain and other chains within
  /// the SCC to be. Thus, since we expect branches between chains to be
  /// predicted when backwards and not predicted when forwards, the higher this
  /// is the more important that this chain is laid out first among those
  /// chains in the same SCC as it.
  BlockFrequency InChainEdgeFrequency;

  /// \brief Construct a new BlockChain.
  ///
  /// This builds a new block chain representing a single basic block in the
  /// function. It also registers itself as the chain that block participates
  /// in with the BlockToChain mapping.
  BlockChain(BlockToChainMapType &BlockToChain, MachineBasicBlock *BB)
    : FirstBB(BB), LastBB(BB), BlockToChain(BlockToChain) {
    assert(BB && "Cannot create a chain with a null basic block");
    BlockToChain[BB] = this;
  }

  /// \brief Merge another block chain into this one.
  ///
  /// This routine merges a block chain into this one. It takes care of forming
  /// a contiguous sequence of basic blocks, updating the edge list, and
  /// updating the block -> chain mapping. It does not free or tear down the
  /// old chain, but the old chain's block list is no longer valid.
  void merge(BlockChain *Chain) {
    assert(Chain && "Cannot merge a null chain");
    MachineFunction::iterator EndBB = llvm::next(LastBB);
    MachineFunction::iterator ChainEndBB = llvm::next(Chain->LastBB);

    // Update the incoming blocks to point to this chain.
    for (MachineFunction::iterator BI = Chain->FirstBB, BE = ChainEndBB;
         BI != BE; ++BI) {
      assert(BlockToChain[BI] == Chain && "Incoming blocks not in chain");
      BlockToChain[BI] = this;
    }

    // We splice the blocks together within the function (unless they already
    // are adjacent) so we can represent the new chain with a pair of pointers
    // to basic blocks within the function. This is also useful as each chain
    // of blocks will end up being laid out contiguously within the function.
    if (EndBB != Chain->FirstBB)
      FirstBB->getParent()->splice(EndBB, Chain->FirstBB, ChainEndBB);
    LastBB = Chain->LastBB;
  }
};
}

namespace {
/// \brief Successor iterator for BlockChains.
///
/// This is an iterator that walks over the successor block chains by looking
/// through its blocks successors and mapping those back to block chains. This
/// iterator is not a fully-functioning iterator, it is designed specifically
/// to support the interface required by SCCIterator when forming and walking
/// SCCs of BlockChains.
///
/// Note that this iterator cannot be used while the chains are still being
/// formed and/or merged. Unlike the chains themselves, it does store end
/// iterators which could be moved if the chains are re-ordered. Once we begin
/// forming and iterating over an SCC of chains, the order of blocks within the
/// function must not change until we finish using the SCC iterators.
class BlockChain::SuccIterator
    : public std::iterator<std::forward_iterator_tag,
                           BlockChain *, ptrdiff_t> {
  BlockChain *Chain;
  MachineFunction::iterator BI, BE;
  MachineBasicBlock::succ_iterator SI;

public:
  explicit SuccIterator(BlockChain *Chain)
    : Chain(Chain), BI(Chain->FirstBB), BE(llvm::next(Chain->LastBB)),
      SI(BI->succ_begin()) {
    while (BI != BE && BI->succ_begin() == BI->succ_end())
      ++BI;
    if (BI != BE)
      SI = BI->succ_begin();
  }

  /// \brief Helper function to create an end iterator for a particular chain.
  ///
  /// The "end" state is extremely arbitrary. We chose to have BI == BE, and SI
  /// == Chain->FirstBB->succ_begin(). The value of SI doesn't really make any
  /// sense, but rather than try to rationalize SI and our increment, when we
  /// detect an "end" state, we just immediately call this function to build
  /// the canonical end iterator.
  static SuccIterator CreateEnd(BlockChain *Chain) {
    SuccIterator It(Chain);
    It.BI = It.BE;
    return It;
  }

  bool operator==(const SuccIterator &RHS) const {
    return (Chain == RHS.Chain && BI == RHS.BI && SI == RHS.SI);
  }
  bool operator!=(const SuccIterator &RHS) const {
    return !operator==(RHS);
  }

  SuccIterator& operator++() {
    assert(*this != CreateEnd(Chain) && "Cannot increment the end iterator");
    // There may be null successor pointers, skip over them.
    // FIXME: I don't understand *why* there are null successor pointers.
    do {
      ++SI;
      if (SI != BI->succ_end() && *SI)
        return *this;

      // There may be a basic block without successors. Skip over them.
      do {
        ++BI;
        if (BI == BE)
          return *this = CreateEnd(Chain);
      } while (BI->succ_begin() == BI->succ_end());
      SI = BI->succ_begin();
    } while (!*SI);
    return *this;
  }
  SuccIterator operator++(int) {
    SuccIterator tmp = *this;
    ++*this;
    return tmp;
  }

  BlockChain *operator*() const {
    assert(Chain->BlockToChain.lookup(*SI) && "Missing chain");
    return Chain->BlockToChain.lookup(*SI);
  }
};
}

namespace {
/// \brief Sorter used with containers of BlockChain pointers.
///
/// Sorts based on the \see BlockChain::InChainEdgeFrequency -- see its
/// comments for details on what this ordering represents.
struct ChainPtrPrioritySorter {
  bool operator()(const BlockChain *LHS, const BlockChain *RHS) const {
    assert(LHS && RHS && "Null chain entry");
    return LHS->InChainEdgeFrequency < RHS->InChainEdgeFrequency;
  }
};
}

namespace {
class MachineBlockPlacement : public MachineFunctionPass {
  /// \brief A handle to the branch probability pass.
  const MachineBranchProbabilityInfo *MBPI;

  /// \brief A handle to the function-wide block frequency pass.
  const MachineBlockFrequencyInfo *MBFI;

  /// \brief A handle to the target's instruction info.
  const TargetInstrInfo *TII;

  /// \brief A prioritized list of edges in the BB-graph.
  ///
  /// For each function, we insert all control flow edges between BBs, along
  /// with their "global" frequency. The Frequency of an edge being taken is
  /// defined as the frequency of entering the source BB (from MBFI) times the
  /// probability of taking a particular branch out of that block (from MBPI).
  ///
  /// Once built, this list is sorted in ascending frequency, making the last
  /// edge the hottest one in the function.
  SmallVector<WeightedEdge, 64> Edges;

  /// \brief Allocator and owner of BlockChain structures.
  ///
  /// We build BlockChains lazily by merging together high probability BB
  /// sequences acording to the "Algo2" in the paper mentioned at the top of
  /// the file. To reduce malloc traffic, we allocate them using this slab-like
  /// allocator, and destroy them after the pass completes.
  SpecificBumpPtrAllocator<BlockChain> ChainAllocator;

  /// \brief Function wide BasicBlock to BlockChain mapping.
  ///
  /// This mapping allows efficiently moving from any given basic block to the
  /// BlockChain it participates in, if any. We use it to, among other things,
  /// allow implicitly defining edges between chains as the existing edges
  /// between basic blocks.
  DenseMap<MachineBasicBlock *, BlockChain *> BlockToChain;

  /// \brief A prioritized sequence of chains.
  ///
  /// We build up the ideal sequence of basic block chains in reverse order
  /// here, and then walk backwards to arrange the final function ordering.
  SmallVector<BlockChain *, 16> PChains;

#ifndef NDEBUG
  /// \brief A set of active chains used to sanity-check the pass algorithm.
  ///
  /// All operations on this member should be wrapped in an assert or NDEBUG.
  SmallPtrSet<BlockChain *, 16> ActiveChains;
#endif

  BlockChain *CreateChain(MachineBasicBlock *BB);
  void PrioritizeEdges(MachineFunction &F);
  void BuildBlockChains();
  void PrioritizeChains(MachineFunction &F);
  void PlaceBlockChains(MachineFunction &F);

public:
  static char ID; // Pass identification, replacement for typeid
  MachineBlockPlacement() : MachineFunctionPass(ID) {
    initializeMachineBlockPlacementPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &F);

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<MachineBranchProbabilityInfo>();
    AU.addRequired<MachineBlockFrequencyInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  const char *getPassName() const { return "Block Placement"; }
};
}

char MachineBlockPlacement::ID = 0;
INITIALIZE_PASS_BEGIN(MachineBlockPlacement, "block-placement2",
                      "Branch Probability Basic Block Placement", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfo)
INITIALIZE_PASS_END(MachineBlockPlacement, "block-placement2",
                    "Branch Probability Basic Block Placement", false, false)

FunctionPass *llvm::createMachineBlockPlacementPass() {
  return new MachineBlockPlacement();
}

namespace llvm {
/// \brief GraphTraits specialization for our BlockChain graph.
template <> struct GraphTraits<BlockChain *> {
  typedef BlockChain NodeType;
  typedef BlockChain::SuccIterator ChildIteratorType;

  static NodeType *getEntryNode(NodeType *N) { return N; }
  static BlockChain::SuccIterator child_begin(NodeType *N) {
    return BlockChain::SuccIterator(N);
  }
  static BlockChain::SuccIterator child_end(NodeType *N) {
    return BlockChain::SuccIterator::CreateEnd(N);
  }
};
}

/// \brief Helper to create a new chain for a single BB.
///
/// Takes care of growing the Chains, setting up the BlockChain object, and any
/// debug checking logic.
/// \returns A pointer to the new BlockChain.
BlockChain *MachineBlockPlacement::CreateChain(MachineBasicBlock *BB) {
  BlockChain *Chain =
    new (ChainAllocator.Allocate()) BlockChain(BlockToChain, BB);
  assert(ActiveChains.insert(Chain));
  return Chain;
}

/// \brief Build a prioritized list of edges.
///
/// The priority is determined by the product of the block frequency (how
/// likely it is to arrive at a particular block) times the probability of
/// taking this particular edge out of the block. This provides a function-wide
/// ordering of the edges.
void MachineBlockPlacement::PrioritizeEdges(MachineFunction &F) {
  assert(Edges.empty() && "Already have an edge list");
  SmallVector<MachineOperand, 4> Cond; // For AnalyzeBranch.
  BlockChain *RequiredChain = 0;
  for (MachineFunction::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    MachineBasicBlock *From = &*FI;
    // We only consider MBBs with analyzable branches. Even if the analysis
    // fails, if there is no fallthrough, we can still work with the MBB.
    MachineBasicBlock *TBB = 0, *FBB = 0;
    Cond.clear();
    if (TII->AnalyzeBranch(*From, TBB, FBB, Cond) && From->canFallThrough()) {
      // We push all unanalyzed blocks onto a chain eagerly to prevent them
      // from being split later. Create the chain if needed, otherwise just
      // keep track that these blocks reside on it.
      if (!RequiredChain)
        RequiredChain = CreateChain(From);
      else
        BlockToChain[From] = RequiredChain;
    } else {
      // As soon as we find an analyzable branch, add that block to and
      // finalize any required chain that has been started. The required chain
      // is only modeling potentially inexplicable fallthrough, so the first
      // block to have analyzable fallthrough is a known-safe stopping point.
      if (RequiredChain) {
        BlockToChain[From] = RequiredChain;
        RequiredChain->LastBB = FI;
        RequiredChain = 0;
      }
    }

    BlockFrequency BaseFrequency = MBFI->getBlockFreq(From);
    for (MachineBasicBlock::succ_iterator SI = From->succ_begin(),
                                          SE = From->succ_end();
         SI != SE; ++SI) {
      MachineBasicBlock *To = *SI;
      WeightedEdge WE = { BaseFrequency * MBPI->getEdgeProbability(From, To),
                          From, To };
      Edges.push_back(WE);
    }
  }
  assert(!RequiredChain && "Never found a terminator for a required chain");
  std::stable_sort(Edges.begin(), Edges.end());
}

/// \brief Build chains of basic blocks along hot paths.
///
/// Build chains by trying to merge each pair of blocks from the mostly costly
/// edge first. This is essentially "Algo2" from the Profile Guided Code
/// Placement paper. While each node is considered a chain of one block, this
/// routine lazily build the chain objects themselves so that when possible it
/// can just merge a block into an existing chain.
void MachineBlockPlacement::BuildBlockChains() {
  for (SmallVectorImpl<WeightedEdge>::reverse_iterator EI = Edges.rbegin(),
                                                       EE = Edges.rend();
       EI != EE; ++EI) {
    MachineBasicBlock *SourceB = EI->From, *DestB = EI->To;
    if (SourceB == DestB) continue;

    BlockChain *SourceChain = BlockToChain.lookup(SourceB);
    if (!SourceChain) SourceChain = CreateChain(SourceB);
    BlockChain *DestChain = BlockToChain.lookup(DestB);
    if (!DestChain) DestChain = CreateChain(DestB);
    if (SourceChain == DestChain)
      continue;

    bool IsSourceTail =
      SourceChain->LastBB == MachineFunction::iterator(SourceB);
    bool IsDestHead =
      DestChain->FirstBB == MachineFunction::iterator(DestB);

    if (!IsSourceTail || !IsDestHead)
      continue;

    SourceChain->merge(DestChain);
    assert(ActiveChains.erase(DestChain));
  }
}

/// \brief Prioritize the chains to minimize back-edges between chains.
///
/// This is the trickiest part of the placement algorithm. Each chain is
/// a hot-path through a sequence of basic blocks, but there are conditional
/// branches away from this hot path, and to some other chain. Hardware branch
/// predictors favor back edges over forward edges, and so it is desirable to
/// arrange the targets of branches away from a hot path and to some other
/// chain to come later in the function, making them forward branches, and
/// helping the branch predictor to predict fallthrough.
///
/// In some cases, this is easy. simply topologically walking from the entry
/// chain through its successors in order would work if there were no cycles
/// between the chains of blocks, but often there are. In such a case, we first
/// need to identify the participants in the cycle, and then rank them so that
/// the linearizing of the chains has the lowest *probability* of causing
/// a mispredicted branch. To compute the correct rank for a chain, we take the
/// complement of the branch probability for each branch leading away from the
/// chain and multiply it by the frequency of the source block for that branch.
/// This gives us the probability of that particular branch *not* being taken
/// in this function. The sum of these probabilities for each chain is used as
/// a rank, so that we order the chain with the highest such sum first.
/// FIXME: This seems like a good approximation, but there is probably a known
/// technique for ordering of an SCC given edge weights. It would be good to
/// use that, or even use its code if possible.
///
/// Also notable is that we prioritize the chains from the bottom up, and so
/// all of the "first" and "before" relationships end up inverted in the code.
void MachineBlockPlacement::PrioritizeChains(MachineFunction &F) {
  MachineBasicBlock *EntryB = &F.front();
  BlockChain *EntryChain = BlockToChain[EntryB];
  assert(EntryChain && "Missing chain for entry block");
  assert(EntryChain->FirstBB == F.begin() &&
         "Entry block is not the head of the entry block chain");

  // Form an SCC and walk it from the bottom up.
  SmallPtrSet<BlockChain *, 4> IsInSCC;
  for (scc_iterator<BlockChain *> I = scc_begin(EntryChain);
       !I.isAtEnd(); ++I) {
    const std::vector<BlockChain *> &SCC = *I;
    PChains.insert(PChains.end(), SCC.begin(), SCC.end());

    // If there is only one chain in the SCC, it's trivially sorted so just
    // bail out early. Sorting the SCC is expensive.
    if (SCC.size() == 1)
      continue;

    // We work strictly on the PChains range from here on out to maximize
    // locality.
    SmallVectorImpl<BlockChain *>::iterator SCCEnd = PChains.end(),
                                            SCCBegin = SCCEnd - SCC.size();
    IsInSCC.clear();
    IsInSCC.insert(SCCBegin, SCCEnd);

    // Compute the edge frequency of staying in a chain, despite the existency
    // of an edge to some other chain within this SCC.
    for (SmallVectorImpl<BlockChain *>::iterator SCCI = SCCBegin;
         SCCI != SCCEnd; ++SCCI) {
      BlockChain *Chain = *SCCI;

      // Special case the entry chain. Regardless of the weights of other
      // chains, the entry chain *must* come first, so move it to the end, and
      // avoid processing that chain at all.
      if (Chain == EntryChain) {
        --SCCEnd;
        if (SCCI == SCCEnd) break;
        Chain = *SCCI = *SCCEnd;
        *SCCEnd = EntryChain;
      }

      // Walk over every block in this chain looking for out-bound edges to
      // other chains in this SCC.
      for (MachineFunction::iterator BI = Chain->FirstBB,
                                     BE = llvm::next(Chain->LastBB);
           BI != BE; ++BI) {
        MachineBasicBlock *From = &*BI;
        for (MachineBasicBlock::succ_iterator SI = BI->succ_begin(),
                                              SE = BI->succ_end();
             SI != SE; ++SI) {
          MachineBasicBlock *To = *SI;
          if (!To || !IsInSCC.count(BlockToChain[To]))
            continue;
          BranchProbability ComplEdgeProb =
            MBPI->getEdgeProbability(From, To).getCompl();
          Chain->InChainEdgeFrequency +=
            MBFI->getBlockFreq(From) * ComplEdgeProb;
        }
      }
    }

    // Sort the chains within the SCC according to their edge frequencies,
    // which should make the least costly chain of blocks to mis-place be
    // ordered first in the prioritized sequence.
    std::stable_sort(SCCBegin, SCCEnd, ChainPtrPrioritySorter());
  }
}

/// \brief Splice the function blocks together based on the chain priorities.
///
/// Each chain is already represented as a contiguous range of blocks in the
/// function. Simply walk backwards down the prioritized chains and splice in
/// any chains out of order. Note that the first chain we visit is necessarily
/// the entry chain. It has no predecessors and so must be the top of the SCC.
/// Also, we cannot splice any chain prior to the entry chain as we can't
/// splice any blocks prior to the entry block.
void MachineBlockPlacement::PlaceBlockChains(MachineFunction &F) {
  assert(!PChains.empty() && "No chains were prioritized");
  assert(PChains.back() == BlockToChain[&F.front()] &&
         "The entry chain must always be the final chain");

  MachineFunction::iterator InsertPos = F.begin();
  for (SmallVectorImpl<BlockChain *>::reverse_iterator CI = PChains.rbegin(),
                                                       CE = PChains.rend();
       CI != CE; ++CI) {
    BlockChain *Chain = *CI;
    // Check that we process this chain only once for debugging.
    assert(ActiveChains.erase(Chain) && "Processed a chain twice");

    // If this chain is already in the right position, just skip past it.
    // Otherwise, splice it into position.
    if (InsertPos == Chain->FirstBB)
      InsertPos = llvm::next(Chain->LastBB);
    else
      F.splice(InsertPos, Chain->FirstBB, llvm::next(Chain->LastBB));
  }

  // Note that we can't assert this is empty as there may be unreachable blocks
  // in the function.
#ifndef NDEBUG
  ActiveChains.clear();
#endif

  // Now that every block is in its final position, update all of the
  // terminators.
  SmallVector<MachineOperand, 4> Cond; // For AnalyzeBranch.
  for (MachineFunction::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    // FIXME: It would be awesome of updateTerminator would just return rather
    // than assert when the branch cannot be analyzed in order to remove this
    // boiler plate.
    Cond.clear();
    MachineBasicBlock *TBB = 0, *FBB = 0; // For AnalyzeBranch.
    if (!TII->AnalyzeBranch(*FI, TBB, FBB, Cond))
      FI->updateTerminator();
  }
}

bool MachineBlockPlacement::runOnMachineFunction(MachineFunction &F) {
  // Check for single-block functions and skip them.
  if (llvm::next(F.begin()) == F.end())
    return false;

  MBPI = &getAnalysis<MachineBranchProbabilityInfo>();
  MBFI = &getAnalysis<MachineBlockFrequencyInfo>();
  TII = F.getTarget().getInstrInfo();
  assert(Edges.empty());
  assert(BlockToChain.empty());
  assert(PChains.empty());
  assert(ActiveChains.empty());

  PrioritizeEdges(F);
  BuildBlockChains();
  PrioritizeChains(F);
  PlaceBlockChains(F);

  Edges.clear();
  BlockToChain.clear();
  PChains.clear();
  ChainAllocator.DestroyAll();

  // We always return true as we have no way to track whether the final order
  // differs from the original order.
  return true;
}
