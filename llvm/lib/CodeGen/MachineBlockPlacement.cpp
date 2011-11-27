//===-- MachineBlockPlacement.cpp - Basic Block Code Layout optimization --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements basic block placement transformations using the CFG
// structure and branch probability estimates.
//
// The pass strives to preserve the structure of the CFG (that is, retain
// a topological ordering of basic blocks) in the absense of a *strong* signal
// to the contrary from probabilities. However, within the CFG structure, it
// attempts to choose an ordering which favors placing more likely sequences of
// blocks adjacent to each other.
//
// The algorithm works from the inner-most loop within a function outward, and
// at each stage walks through the basic blocks, trying to coalesce them into
// sequential chains where allowed by the CFG (or demanded by heavy
// probabilities). Finally, it walks the blocks in topological order, and the
// first time it reaches a chain of basic blocks, it schedules them in the
// function in-order.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "block-placement2"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumCondBranches, "Number of conditional branches");
STATISTIC(NumUncondBranches, "Number of uncondittional branches");
STATISTIC(CondBranchTakenFreq,
          "Potential frequency of taking conditional branches");
STATISTIC(UncondBranchTakenFreq,
          "Potential frequency of taking unconditional branches");

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
class BlockChain;
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
class BlockChain {
  /// \brief The sequence of blocks belonging to this chain.
  ///
  /// This is the sequence of blocks for a particular chain. These will be laid
  /// out in-order within the function.
  SmallVector<MachineBasicBlock *, 4> Blocks;

  /// \brief A handle to the function-wide basic block to block chain mapping.
  ///
  /// This is retained in each block chain to simplify the computation of child
  /// block chains for SCC-formation and iteration. We store the edges to child
  /// basic blocks, and map them back to their associated chains using this
  /// structure.
  BlockToChainMapType &BlockToChain;

public:
  /// \brief Construct a new BlockChain.
  ///
  /// This builds a new block chain representing a single basic block in the
  /// function. It also registers itself as the chain that block participates
  /// in with the BlockToChain mapping.
  BlockChain(BlockToChainMapType &BlockToChain, MachineBasicBlock *BB)
    : Blocks(1, BB), BlockToChain(BlockToChain), LoopPredecessors(0) {
    assert(BB && "Cannot create a chain with a null basic block");
    BlockToChain[BB] = this;
  }

  /// \brief Iterator over blocks within the chain.
  typedef SmallVectorImpl<MachineBasicBlock *>::iterator iterator;
  typedef SmallVectorImpl<MachineBasicBlock *>::reverse_iterator
      reverse_iterator;

  /// \brief Beginning of blocks within the chain.
  iterator begin() { return Blocks.begin(); }
  reverse_iterator rbegin() { return Blocks.rbegin(); }

  /// \brief End of blocks within the chain.
  iterator end() { return Blocks.end(); }
  reverse_iterator rend() { return Blocks.rend(); }

  /// \brief Merge a block chain into this one.
  ///
  /// This routine merges a block chain into this one. It takes care of forming
  /// a contiguous sequence of basic blocks, updating the edge list, and
  /// updating the block -> chain mapping. It does not free or tear down the
  /// old chain, but the old chain's block list is no longer valid.
  void merge(MachineBasicBlock *BB, BlockChain *Chain) {
    assert(BB);
    assert(!Blocks.empty());

    // Fast path in case we don't have a chain already.
    if (!Chain) {
      assert(!BlockToChain[BB]);
      Blocks.push_back(BB);
      BlockToChain[BB] = this;
      return;
    }

    assert(BB == *Chain->begin());
    assert(Chain->begin() != Chain->end());

    // Update the incoming blocks to point to this chain, and add them to the
    // chain structure.
    for (BlockChain::iterator BI = Chain->begin(), BE = Chain->end();
         BI != BE; ++BI) {
      Blocks.push_back(*BI);
      assert(BlockToChain[*BI] == Chain && "Incoming blocks not in chain");
      BlockToChain[*BI] = this;
    }
  }

  /// \brief Count of predecessors within the loop currently being processed.
  ///
  /// This count is updated at each loop we process to represent the number of
  /// in-loop predecessors of this chain.
  unsigned LoopPredecessors;
};
}

namespace {
class MachineBlockPlacement : public MachineFunctionPass {
  /// \brief A typedef for a block filter set.
  typedef SmallPtrSet<MachineBasicBlock *, 16> BlockFilterSet;

  /// \brief A handle to the branch probability pass.
  const MachineBranchProbabilityInfo *MBPI;

  /// \brief A handle to the function-wide block frequency pass.
  const MachineBlockFrequencyInfo *MBFI;

  /// \brief A handle to the loop info.
  const MachineLoopInfo *MLI;

  /// \brief A handle to the target's instruction info.
  const TargetInstrInfo *TII;

  /// \brief A handle to the target's lowering info.
  const TargetLowering *TLI;

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

  void markChainSuccessors(BlockChain &Chain,
                           MachineBasicBlock *LoopHeaderBB,
                           SmallVectorImpl<MachineBasicBlock *> &BlockWorkList,
                           const BlockFilterSet *BlockFilter = 0);
  MachineBasicBlock *selectBestSuccessor(MachineBasicBlock *BB,
                                         BlockChain &Chain,
                                         const BlockFilterSet *BlockFilter);
  MachineBasicBlock *selectBestCandidateBlock(
      BlockChain &Chain, SmallVectorImpl<MachineBasicBlock *> &WorkList,
      const BlockFilterSet *BlockFilter);
  MachineBasicBlock *getFirstUnplacedBlock(
      MachineFunction &F,
      const BlockChain &PlacedChain,
      MachineFunction::iterator &PrevUnplacedBlockIt,
      const BlockFilterSet *BlockFilter);
  void buildChain(MachineBasicBlock *BB, BlockChain &Chain,
                  SmallVectorImpl<MachineBasicBlock *> &BlockWorkList,
                  const BlockFilterSet *BlockFilter = 0);
  void rotateLoop(MachineLoop &L, BlockChain &LoopChain,
                  const BlockFilterSet &LoopBlockSet);
  void buildLoopChains(MachineFunction &F, MachineLoop &L);
  void buildCFGChains(MachineFunction &F);
  void AlignLoops(MachineFunction &F);

public:
  static char ID; // Pass identification, replacement for typeid
  MachineBlockPlacement() : MachineFunctionPass(ID) {
    initializeMachineBlockPlacementPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &F);

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<MachineBranchProbabilityInfo>();
    AU.addRequired<MachineBlockFrequencyInfo>();
    AU.addRequired<MachineLoopInfo>();
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
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(MachineBlockPlacement, "block-placement2",
                    "Branch Probability Basic Block Placement", false, false)

FunctionPass *llvm::createMachineBlockPlacementPass() {
  return new MachineBlockPlacement();
}

#ifndef NDEBUG
/// \brief Helper to print the name of a MBB.
///
/// Only used by debug logging.
static std::string getBlockName(MachineBasicBlock *BB) {
  std::string Result;
  raw_string_ostream OS(Result);
  OS << "BB#" << BB->getNumber()
     << " (derived from LLVM BB '" << BB->getName() << "')";
  OS.flush();
  return Result;
}

/// \brief Helper to print the number of a MBB.
///
/// Only used by debug logging.
static std::string getBlockNum(MachineBasicBlock *BB) {
  std::string Result;
  raw_string_ostream OS(Result);
  OS << "BB#" << BB->getNumber();
  OS.flush();
  return Result;
}
#endif

/// \brief Mark a chain's successors as having one fewer preds.
///
/// When a chain is being merged into the "placed" chain, this routine will
/// quickly walk the successors of each block in the chain and mark them as
/// having one fewer active predecessor. It also adds any successors of this
/// chain which reach the zero-predecessor state to the worklist passed in.
void MachineBlockPlacement::markChainSuccessors(
    BlockChain &Chain,
    MachineBasicBlock *LoopHeaderBB,
    SmallVectorImpl<MachineBasicBlock *> &BlockWorkList,
    const BlockFilterSet *BlockFilter) {
  // Walk all the blocks in this chain, marking their successors as having
  // a predecessor placed.
  for (BlockChain::iterator CBI = Chain.begin(), CBE = Chain.end();
       CBI != CBE; ++CBI) {
    // Add any successors for which this is the only un-placed in-loop
    // predecessor to the worklist as a viable candidate for CFG-neutral
    // placement. No subsequent placement of this block will violate the CFG
    // shape, so we get to use heuristics to choose a favorable placement.
    for (MachineBasicBlock::succ_iterator SI = (*CBI)->succ_begin(),
                                          SE = (*CBI)->succ_end();
         SI != SE; ++SI) {
      if (BlockFilter && !BlockFilter->count(*SI))
        continue;
      BlockChain &SuccChain = *BlockToChain[*SI];
      // Disregard edges within a fixed chain, or edges to the loop header.
      if (&Chain == &SuccChain || *SI == LoopHeaderBB)
        continue;

      // This is a cross-chain edge that is within the loop, so decrement the
      // loop predecessor count of the destination chain.
      if (SuccChain.LoopPredecessors > 0 && --SuccChain.LoopPredecessors == 0)
        BlockWorkList.push_back(*SuccChain.begin());
    }
  }
}

/// \brief Select the best successor for a block.
///
/// This looks across all successors of a particular block and attempts to
/// select the "best" one to be the layout successor. It only considers direct
/// successors which also pass the block filter. It will attempt to avoid
/// breaking CFG structure, but cave and break such structures in the case of
/// very hot successor edges.
///
/// \returns The best successor block found, or null if none are viable.
MachineBasicBlock *MachineBlockPlacement::selectBestSuccessor(
    MachineBasicBlock *BB, BlockChain &Chain,
    const BlockFilterSet *BlockFilter) {
  const BranchProbability HotProb(4, 5); // 80%

  MachineBasicBlock *BestSucc = 0;
  // FIXME: Due to the performance of the probability and weight routines in
  // the MBPI analysis, we manually compute probabilities using the edge
  // weights. This is suboptimal as it means that the somewhat subtle
  // definition of edge weight semantics is encoded here as well. We should
  // improve the MBPI interface to effeciently support query patterns such as
  // this.
  uint32_t BestWeight = 0;
  uint32_t WeightScale = 0;
  uint32_t SumWeight = MBPI->getSumForBlock(BB, WeightScale);
  DEBUG(dbgs() << "Attempting merge from: " << getBlockName(BB) << "\n");
  for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
                                        SE = BB->succ_end();
       SI != SE; ++SI) {
    if (BlockFilter && !BlockFilter->count(*SI))
      continue;
    BlockChain &SuccChain = *BlockToChain[*SI];
    if (&SuccChain == &Chain) {
      DEBUG(dbgs() << "    " << getBlockName(*SI) << " -> Already merged!\n");
      continue;
    }
    if (*SI != *SuccChain.begin()) {
      DEBUG(dbgs() << "    " << getBlockName(*SI) << " -> Mid chain!\n");
      continue;
    }

    uint32_t SuccWeight = MBPI->getEdgeWeight(BB, *SI);
    BranchProbability SuccProb(SuccWeight / WeightScale, SumWeight);

    // Only consider successors which are either "hot", or wouldn't violate
    // any CFG constraints.
    if (SuccChain.LoopPredecessors != 0) {
      if (SuccProb < HotProb) {
        DEBUG(dbgs() << "    " << getBlockName(*SI) << " -> CFG conflict\n");
        continue;
      }

      // Make sure that a hot successor doesn't have a globally more important
      // predecessor.
      BlockFrequency CandidateEdgeFreq
        = MBFI->getBlockFreq(BB) * SuccProb * HotProb.getCompl();
      bool BadCFGConflict = false;
      for (MachineBasicBlock::pred_iterator PI = (*SI)->pred_begin(),
                                            PE = (*SI)->pred_end();
           PI != PE; ++PI) {
        if (*PI == *SI || (BlockFilter && !BlockFilter->count(*PI)) ||
            BlockToChain[*PI] == &Chain)
          continue;
        BlockFrequency PredEdgeFreq
          = MBFI->getBlockFreq(*PI) * MBPI->getEdgeProbability(*PI, *SI);
        if (PredEdgeFreq >= CandidateEdgeFreq) {
          BadCFGConflict = true;
          break;
        }
      }
      if (BadCFGConflict) {
        DEBUG(dbgs() << "    " << getBlockName(*SI)
                               << " -> non-cold CFG conflict\n");
        continue;
      }
    }

    DEBUG(dbgs() << "    " << getBlockName(*SI) << " -> " << SuccProb
                 << " (prob)"
                 << (SuccChain.LoopPredecessors != 0 ? " (CFG break)" : "")
                 << "\n");
    if (BestSucc && BestWeight >= SuccWeight)
      continue;
    BestSucc = *SI;
    BestWeight = SuccWeight;
  }
  return BestSucc;
}

namespace {
/// \brief Predicate struct to detect blocks already placed.
class IsBlockPlaced {
  const BlockChain &PlacedChain;
  const BlockToChainMapType &BlockToChain;

public:
  IsBlockPlaced(const BlockChain &PlacedChain,
                const BlockToChainMapType &BlockToChain)
      : PlacedChain(PlacedChain), BlockToChain(BlockToChain) {}

  bool operator()(MachineBasicBlock *BB) const {
    return BlockToChain.lookup(BB) == &PlacedChain;
  }
};
}

/// \brief Select the best block from a worklist.
///
/// This looks through the provided worklist as a list of candidate basic
/// blocks and select the most profitable one to place. The definition of
/// profitable only really makes sense in the context of a loop. This returns
/// the most frequently visited block in the worklist, which in the case of
/// a loop, is the one most desirable to be physically close to the rest of the
/// loop body in order to improve icache behavior.
///
/// \returns The best block found, or null if none are viable.
MachineBasicBlock *MachineBlockPlacement::selectBestCandidateBlock(
    BlockChain &Chain, SmallVectorImpl<MachineBasicBlock *> &WorkList,
    const BlockFilterSet *BlockFilter) {
  // Once we need to walk the worklist looking for a candidate, cleanup the
  // worklist of already placed entries.
  // FIXME: If this shows up on profiles, it could be folded (at the cost of
  // some code complexity) into the loop below.
  WorkList.erase(std::remove_if(WorkList.begin(), WorkList.end(),
                                IsBlockPlaced(Chain, BlockToChain)),
                 WorkList.end());

  MachineBasicBlock *BestBlock = 0;
  BlockFrequency BestFreq;
  for (SmallVectorImpl<MachineBasicBlock *>::iterator WBI = WorkList.begin(),
                                                      WBE = WorkList.end();
       WBI != WBE; ++WBI) {
    assert(!BlockFilter || BlockFilter->count(*WBI));
    BlockChain &SuccChain = *BlockToChain[*WBI];
    if (&SuccChain == &Chain) {
      DEBUG(dbgs() << "    " << getBlockName(*WBI)
                   << " -> Already merged!\n");
      continue;
    }
    assert(SuccChain.LoopPredecessors == 0 && "Found CFG-violating block");

    BlockFrequency CandidateFreq = MBFI->getBlockFreq(*WBI);
    DEBUG(dbgs() << "    " << getBlockName(*WBI) << " -> " << CandidateFreq
                 << " (freq)\n");
    if (BestBlock && BestFreq >= CandidateFreq)
      continue;
    BestBlock = *WBI;
    BestFreq = CandidateFreq;
  }
  return BestBlock;
}

/// \brief Retrieve the first unplaced basic block.
///
/// This routine is called when we are unable to use the CFG to walk through
/// all of the basic blocks and form a chain due to unnatural loops in the CFG.
/// We walk through the function's blocks in order, starting from the
/// LastUnplacedBlockIt. We update this iterator on each call to avoid
/// re-scanning the entire sequence on repeated calls to this routine.
MachineBasicBlock *MachineBlockPlacement::getFirstUnplacedBlock(
    MachineFunction &F, const BlockChain &PlacedChain,
    MachineFunction::iterator &PrevUnplacedBlockIt,
    const BlockFilterSet *BlockFilter) {
  for (MachineFunction::iterator I = PrevUnplacedBlockIt, E = F.end(); I != E;
       ++I) {
    if (BlockFilter && !BlockFilter->count(I))
      continue;
    if (BlockToChain[I] != &PlacedChain) {
      PrevUnplacedBlockIt = I;
      // Now select the head of the chain to which the unplaced block belongs
      // as the block to place. This will force the entire chain to be placed,
      // and satisfies the requirements of merging chains.
      return *BlockToChain[I]->begin();
    }
  }
  return 0;
}

void MachineBlockPlacement::buildChain(
    MachineBasicBlock *BB,
    BlockChain &Chain,
    SmallVectorImpl<MachineBasicBlock *> &BlockWorkList,
    const BlockFilterSet *BlockFilter) {
  assert(BB);
  assert(BlockToChain[BB] == &Chain);
  MachineFunction &F = *BB->getParent();
  MachineFunction::iterator PrevUnplacedBlockIt = F.begin();

  MachineBasicBlock *LoopHeaderBB = BB;
  markChainSuccessors(Chain, LoopHeaderBB, BlockWorkList, BlockFilter);
  BB = *llvm::prior(Chain.end());
  for (;;) {
    assert(BB);
    assert(BlockToChain[BB] == &Chain);
    assert(*llvm::prior(Chain.end()) == BB);
    MachineBasicBlock *BestSucc = 0;

    // Look for the best viable successor if there is one to place immediately
    // after this block.
    BestSucc = selectBestSuccessor(BB, Chain, BlockFilter);

    // If an immediate successor isn't available, look for the best viable
    // block among those we've identified as not violating the loop's CFG at
    // this point. This won't be a fallthrough, but it will increase locality.
    if (!BestSucc)
      BestSucc = selectBestCandidateBlock(Chain, BlockWorkList, BlockFilter);

    if (!BestSucc) {
      BestSucc = getFirstUnplacedBlock(F, Chain, PrevUnplacedBlockIt,
                                       BlockFilter);
      if (!BestSucc)
        break;

      DEBUG(dbgs() << "Unnatural loop CFG detected, forcibly merging the "
                      "layout successor until the CFG reduces\n");
    }

    // Place this block, updating the datastructures to reflect its placement.
    BlockChain &SuccChain = *BlockToChain[BestSucc];
    // Zero out LoopPredecessors for the successor we're about to merge in case
    // we selected a successor that didn't fit naturally into the CFG.
    SuccChain.LoopPredecessors = 0;
    DEBUG(dbgs() << "Merging from " << getBlockNum(BB)
                 << " to " << getBlockNum(BestSucc) << "\n");
    markChainSuccessors(SuccChain, LoopHeaderBB, BlockWorkList, BlockFilter);
    Chain.merge(BestSucc, &SuccChain);
    BB = *llvm::prior(Chain.end());
  };

  DEBUG(dbgs() << "Finished forming chain for header block "
               << getBlockNum(*Chain.begin()) << "\n");
}

/// \brief Attempt to rotate loop chains ending in an unconditional backedge.
///
/// This is a very conservative attempt to rotate unconditional backedge jumps
/// into fallthrough opportunities. It only attempts to perform the rotation
/// when it is trivial to find a block within the loop which has a conditional
/// loop exit. This block is then made the bottom of the chain, and the in-loop
/// fallthrough block the top. That turns a conditional branch out of the loop
/// into a conditional branch to the top of the loop while completely
/// eliminitating an unconditional branch within the loop.
void MachineBlockPlacement::rotateLoop(MachineLoop &L,
                                       BlockChain &LoopChain,
                                       const BlockFilterSet &LoopBlockSet) {
  MachineBasicBlock *Header = *L.block_begin();
  // Ensure that we have a chain of blocks which starts with the loop header.
  // Otherwise, rotating the blocks might break an analyzable branch.
  if (Header != *LoopChain.begin())
    return;

  // We only support rotating the loop chain as a unit, so look directly at the
  // back of the chain and check that it has a backedge.
  MachineBasicBlock *Latch = *llvm::prior(LoopChain.end());
  if (Latch == Header ||
      !Latch->isSuccessor(Header))
    return;

  // We need to analyze the branch and determine if rotating the loop will
  // completely remove a branch. We bail if the analysis fails or we don't find
  // an unconditional backedge. Note that this has to handle cases where the
  // original order was rotated, and the chain has un-done it. As a result,
  // there may not (yet) be the uncondiationl branch, but we can tell whether
  // one will be added when updating the terminators.
  SmallVector<MachineOperand, 4> Cond;
  MachineBasicBlock *TBB = 0, *FBB = 0;
  if (TII->AnalyzeBranch(*Latch, TBB, FBB, Cond) || !Cond.empty())
    return;

  // Next we need to find a split point. This rotate algorithm is *very*
  // narrow, and it only tries to do the rotate when it can find a split point
  // which is an analyzable branch that exits the loop. Splitting there allows
  // us to form a fallthrough out of the loop and a conditional jump to the top
  // of the loop after rotation.
  MachineBasicBlock *NewBottom = 0, *NewTop = 0;
  BlockChain::iterator SplitIt = LoopChain.end();
  for (BlockChain::reverse_iterator I = llvm::next(LoopChain.rbegin()),
                                    E = LoopChain.rend();
       I != E; ++I) {
    Cond.clear();
    TBB = FBB = 0;
    // Ensure that this is a block with a conditional branch which we can
    // analyze and re-form after rotating the loop. While it might be tempting
    // to use the TBB or FBB output parameters from this, they will often be
    // lies as they are only correct after the terminators have been updated,
    // and we are mid-chain formation.
    if (TII->AnalyzeBranch(**I, TBB, FBB, Cond) || Cond.empty())
      continue;

    // See if this block is an exiting block from the loop. LoopInfo provides
    // a nice API for this, but it's actuall N*M runtime where N is the number
    // of blocks in the loop and M is the number of successors. We can
    // eliminate the N by doing this ourselves.
    // FIXME: The LoopInfo datastructure should be improved for these types of
    // queries.
    MachineBasicBlock *ExitB = 0;
    for (MachineBasicBlock::succ_iterator SI = (*I)->succ_begin(), SE = (*I)->succ_end();
         SI != SE; ++SI) {
      if (!(*SI)->isLandingPad() && *SI != *I && !LoopBlockSet.count(*SI)) {
        ExitB = *SI;
        break;
      }
    }
    if (!ExitB)
      continue;

    NewBottom = *I;
    NewTop = *llvm::prior(I);
    SplitIt = I.base();
    break;
  }
  if (!NewBottom || !NewTop ||
      SplitIt == LoopChain.end() || SplitIt == LoopChain.begin())
    return;
  assert(BlockToChain[NewBottom] == &LoopChain);
  assert(BlockToChain[NewTop] == &LoopChain);
  assert(*SplitIt == NewTop);

  // Rotate the chain and we're done.
  DEBUG(dbgs() << "Rotating loop headed by: " << getBlockName(Header) << "\n"
               << "  new top:    " << getBlockName(NewTop) << "\n"
               << "  new bottom: " << getBlockName(NewBottom) << "\n");
  std::rotate(LoopChain.begin(), SplitIt, LoopChain.end());
}

/// \brief Forms basic block chains from the natural loop structures.
///
/// These chains are designed to preserve the existing *structure* of the code
/// as much as possible. We can then stitch the chains together in a way which
/// both preserves the topological structure and minimizes taken conditional
/// branches.
void MachineBlockPlacement::buildLoopChains(MachineFunction &F,
                                            MachineLoop &L) {
  // First recurse through any nested loops, building chains for those inner
  // loops.
  for (MachineLoop::iterator LI = L.begin(), LE = L.end(); LI != LE; ++LI)
    buildLoopChains(F, **LI);

  SmallVector<MachineBasicBlock *, 16> BlockWorkList;
  BlockFilterSet LoopBlockSet(L.block_begin(), L.block_end());
  BlockChain &LoopChain = *BlockToChain[L.getHeader()];

  // FIXME: This is a really lame way of walking the chains in the loop: we
  // walk the blocks, and use a set to prevent visiting a particular chain
  // twice.
  SmallPtrSet<BlockChain *, 4> UpdatedPreds;
  for (MachineLoop::block_iterator BI = L.block_begin(),
                                   BE = L.block_end();
       BI != BE; ++BI) {
    BlockChain &Chain = *BlockToChain[*BI];
    if (!UpdatedPreds.insert(&Chain) || BI == L.block_begin())
      continue;

    assert(Chain.LoopPredecessors == 0);
    for (BlockChain::iterator BCI = Chain.begin(), BCE = Chain.end();
         BCI != BCE; ++BCI) {
      assert(BlockToChain[*BCI] == &Chain);
      for (MachineBasicBlock::pred_iterator PI = (*BCI)->pred_begin(),
                                            PE = (*BCI)->pred_end();
           PI != PE; ++PI) {
        if (BlockToChain[*PI] == &Chain || !LoopBlockSet.count(*PI))
          continue;
        ++Chain.LoopPredecessors;
      }
    }

    if (Chain.LoopPredecessors == 0)
      BlockWorkList.push_back(*Chain.begin());
  }

  buildChain(*L.block_begin(), LoopChain, BlockWorkList, &LoopBlockSet);
  rotateLoop(L, LoopChain, LoopBlockSet);

  DEBUG({
    // Crash at the end so we get all of the debugging output first.
    bool BadLoop = false;
    if (LoopChain.LoopPredecessors) {
      BadLoop = true;
      dbgs() << "Loop chain contains a block without its preds placed!\n"
             << "  Loop header:  " << getBlockName(*L.block_begin()) << "\n"
             << "  Chain header: " << getBlockName(*LoopChain.begin()) << "\n";
    }
    for (BlockChain::iterator BCI = LoopChain.begin(), BCE = LoopChain.end();
         BCI != BCE; ++BCI)
      if (!LoopBlockSet.erase(*BCI)) {
        // We don't mark the loop as bad here because there are real situations
        // where this can occur. For example, with an unanalyzable fallthrough
        // from a loop block to a non-loop block or vice versa.
        dbgs() << "Loop chain contains a block not contained by the loop!\n"
               << "  Loop header:  " << getBlockName(*L.block_begin()) << "\n"
               << "  Chain header: " << getBlockName(*LoopChain.begin()) << "\n"
               << "  Bad block:    " << getBlockName(*BCI) << "\n";
      }

    if (!LoopBlockSet.empty()) {
      BadLoop = true;
      for (BlockFilterSet::iterator LBI = LoopBlockSet.begin(),
                                    LBE = LoopBlockSet.end();
           LBI != LBE; ++LBI)
        dbgs() << "Loop contains blocks never placed into a chain!\n"
               << "  Loop header:  " << getBlockName(*L.block_begin()) << "\n"
               << "  Chain header: " << getBlockName(*LoopChain.begin()) << "\n"
               << "  Bad block:    " << getBlockName(*LBI) << "\n";
    }
    assert(!BadLoop && "Detected problems with the placement of this loop.");
  });
}

void MachineBlockPlacement::buildCFGChains(MachineFunction &F) {
  // Ensure that every BB in the function has an associated chain to simplify
  // the assumptions of the remaining algorithm.
  SmallVector<MachineOperand, 4> Cond; // For AnalyzeBranch.
  for (MachineFunction::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    MachineBasicBlock *BB = FI;
    BlockChain *Chain
      = new (ChainAllocator.Allocate()) BlockChain(BlockToChain, BB);
    // Also, merge any blocks which we cannot reason about and must preserve
    // the exact fallthrough behavior for.
    for (;;) {
      Cond.clear();
      MachineBasicBlock *TBB = 0, *FBB = 0; // For AnalyzeBranch.
      if (!TII->AnalyzeBranch(*BB, TBB, FBB, Cond) || !FI->canFallThrough())
        break;

      MachineFunction::iterator NextFI(llvm::next(FI));
      MachineBasicBlock *NextBB = NextFI;
      // Ensure that the layout successor is a viable block, as we know that
      // fallthrough is a possibility.
      assert(NextFI != FE && "Can't fallthrough past the last block.");
      DEBUG(dbgs() << "Pre-merging due to unanalyzable fallthrough: "
                   << getBlockName(BB) << " -> " << getBlockName(NextBB)
                   << "\n");
      Chain->merge(NextBB, 0);
      FI = NextFI;
      BB = NextBB;
    }
  }

  // Build any loop-based chains.
  for (MachineLoopInfo::iterator LI = MLI->begin(), LE = MLI->end(); LI != LE;
       ++LI)
    buildLoopChains(F, **LI);

  SmallVector<MachineBasicBlock *, 16> BlockWorkList;

  SmallPtrSet<BlockChain *, 4> UpdatedPreds;
  for (MachineFunction::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    MachineBasicBlock *BB = &*FI;
    BlockChain &Chain = *BlockToChain[BB];
    if (!UpdatedPreds.insert(&Chain))
      continue;

    assert(Chain.LoopPredecessors == 0);
    for (BlockChain::iterator BCI = Chain.begin(), BCE = Chain.end();
         BCI != BCE; ++BCI) {
      assert(BlockToChain[*BCI] == &Chain);
      for (MachineBasicBlock::pred_iterator PI = (*BCI)->pred_begin(),
                                            PE = (*BCI)->pred_end();
           PI != PE; ++PI) {
        if (BlockToChain[*PI] == &Chain)
          continue;
        ++Chain.LoopPredecessors;
      }
    }

    if (Chain.LoopPredecessors == 0)
      BlockWorkList.push_back(*Chain.begin());
  }

  BlockChain &FunctionChain = *BlockToChain[&F.front()];
  buildChain(&F.front(), FunctionChain, BlockWorkList);

  typedef SmallPtrSet<MachineBasicBlock *, 16> FunctionBlockSetType;
  DEBUG({
    // Crash at the end so we get all of the debugging output first.
    bool BadFunc = false;
    FunctionBlockSetType FunctionBlockSet;
    for (MachineFunction::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
      FunctionBlockSet.insert(FI);

    for (BlockChain::iterator BCI = FunctionChain.begin(),
                              BCE = FunctionChain.end();
         BCI != BCE; ++BCI)
      if (!FunctionBlockSet.erase(*BCI)) {
        BadFunc = true;
        dbgs() << "Function chain contains a block not in the function!\n"
               << "  Bad block:    " << getBlockName(*BCI) << "\n";
      }

    if (!FunctionBlockSet.empty()) {
      BadFunc = true;
      for (FunctionBlockSetType::iterator FBI = FunctionBlockSet.begin(),
                                          FBE = FunctionBlockSet.end();
           FBI != FBE; ++FBI)
        dbgs() << "Function contains blocks never placed into a chain!\n"
               << "  Bad block:    " << getBlockName(*FBI) << "\n";
    }
    assert(!BadFunc && "Detected problems with the block placement.");
  });

  // Splice the blocks into place.
  MachineFunction::iterator InsertPos = F.begin();
  for (BlockChain::iterator BI = FunctionChain.begin(),
                            BE = FunctionChain.end();
       BI != BE; ++BI) {
    DEBUG(dbgs() << (BI == FunctionChain.begin() ? "Placing chain "
                                                  : "          ... ")
          << getBlockName(*BI) << "\n");
    if (InsertPos != MachineFunction::iterator(*BI))
      F.splice(InsertPos, *BI);
    else
      ++InsertPos;

    // Update the terminator of the previous block.
    if (BI == FunctionChain.begin())
      continue;
    MachineBasicBlock *PrevBB = llvm::prior(MachineFunction::iterator(*BI));

    // FIXME: It would be awesome of updateTerminator would just return rather
    // than assert when the branch cannot be analyzed in order to remove this
    // boiler plate.
    Cond.clear();
    MachineBasicBlock *TBB = 0, *FBB = 0; // For AnalyzeBranch.
    if (!TII->AnalyzeBranch(*PrevBB, TBB, FBB, Cond))
      PrevBB->updateTerminator();
  }

  // Fixup the last block.
  Cond.clear();
  MachineBasicBlock *TBB = 0, *FBB = 0; // For AnalyzeBranch.
  if (!TII->AnalyzeBranch(F.back(), TBB, FBB, Cond))
    F.back().updateTerminator();
}

/// \brief Recursive helper to align a loop and any nested loops.
static void AlignLoop(MachineFunction &F, MachineLoop *L, unsigned Align) {
  // Recurse through nested loops.
  for (MachineLoop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    AlignLoop(F, *I, Align);

  L->getTopBlock()->setAlignment(Align);
}

/// \brief Align loop headers to target preferred alignments.
void MachineBlockPlacement::AlignLoops(MachineFunction &F) {
  if (F.getFunction()->hasFnAttr(Attribute::OptimizeForSize))
    return;

  unsigned Align = TLI->getPrefLoopAlignment();
  if (!Align)
    return;  // Don't care about loop alignment.

  for (MachineLoopInfo::iterator I = MLI->begin(), E = MLI->end(); I != E; ++I)
    AlignLoop(F, *I, Align);
}

bool MachineBlockPlacement::runOnMachineFunction(MachineFunction &F) {
  // Check for single-block functions and skip them.
  if (llvm::next(F.begin()) == F.end())
    return false;

  MBPI = &getAnalysis<MachineBranchProbabilityInfo>();
  MBFI = &getAnalysis<MachineBlockFrequencyInfo>();
  MLI = &getAnalysis<MachineLoopInfo>();
  TII = F.getTarget().getInstrInfo();
  TLI = F.getTarget().getTargetLowering();
  assert(BlockToChain.empty());

  buildCFGChains(F);
  AlignLoops(F);

  BlockToChain.clear();
  ChainAllocator.DestroyAll();

  // We always return true as we have no way to track whether the final order
  // differs from the original order.
  return true;
}

namespace {
/// \brief A pass to compute block placement statistics.
///
/// A separate pass to compute interesting statistics for evaluating block
/// placement. This is separate from the actual placement pass so that they can
/// be computed in the absense of any placement transformations or when using
/// alternative placement strategies.
class MachineBlockPlacementStats : public MachineFunctionPass {
  /// \brief A handle to the branch probability pass.
  const MachineBranchProbabilityInfo *MBPI;

  /// \brief A handle to the function-wide block frequency pass.
  const MachineBlockFrequencyInfo *MBFI;

public:
  static char ID; // Pass identification, replacement for typeid
  MachineBlockPlacementStats() : MachineFunctionPass(ID) {
    initializeMachineBlockPlacementStatsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &F);

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<MachineBranchProbabilityInfo>();
    AU.addRequired<MachineBlockFrequencyInfo>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  const char *getPassName() const { return "Block Placement Stats"; }
};
}

char MachineBlockPlacementStats::ID = 0;
INITIALIZE_PASS_BEGIN(MachineBlockPlacementStats, "block-placement-stats",
                      "Basic Block Placement Stats", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfo)
INITIALIZE_PASS_END(MachineBlockPlacementStats, "block-placement-stats",
                    "Basic Block Placement Stats", false, false)

FunctionPass *llvm::createMachineBlockPlacementStatsPass() {
  return new MachineBlockPlacementStats();
}

bool MachineBlockPlacementStats::runOnMachineFunction(MachineFunction &F) {
  // Check for single-block functions and skip them.
  if (llvm::next(F.begin()) == F.end())
    return false;

  MBPI = &getAnalysis<MachineBranchProbabilityInfo>();
  MBFI = &getAnalysis<MachineBlockFrequencyInfo>();

  for (MachineFunction::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BlockFrequency BlockFreq = MBFI->getBlockFreq(I);
    Statistic &NumBranches = (I->succ_size() > 1) ? NumCondBranches
                                                  : NumUncondBranches;
    Statistic &BranchTakenFreq = (I->succ_size() > 1) ? CondBranchTakenFreq
                                                      : UncondBranchTakenFreq;
    for (MachineBasicBlock::succ_iterator SI = I->succ_begin(),
                                          SE = I->succ_end();
         SI != SE; ++SI) {
      // Skip if this successor is a fallthrough.
      if (I->isLayoutSuccessor(*SI))
        continue;

      BlockFrequency EdgeFreq = BlockFreq * MBPI->getEdgeProbability(I, *SI);
      ++NumBranches;
      BranchTakenFreq += EdgeFreq.getFrequency();
    }
  }

  return false;
}

