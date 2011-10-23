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
    : Blocks(1, BB), BlockToChain(BlockToChain) {
    assert(BB && "Cannot create a chain with a null basic block");
    BlockToChain[BB] = this;
  }

  /// \brief Iterator over blocks within the chain.
  typedef SmallVectorImpl<MachineBasicBlock *>::const_iterator iterator;

  /// \brief Beginning of blocks within the chain.
  iterator begin() const { return Blocks.begin(); }

  /// \brief End of blocks within the chain.
  iterator end() const { return Blocks.end(); }

  /// \brief Merge a block chain into this one.
  ///
  /// This routine merges a block chain into this one. It takes care of forming
  /// a contiguous sequence of basic blocks, updating the edge list, and
  /// updating the block -> chain mapping. It does not free or tear down the
  /// old chain, but the old chain's block list is no longer valid.
  void merge(MachineBasicBlock *BB, BlockChain *Chain) {
    assert(BB);
    assert(!Blocks.empty());
    assert(Blocks.back()->isSuccessor(BB));

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

  BlockChain *CreateChain(MachineBasicBlock *BB);
  void mergeSuccessor(MachineBasicBlock *BB, BlockChain *Chain,
                      BlockFilterSet *Filter = 0);
  void buildLoopChains(MachineFunction &F, MachineLoop &L);
  void buildCFGChains(MachineFunction &F);
  void placeChainsTopologically(MachineFunction &F);
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

/// \brief Helper to create a new chain for a single BB.
///
/// Takes care of growing the Chains, setting up the BlockChain object, and any
/// debug checking logic.
/// \returns A pointer to the new BlockChain.
BlockChain *MachineBlockPlacement::CreateChain(MachineBasicBlock *BB) {
  BlockChain *Chain =
    new (ChainAllocator.Allocate()) BlockChain(BlockToChain, BB);
  //assert(ActiveChains.insert(Chain));
  return Chain;
}

/// \brief Merge a chain with any viable successor.
///
/// This routine walks the predecessors of the current block, looking for
/// viable merge candidates. It has strict rules it uses to determine when
/// a predecessor can be merged with the current block, which center around
/// preserving the CFG structure. It performs the merge if any viable candidate
/// is found.
void MachineBlockPlacement::mergeSuccessor(MachineBasicBlock *BB,
                                           BlockChain *Chain,
                                           BlockFilterSet *Filter) {
  assert(BB);
  assert(Chain);

  // If this block is not at the end of its chain, it cannot merge with any
  // other chain.
  if (Chain && *llvm::prior(Chain->end()) != BB)
    return;

  // Walk through the successors looking for the highest probability edge.
  // FIXME: This is an annoying way to do the comparison, but it's correct.
  // Support should be added to BranchProbability to properly compare two.
  MachineBasicBlock *Successor = 0;
  BlockFrequency BestFreq;
  DEBUG(dbgs() << "Attempting merge from: " << getBlockName(BB) << "\n");
  for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
                                        SE = BB->succ_end();
       SI != SE; ++SI) {
    if (BB == *SI || (Filter && !Filter->count(*SI)))
      continue;

    BlockFrequency SuccFreq(BlockFrequency::getEntryFrequency());
    SuccFreq *= MBPI->getEdgeProbability(BB, *SI);
    DEBUG(dbgs() << "    " << getBlockName(*SI) << " -> " << SuccFreq << "\n");
    if (!Successor || SuccFreq > BestFreq || (!(SuccFreq < BestFreq) &&
                                              BB->isLayoutSuccessor(*SI))) {
      Successor = *SI;
      BestFreq = SuccFreq;
    }
  }
  if (!Successor)
    return;

  // Grab a chain if it exists already for this successor and make sure the
  // successor is at the start of the chain as we can't merge mid-chain. Also,
  // if the successor chain is the same as our chain, we're already merged.
  BlockChain *SuccChain = BlockToChain[Successor];
  if (SuccChain && (SuccChain == Chain || Successor != *SuccChain->begin()))
    return;

  // We only merge chains across a CFG merge when the desired merge path is
  // significantly hotter than the incoming edge. We define a hot edge more
  // strictly than the BranchProbabilityInfo does, as the two predecessor
  // blocks may have dramatically different incoming probabilities we need to
  // account for. Therefor we use the "global" edge weight which is the
  // branch's probability times the block frequency of the predecessor.
  BlockFrequency MergeWeight = MBFI->getBlockFreq(BB);
  MergeWeight *= MBPI->getEdgeProbability(BB, Successor);
  // We only want to consider breaking the CFG when the merge weight is much
  // higher (80% vs. 20%), so multiply it by 1/4. This will require the merged
  // edge to be 4x more likely before we disrupt the CFG. This number matches
  // the definition of "hot" in BranchProbabilityAnalysis (80% vs. 20%).
  MergeWeight *= BranchProbability(1, 4);
  for (MachineBasicBlock::pred_iterator PI = Successor->pred_begin(),
                                        PE = Successor->pred_end();
       PI != PE; ++PI) {
    if (BB == *PI || Successor == *PI) continue;
    BlockFrequency PredWeight = MBFI->getBlockFreq(*PI);
    PredWeight *= MBPI->getEdgeProbability(*PI, Successor);

    // Return on the first predecessor we find which outstrips our merge weight.
    if (MergeWeight < PredWeight)
      return;
    DEBUG(dbgs() << "Breaking CFG edge!\n"
                 << "  Edge from " << getBlockNum(BB) << " to "
                 << getBlockNum(Successor) << ": " << MergeWeight << "\n"
                 << "        vs. " << getBlockNum(BB) << " to "
                 << getBlockNum(*PI) << ": " << PredWeight << "\n");
  }

  DEBUG(dbgs() << "Merging from " << getBlockNum(BB) << " to "
               << getBlockNum(Successor) << "\n");
  Chain->merge(Successor, SuccChain);
}

/// \brief Forms basic block chains from the natural loop structures.
///
/// These chains are designed to preserve the existing *structure* of the code
/// as much as possible. We can then stitch the chains together in a way which
/// both preserves the topological structure and minimizes taken conditional
/// branches.
void MachineBlockPlacement::buildLoopChains(MachineFunction &F, MachineLoop &L) {
  // First recurse through any nested loops, building chains for those inner
  // loops.
  for (MachineLoop::iterator LI = L.begin(), LE = L.end(); LI != LE; ++LI)
    buildLoopChains(F, **LI);

  SmallPtrSet<MachineBasicBlock *, 16> LoopBlockSet(L.block_begin(),
                                                    L.block_end());

  // Begin building up a set of chains of blocks within this loop which should
  // remain contiguous. Some of the blocks already belong to a chain which
  // represents an inner loop.
  for (MachineLoop::block_iterator BI = L.block_begin(), BE = L.block_end();
       BI != BE; ++BI) {
    MachineBasicBlock *BB = *BI;
    BlockChain *Chain = BlockToChain[BB];
    if (!Chain) Chain = CreateChain(BB);
    mergeSuccessor(BB, Chain, &LoopBlockSet);
  }
}

void MachineBlockPlacement::buildCFGChains(MachineFunction &F) {
  // First build any loop-based chains.
  for (MachineLoopInfo::iterator LI = MLI->begin(), LE = MLI->end(); LI != LE;
       ++LI)
    buildLoopChains(F, **LI);

  // Now walk the blocks of the function forming chains where they don't
  // violate any CFG structure.
  for (MachineFunction::iterator BI = F.begin(), BE = F.end();
       BI != BE; ++BI) {
    MachineBasicBlock *BB = BI;
    BlockChain *Chain = BlockToChain[BB];
    if (!Chain) Chain = CreateChain(BB);
    mergeSuccessor(BB, Chain);
  }
}

void MachineBlockPlacement::placeChainsTopologically(MachineFunction &F) {
  MachineBasicBlock *EntryB = &F.front();
  BlockChain *EntryChain = BlockToChain[EntryB];
  assert(EntryChain && "Missing chain for entry block");
  assert(*EntryChain->begin() == EntryB &&
         "Entry block is not the head of the entry block chain");

  // Walk the blocks in RPO, and insert each block for a chain in order the
  // first time we see that chain.
  MachineFunction::iterator InsertPos = F.begin();
  SmallPtrSet<BlockChain *, 16> VisitedChains;
  ReversePostOrderTraversal<MachineBasicBlock *> RPOT(EntryB);
  typedef ReversePostOrderTraversal<MachineBasicBlock *>::rpo_iterator
    rpo_iterator;
  for (rpo_iterator I = RPOT.begin(), E = RPOT.end(); I != E; ++I) {
    BlockChain *Chain = BlockToChain[*I];
    assert(Chain);
    if(!VisitedChains.insert(Chain))
      continue;
    for (BlockChain::iterator BI = Chain->begin(), BE = Chain->end(); BI != BE;
         ++BI) {
      DEBUG(dbgs() << (BI == Chain->begin() ? "Placing chain "
                                            : "          ... ")
                   << getBlockName(*BI) << "\n");
      if (InsertPos != MachineFunction::iterator(*BI))
        F.splice(InsertPos, *BI);
      else
        ++InsertPos;
    }
  }

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
  placeChainsTopologically(F);
  AlignLoops(F);

  BlockToChain.clear();

  // We always return true as we have no way to track whether the final order
  // differs from the original order.
  return true;
}
