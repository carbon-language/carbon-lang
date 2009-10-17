//===-- CodePlacementOpt.cpp - Code Placement pass. -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass that optimize code placement and align loop
// headers to target specific alignment boundary.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "code-placement"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumLoopsAligned,  "Number of loops aligned");
STATISTIC(NumIntraElim,     "Number of intra loop branches eliminated");
STATISTIC(NumIntraMoved,    "Number of intra loop branches moved");

namespace {
  class CodePlacementOpt : public MachineFunctionPass {
    const MachineLoopInfo *MLI;
    const TargetInstrInfo *TII;
    const TargetLowering  *TLI;

  public:
    static char ID;
    CodePlacementOpt() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);
    virtual const char *getPassName() const {
      return "Code Placement Optimizater";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<MachineLoopInfo>();
      AU.addPreservedID(MachineDominatorsID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

  private:
    bool HasFallthrough(MachineBasicBlock *MBB);
    bool HasAnalyzableTerminator(MachineBasicBlock *MBB);
    void Splice(MachineFunction &MF,
                MachineFunction::iterator InsertPt,
                MachineFunction::iterator Begin,
                MachineFunction::iterator End);
    void UpdateTerminator(MachineBasicBlock *MBB);
    bool OptimizeIntraLoopEdges(MachineFunction &MF);
    bool OptimizeIntraLoopEdgesInLoop(MachineFunction &MF, MachineLoop *L);
    bool AlignLoops(MachineFunction &MF);
    bool AlignLoop(MachineFunction &MF, MachineLoop *L, unsigned Align);
  };

  char CodePlacementOpt::ID = 0;
} // end anonymous namespace

FunctionPass *llvm::createCodePlacementOptPass() {
  return new CodePlacementOpt();
}

/// HasFallthrough - Test whether the given branch has a fallthrough, either as
/// a plain fallthrough or as a fallthrough case of a conditional branch.
///
bool CodePlacementOpt::HasFallthrough(MachineBasicBlock *MBB) {
  MachineBasicBlock *TBB = 0, *FBB = 0;
  SmallVector<MachineOperand, 4> Cond;
  if (TII->AnalyzeBranch(*MBB, TBB, FBB, Cond))
    return false;
  // This conditional branch has no fallthrough.
  if (FBB)
    return false;
  // An unconditional branch has no fallthrough.
  if (Cond.empty() && TBB)
    return false;
  // It has a fallthrough.
  return true;
}

/// HasAnalyzableTerminator - Test whether AnalyzeBranch will succeed on MBB.
/// This is called before major changes are begun to test whether it will be
/// possible to complete the changes.
///
/// Target-specific code is hereby encouraged to make AnalyzeBranch succeed
/// whenever possible.
///
bool CodePlacementOpt::HasAnalyzableTerminator(MachineBasicBlock *MBB) {
  // Conservatively ignore EH landing pads.
  if (MBB->isLandingPad()) return false;

  // Ignore blocks which look like they might have EH-related control flow.
  // At the time of this writing, there are blocks which AnalyzeBranch
  // thinks end in single uncoditional branches, yet which have two CFG
  // successors. Code in this file is not prepared to reason about such things.
  if (!MBB->empty() && MBB->back().getOpcode() == TargetInstrInfo::EH_LABEL)
    return false;

  // Aggressively handle return blocks and similar constructs.
  if (MBB->succ_empty()) return true;

  // Ask the target's AnalyzeBranch if it can handle this block.
  MachineBasicBlock *TBB = 0, *FBB = 0;
  SmallVector<MachineOperand, 4> Cond;
  // Make the the terminator is understood.
  if (TII->AnalyzeBranch(*MBB, TBB, FBB, Cond))
    return false;
  // Make sure we have the option of reversing the condition.
  if (!Cond.empty() && TII->ReverseBranchCondition(Cond))
    return false;
  return true;
}

/// Splice - Move the sequence of instructions [Begin,End) to just before
/// InsertPt. Update branch instructions as needed to account for broken
/// fallthrough edges and to take advantage of newly exposed fallthrough
/// opportunities.
///
void CodePlacementOpt::Splice(MachineFunction &MF,
                              MachineFunction::iterator InsertPt,
                              MachineFunction::iterator Begin,
                              MachineFunction::iterator End) {
  assert(Begin != MF.begin() && End != MF.begin() && InsertPt != MF.begin() &&
         "Splice can't change the entry block!");
  MachineFunction::iterator OldBeginPrior = prior(Begin);
  MachineFunction::iterator OldEndPrior = prior(End);

  MF.splice(InsertPt, Begin, End);

  UpdateTerminator(prior(Begin));
  UpdateTerminator(OldBeginPrior);
  UpdateTerminator(OldEndPrior);
}

/// UpdateTerminator - Update the terminator instructions in MBB to account
/// for changes to the layout. If the block previously used a fallthrough,
/// it may now need a branch, and if it previously used branching it may now
/// be able to use a fallthrough.
///
void CodePlacementOpt::UpdateTerminator(MachineBasicBlock *MBB) {
  // A block with no successors has no concerns with fall-through edges.
  if (MBB->succ_empty()) return;

  MachineBasicBlock *TBB = 0, *FBB = 0;
  SmallVector<MachineOperand, 4> Cond;
  bool B = TII->AnalyzeBranch(*MBB, TBB, FBB, Cond);
  (void) B;
  assert(!B && "UpdateTerminators requires analyzable predecessors!");
  if (Cond.empty()) {
    if (TBB) {
      // The block has an unconditional branch. If its successor is now
      // its layout successor, delete the branch.
      if (MBB->isLayoutSuccessor(TBB))
        TII->RemoveBranch(*MBB);
    } else {
      // The block has an unconditional fallthrough. If its successor is not
      // its layout successor, insert a branch.
      TBB = *MBB->succ_begin();
      if (!MBB->isLayoutSuccessor(TBB))
        TII->InsertBranch(*MBB, TBB, 0, Cond);
    }
  } else {
    if (FBB) {
      // The block has a non-fallthrough conditional branch. If one of its
      // successors is its layout successor, rewrite it to a fallthrough
      // conditional branch.
      if (MBB->isLayoutSuccessor(TBB)) {
        TII->RemoveBranch(*MBB);
        TII->ReverseBranchCondition(Cond);
        TII->InsertBranch(*MBB, FBB, 0, Cond);
      } else if (MBB->isLayoutSuccessor(FBB)) {
        TII->RemoveBranch(*MBB);
        TII->InsertBranch(*MBB, TBB, 0, Cond);
      }
    } else {
      // The block has a fallthrough conditional branch.
      MachineBasicBlock *MBBA = *MBB->succ_begin();
      MachineBasicBlock *MBBB = *next(MBB->succ_begin());
      if (MBBA == TBB) std::swap(MBBB, MBBA);
      if (MBB->isLayoutSuccessor(TBB)) {
        TII->RemoveBranch(*MBB);
        TII->ReverseBranchCondition(Cond);
        TII->InsertBranch(*MBB, MBBA, 0, Cond);
      } else if (!MBB->isLayoutSuccessor(MBBA)) {
        TII->RemoveBranch(*MBB);
        TII->InsertBranch(*MBB, TBB, MBBA, Cond);
      }
    }
  }
}

/// OptimizeIntraLoopEdges - Reposition loop blocks to minimize
/// intra-loop branching and to form contiguous loops.
///
bool CodePlacementOpt::OptimizeIntraLoopEdges(MachineFunction &MF) {
  bool Changed = false;

  if (!TLI->shouldOptimizeCodePlacement())
    return Changed;

  for (MachineLoopInfo::iterator I = MLI->begin(), E = MLI->end();
       I != E; ++I)
    Changed |= OptimizeIntraLoopEdgesInLoop(MF, *I);

  return Changed;
}

/// OptimizeIntraLoopEdgesInLoop - Reposition loop blocks to minimize
/// intra-loop branching and to form contiguous loops.
///
/// This code takes the approach of making minor changes to the existing
/// layout to fix specific loop-oriented problems. Also, it depends on
/// AnalyzeBranch, which can't understand complex control instructions.
///
bool CodePlacementOpt::OptimizeIntraLoopEdgesInLoop(MachineFunction &MF,
                                                    MachineLoop *L) {
  bool Changed = false;

  // Do optimization for nested loops.
  for (MachineLoop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    Changed |= OptimizeIntraLoopEdgesInLoop(MF, *I);

  // Keep a record of which blocks are in the portion of the loop contiguous
  // with the loop header.
  SmallPtrSet<MachineBasicBlock *, 8> ContiguousBlocks;
  ContiguousBlocks.insert(L->getHeader());

  // Find the loop "top", ignoring any discontiguous parts.
  MachineBasicBlock *TopMBB = L->getHeader();
  if (TopMBB != MF.begin()) {
    MachineBasicBlock *PriorMBB = prior(MachineFunction::iterator(TopMBB));
    while (L->contains(PriorMBB)) {
      ContiguousBlocks.insert(PriorMBB);
      TopMBB = PriorMBB;
      if (TopMBB == MF.begin()) break;
      PriorMBB = prior(MachineFunction::iterator(TopMBB));
    }
  }

  // Find the loop "bottom", ignoring any discontiguous parts.
  MachineBasicBlock *BotMBB = L->getHeader();
  if (BotMBB != prior(MF.end())) {
    MachineBasicBlock *NextMBB = next(MachineFunction::iterator(BotMBB));
    while (L->contains(NextMBB)) {
      ContiguousBlocks.insert(NextMBB);
      BotMBB = NextMBB;
      if (BotMBB == next(MachineFunction::iterator(BotMBB))) break;
      NextMBB = next(MachineFunction::iterator(BotMBB));
    }
  }

  // First, move blocks which unconditionally jump to the loop top to the
  // top of the loop so that they have a fall through. This can introduce a
  // branch on entry to the loop, but it can eliminate a branch within the
  // loop. See the @simple case in test/CodeGen/X86/loop_blocks.ll for an
  // example of this.

  bool BotHasFallthrough = HasFallthrough(BotMBB);

  if (TopMBB == MF.begin() ||
      HasAnalyzableTerminator(prior(MachineFunction::iterator(TopMBB)))) {
  new_top:
    for (MachineBasicBlock::pred_iterator PI = TopMBB->pred_begin(),
         PE = TopMBB->pred_end(); PI != PE; ++PI) {
      MachineBasicBlock *Pred = *PI;
      if (Pred == TopMBB) continue;
      if (HasFallthrough(Pred)) continue;
      if (!L->contains(Pred)) continue;

      // Verify that we can analyze all the loop entry edges before beginning
      // any changes which will require us to be able to analyze them.
      if (Pred == MF.begin())
        continue;
      if (!HasAnalyzableTerminator(Pred))
        continue;
      if (!HasAnalyzableTerminator(prior(MachineFunction::iterator(Pred))))
        continue;

      // Move the block.
      Changed = true;
      ContiguousBlocks.insert(Pred);

      // Move it and all the blocks that can reach it via fallthrough edges
      // exclusively, to keep existing fallthrough-edges intact.
      MachineFunction::iterator Begin = Pred;
      MachineFunction::iterator End = next(Begin);
      while (Begin != MF.begin()) {
        MachineFunction::iterator Prior = prior(Begin);
        if (Prior == MF.begin())
          break;
        // Stop when a non-fallthrough edge is found.
        if (!HasFallthrough(Prior))
          break;
        // Stop if a block which could fall-through out of the loop is found.
        if (Prior->isSuccessor(End))
          break;
        // If we've reached the top, stop scanning.
        if (Prior == MachineFunction::iterator(TopMBB)) {
          // We know top currently has a fall through (because we just checked
          // it) which would be lost if we do the transformation, so it isn't
          // worthwhile to do the transformation unless it would expose a new
          // fallthrough edge.
          if (!Prior->isSuccessor(End))
            goto next_pred;
          // Otherwise we can stop scanning and procede to move the blocks.
          break;
        }
        // If we hit a switch or something complicated, don't move anything
        // for this predecessor.
        if (!HasAnalyzableTerminator(prior(MachineFunction::iterator(Prior))))
          break;
        Begin = Prior;
        ContiguousBlocks.insert(Begin);
        ++NumIntraMoved;
      }

      // Update BotMBB, before moving Begin/End around and forgetting where
      // the new bottom is.
      if (BotMBB == prior(End))
        BotMBB = prior(Begin);

      // Move the blocks.
      Splice(MF, TopMBB, Begin, End);

      // Update TopMBB, now that all the updates requiring the old top are
      // complete.
      TopMBB = Begin;

      // We have a new loop top. Iterate on it. We shouldn't have to do this
      // too many times if BranchFolding has done a reasonable job.
      goto new_top;
    next_pred:;
    }
  }

  // If the loop previously didn't exit with a fall-through and it now does,
  // we eliminated a branch.
  if (!BotHasFallthrough && HasFallthrough(BotMBB)) {
    ++NumIntraElim;
    BotHasFallthrough = true;
  }

  // Next, move any loop blocks that are not in the portion of the loop
  // contiguous with the header. This makes the loop contiguous, provided that
  // AnalyzeBranch can handle all the relevant branching. See the @cfg_islands
  // case in test/CodeGen/X86/loop_blocks.ll for an example of this.

  // Determine a position to move orphaned loop blocks to. If TopMBB is not
  // entered via fallthrough and BotMBB is exited via fallthrough, prepend them
  // to the top of the loop to avoid loosing that fallthrough. Otherwise append
  // them to the bottom, even if it previously had a fallthrough, on the theory
  // that it's worth an extra branch to keep the loop contiguous.
  MachineFunction::iterator InsertPt = next(MachineFunction::iterator(BotMBB));
  bool InsertAtTop = false;
  if (TopMBB != MF.begin() &&
      !HasFallthrough(prior(MachineFunction::iterator(TopMBB))) &&
      HasFallthrough(BotMBB)) {
    InsertPt = TopMBB;
    InsertAtTop = true;
  }

  // Find non-contigous blocks and fix them.
  if (InsertPt != MF.begin() && HasAnalyzableTerminator(prior(InsertPt)))
    for (MachineLoop::block_iterator BI = L->block_begin(), BE = L->block_end();
         BI != BE; ++BI) {
      MachineBasicBlock *BB = *BI;

      // Verify that we can analyze all the loop entry edges before beginning
      // any changes which will require us to be able to analyze them.
      if (!HasAnalyzableTerminator(BB))
        continue;
      if (!HasAnalyzableTerminator(prior(MachineFunction::iterator(BB))))
        continue;

      // If the layout predecessor is part of the loop, this block will be
      // processed along with it. This keeps them in their relative order.
      if (BB != MF.begin() &&
          L->contains(prior(MachineFunction::iterator(BB))))
        continue;

      // Check to see if this block is already contiguous with the main
      // portion of the loop.
      if (!ContiguousBlocks.insert(BB))
        continue;

      // Move the block.
      Changed = true;

      // Process this block and all loop blocks contiguous with it, to keep
      // them in their relative order.
      MachineFunction::iterator Begin = BB;
      MachineFunction::iterator End = next(MachineFunction::iterator(BB));
      for (; End != MF.end(); ++End) {
        if (!L->contains(End)) break;
        if (!HasAnalyzableTerminator(End)) break;
        ContiguousBlocks.insert(End);
        ++NumIntraMoved;
      }

      // Update BotMBB.
      if (!InsertAtTop)
        BotMBB = prior(End);

      // If we're inserting at the bottom of the loop, and the code we're
      // moving originally had fall-through successors, bring the sucessors
      // up with the loop blocks to preserve the fall-through edges.
      if (!InsertAtTop)
        for (; End != MF.end(); ++End) {
          if (L->contains(End)) break;
          if (!HasAnalyzableTerminator(End)) break;
          if (!HasFallthrough(prior(End))) break;
        }

      // Move the blocks.
      Splice(MF, InsertPt, Begin, End);

      // Update TopMBB.
      if (InsertAtTop)
        TopMBB = Begin;
    }

  return Changed;
}

/// AlignLoops - Align loop headers to target preferred alignments.
///
bool CodePlacementOpt::AlignLoops(MachineFunction &MF) {
  const Function *F = MF.getFunction();
  if (F->hasFnAttr(Attribute::OptimizeForSize))
    return false;

  unsigned Align = TLI->getPrefLoopAlignment();
  if (!Align)
    return false;  // Don't care about loop alignment.

  bool Changed = false;

  for (MachineLoopInfo::iterator I = MLI->begin(), E = MLI->end();
       I != E; ++I)
    Changed |= AlignLoop(MF, *I, Align);

  return Changed;
}

/// AlignLoop - Align loop headers to target preferred alignments.
///
bool CodePlacementOpt::AlignLoop(MachineFunction &MF, MachineLoop *L,
                                 unsigned Align) {
  bool Changed = false;

  // Do alignment for nested loops.
  for (MachineLoop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    Changed |= AlignLoop(MF, *I, Align);

  MachineBasicBlock *TopMBB = L->getHeader();
  if (TopMBB != MF.begin()) {
    MachineBasicBlock *PredMBB = prior(MachineFunction::iterator(TopMBB));
    while (L->contains(PredMBB)) {
      TopMBB = PredMBB;
      if (TopMBB == MF.begin()) break;
      PredMBB = prior(MachineFunction::iterator(TopMBB));
    }
  }

  TopMBB->setAlignment(Align);
  Changed = true;
  ++NumLoopsAligned;

  return Changed;
}

bool CodePlacementOpt::runOnMachineFunction(MachineFunction &MF) {
  MLI = &getAnalysis<MachineLoopInfo>();
  if (MLI->empty())
    return false;  // No loops.

  TLI = MF.getTarget().getTargetLowering();
  TII = MF.getTarget().getInstrInfo();

  bool Changed = OptimizeIntraLoopEdges(MF);

  Changed |= AlignLoops(MF);

  return Changed;
}
