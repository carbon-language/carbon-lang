//===-- TailDuplication.cpp - Duplicate blocks into predecessors' tails ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass duplicates basic blocks ending in unconditional branches into
// the tails of their predecessors.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "tailduplication"
#include "llvm/Function.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumTailDups  , "Number of tail duplicated blocks");
STATISTIC(NumInstrDups , "Additional instructions due to tail duplication");
STATISTIC(NumDeadBlocks, "Number of dead blocks removed");

// Heuristic for tail duplication.
static cl::opt<unsigned>
TailDuplicateSize("tail-dup-size",
                  cl::desc("Maximum instructions to consider tail duplicating"),
                  cl::init(2), cl::Hidden);

namespace {
  /// TailDuplicatePass - Perform tail duplication.
  class TailDuplicatePass : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    MachineModuleInfo *MMI;

  public:
    static char ID;
    explicit TailDuplicatePass() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);
    virtual const char *getPassName() const { return "Tail Duplication"; }

  private:
    bool TailDuplicateBlocks(MachineFunction &MF);
    bool TailDuplicate(MachineBasicBlock *TailBB, MachineFunction &MF);
    void RemoveDeadBlock(MachineBasicBlock *MBB);
  };

  char TailDuplicatePass::ID = 0;
}

FunctionPass *llvm::createTailDuplicatePass() {
  return new TailDuplicatePass();
}

bool TailDuplicatePass::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();
  MMI = getAnalysisIfAvailable<MachineModuleInfo>();

  bool MadeChange = false;
  bool MadeChangeThisIteration = true;
  while (MadeChangeThisIteration) {
    MadeChangeThisIteration = false;
    MadeChangeThisIteration |= TailDuplicateBlocks(MF);
    MadeChange |= MadeChangeThisIteration;
  }

  return MadeChange;
}

/// TailDuplicateBlocks - Look for small blocks that are unconditionally
/// branched to and do not fall through. Tail-duplicate their instructions
/// into their predecessors to eliminate (dynamic) branches.
bool TailDuplicatePass::TailDuplicateBlocks(MachineFunction &MF) {
  bool MadeChange = false;

  for (MachineFunction::iterator I = ++MF.begin(), E = MF.end(); I != E; ) {
    MachineBasicBlock *MBB = I++;

    // Only duplicate blocks that end with unconditional branches.
    if (MBB->canFallThrough())
      continue;

    MadeChange |= TailDuplicate(MBB, MF);

    // If it is dead, remove it.
    if (MBB->pred_empty()) {
      NumInstrDups -= MBB->size();
      RemoveDeadBlock(MBB);
      MadeChange = true;
      ++NumDeadBlocks;
    }
  }
  return MadeChange;
}

/// TailDuplicate - If it is profitable, duplicate TailBB's contents in each
/// of its predecessors.
bool TailDuplicatePass::TailDuplicate(MachineBasicBlock *TailBB,
                                        MachineFunction &MF) {
  // Don't try to tail-duplicate single-block loops.
  if (TailBB->isSuccessor(TailBB))
    return false;

  // Set the limit on the number of instructions to duplicate, with a default
  // of one less than the tail-merge threshold. When optimizing for size,
  // duplicate only one, because one branch instruction can be eliminated to
  // compensate for the duplication.
  unsigned MaxDuplicateCount;
  if (!TailBB->empty() && TailBB->back().getDesc().isIndirectBranch())
    // If the target has hardware branch prediction that can handle indirect
    // branches, duplicating them can often make them predictable when there
    // are common paths through the code.  The limit needs to be high enough
    // to allow undoing the effects of tail merging.
    MaxDuplicateCount = 20;
  else if (MF.getFunction()->hasFnAttr(Attribute::OptimizeForSize))
    MaxDuplicateCount = 1;
  else
    MaxDuplicateCount = TailDuplicateSize;

  // Check the instructions in the block to determine whether tail-duplication
  // is invalid or unlikely to be profitable.
  unsigned i = 0;
  bool HasCall = false;
  for (MachineBasicBlock::iterator I = TailBB->begin();
       I != TailBB->end(); ++I, ++i) {
    // Non-duplicable things shouldn't be tail-duplicated.
    if (I->getDesc().isNotDuplicable()) return false;
    // Don't duplicate more than the threshold.
    if (i == MaxDuplicateCount) return false;
    // Remember if we saw a call.
    if (I->getDesc().isCall()) HasCall = true;
  }
  // Heuristically, don't tail-duplicate calls if it would expand code size,
  // as it's less likely to be worth the extra cost.
  if (i > 1 && HasCall)
    return false;

  // Iterate through all the unique predecessors and tail-duplicate this
  // block into them, if possible. Copying the list ahead of time also
  // avoids trouble with the predecessor list reallocating.
  bool Changed = false;
  SmallSetVector<MachineBasicBlock *, 8> Preds(TailBB->pred_begin(),
                                               TailBB->pred_end());
  for (SmallSetVector<MachineBasicBlock *, 8>::iterator PI = Preds.begin(),
       PE = Preds.end(); PI != PE; ++PI) {
    MachineBasicBlock *PredBB = *PI;

    assert(TailBB != PredBB &&
           "Single-block loop should have been rejected earlier!");
    if (PredBB->succ_size() > 1) continue;

    MachineBasicBlock *PredTBB, *PredFBB;
    SmallVector<MachineOperand, 4> PredCond;
    if (TII->AnalyzeBranch(*PredBB, PredTBB, PredFBB, PredCond, true))
      continue;
    if (!PredCond.empty())
      continue;
    // EH edges are ignored by AnalyzeBranch.
    if (PredBB->succ_size() != 1)
      continue;
    // Don't duplicate into a fall-through predecessor (at least for now).
    if (PredBB->isLayoutSuccessor(TailBB) && PredBB->canFallThrough())
      continue;

    DEBUG(errs() << "\nTail-duplicating into PredBB: " << *PredBB
                 << "From Succ: " << *TailBB);

    // Remove PredBB's unconditional branch.
    TII->RemoveBranch(*PredBB);
    // Clone the contents of TailBB into PredBB.
    for (MachineBasicBlock::iterator I = TailBB->begin(), E = TailBB->end();
         I != E; ++I) {
      MachineInstr *NewMI = MF.CloneMachineInstr(I);
      PredBB->insert(PredBB->end(), NewMI);
    }
    NumInstrDups += TailBB->size() - 1; // subtract one for removed branch

    // Update the CFG.
    PredBB->removeSuccessor(PredBB->succ_begin());
    assert(PredBB->succ_empty() &&
           "TailDuplicate called on block with multiple successors!");
    for (MachineBasicBlock::succ_iterator I = TailBB->succ_begin(),
         E = TailBB->succ_end(); I != E; ++I)
       PredBB->addSuccessor(*I);

    Changed = true;
    ++NumTailDups;
  }

  // If TailBB was duplicated into all its predecessors except for the prior
  // block, which falls through unconditionally, move the contents of this
  // block into the prior block.
  MachineBasicBlock &PrevBB = *prior(MachineFunction::iterator(TailBB));
  MachineBasicBlock *PriorTBB = 0, *PriorFBB = 0;
  SmallVector<MachineOperand, 4> PriorCond;
  bool PriorUnAnalyzable =
    TII->AnalyzeBranch(PrevBB, PriorTBB, PriorFBB, PriorCond, true);
  // This has to check PrevBB->succ_size() because EH edges are ignored by
  // AnalyzeBranch.
  if (!PriorUnAnalyzable && PriorCond.empty() && !PriorTBB &&
      TailBB->pred_size() == 1 && PrevBB.succ_size() == 1 &&
      !TailBB->hasAddressTaken()) {
    DEBUG(errs() << "\nMerging into block: " << PrevBB
          << "From MBB: " << *TailBB);
    PrevBB.splice(PrevBB.end(), TailBB, TailBB->begin(), TailBB->end());
    PrevBB.removeSuccessor(PrevBB.succ_begin());;
    assert(PrevBB.succ_empty());
    PrevBB.transferSuccessors(TailBB);
    Changed = true;
  }

  return Changed;
}

/// RemoveDeadBlock - Remove the specified dead machine basic block from the
/// function, updating the CFG.
void TailDuplicatePass::RemoveDeadBlock(MachineBasicBlock *MBB) {
  assert(MBB->pred_empty() && "MBB must be dead!");
  DEBUG(errs() << "\nRemoving MBB: " << *MBB);

  // Remove all successors.
  while (!MBB->succ_empty())
    MBB->removeSuccessor(MBB->succ_end()-1);

  // If there are any labels in the basic block, unregister them from
  // MachineModuleInfo.
  if (MMI && !MBB->empty()) {
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         I != E; ++I) {
      if (I->isLabel())
        // The label ID # is always operand #0, an immediate.
        MMI->InvalidateLabel(I->getOperand(0).getImm());
    }
  }

  // Remove the block.
  MBB->eraseFromParent();
}

