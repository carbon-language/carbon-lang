//===-- FuncletLayout.cpp - Contiguously lay out funclets -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements basic block placement transformations which result in
// funclets being contiguous.
//
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

#define DEBUG_TYPE "funclet-layout"

namespace {
class FuncletLayout : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  FuncletLayout() : MachineFunctionPass(ID) {
    initializeFuncletLayoutPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &F) override;
};
}

static void
collectFuncletMembers(DenseMap<MachineBasicBlock *, int> &FuncletMembership,
                      int Funclet, MachineBasicBlock *MBB) {
  // Don't revisit blocks.
  if (FuncletMembership.count(MBB) > 0) {
    // FIXME: This is a hack, we need to assert this unconditionally.
    bool IsProbablyUnreachableBlock =
        MBB->empty() ||
        (MBB->succ_empty() && !MBB->getFirstTerminator()->isReturn() &&
         MBB->size() == 1);

    if (!IsProbablyUnreachableBlock) {
      if (FuncletMembership[MBB] != Funclet) {
        assert(false && "MBB is part of two funclets!");
        report_fatal_error("MBB is part of two funclets!");
      }
    }
    return;
  }

  // Add this MBB to our funclet.
  FuncletMembership[MBB] = Funclet;

  bool IsReturn = false;
  int NumTerminators = 0;
  for (MachineInstr &MI : MBB->terminators()) {
    IsReturn |= MI.isReturn();
    ++NumTerminators;
  }
  assert((!IsReturn || NumTerminators == 1) &&
         "Expected only one terminator when a return is present!");

  // Returns are boundaries where funclet transfer can occur, don't follow
  // successors.
  if (IsReturn)
    return;

  for (MachineBasicBlock *SMBB : MBB->successors())
    if (!SMBB->isEHPad())
      collectFuncletMembers(FuncletMembership, Funclet, SMBB);
}

char FuncletLayout::ID = 0;
char &llvm::FuncletLayoutID = FuncletLayout::ID;
INITIALIZE_PASS(FuncletLayout, "funclet-layout",
                "Contiguously Lay Out Funclets", false, false)

bool FuncletLayout::runOnMachineFunction(MachineFunction &F) {
  // We don't have anything to do if there aren't any EH pads.
  if (!F.getMMI().hasEHFunclets())
    return false;

  const TargetInstrInfo *TII = F.getSubtarget().getInstrInfo();
  SmallVector<MachineBasicBlock *, 16> FuncletBlocks;
  SmallVector<std::pair<MachineBasicBlock *, int>, 16> CatchRetSuccessors;
  for (MachineBasicBlock &MBB : F) {
    if (MBB.isEHFuncletEntry())
      FuncletBlocks.push_back(&MBB);

    MachineBasicBlock::iterator MBBI = MBB.getFirstTerminator();
    if (MBBI->getOpcode() != TII->getCatchReturnOpcode())
      continue;

    MachineBasicBlock *Successor = MBBI->getOperand(0).getMBB();
    MachineBasicBlock *SuccessorColor = MBBI->getOperand(1).getMBB();
    CatchRetSuccessors.push_back({Successor, SuccessorColor->getNumber()});
  }

  // We don't have anything to do if there aren't any EH pads.
  if (FuncletBlocks.empty())
    return false;

  DenseMap<MachineBasicBlock *, int> FuncletMembership;
  // Identify all the basic blocks reachable from the function entry.
  collectFuncletMembers(FuncletMembership, F.front().getNumber(), F.begin());
  // Next, identify all the blocks inside the funclets.
  for (MachineBasicBlock *MBB : FuncletBlocks)
    collectFuncletMembers(FuncletMembership, MBB->getNumber(), MBB);
  // Finally, identify all the targets of a catchret.
  for (std::pair<MachineBasicBlock *, int> CatchRetPair : CatchRetSuccessors)
    collectFuncletMembers(FuncletMembership, CatchRetPair.second,
                          CatchRetPair.first);

  F.sort([&](MachineBasicBlock &x, MachineBasicBlock &y) {
    return FuncletMembership[&x] < FuncletMembership[&y];
  });

  // Conservatively assume we changed something.
  return true;
}
