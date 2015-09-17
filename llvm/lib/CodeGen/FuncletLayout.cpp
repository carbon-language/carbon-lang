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
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
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
collectFuncletMembers(MapVector<MachineBasicBlock *, int> &FuncletMembership,
                      int Funclet, MachineBasicBlock *MBB) {
  // Don't revisit blocks.
  if (FuncletMembership.count(MBB) > 0)
    return;

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

  SmallVector<MachineBasicBlock *, 16> FuncletBlocks;
  for (MachineBasicBlock &MBB : F)
    if (MBB.isEHFuncletEntry())
      FuncletBlocks.push_back(&MBB);

  // We don't have anything to do if there aren't any EH pads.
  if (FuncletBlocks.empty())
    return false;

  MapVector<MachineBasicBlock *, int> FuncletMembership;
  for (MachineBasicBlock *MBB : FuncletBlocks)
    collectFuncletMembers(FuncletMembership, MBB->getNumber(), MBB);

  for (std::pair<llvm::MachineBasicBlock *, int> &FuncletMember :
       FuncletMembership) {
    // Move this block to the end of the function.
    MachineBasicBlock *MBB = FuncletMember.first;
    MBB->moveAfter(--F.end());
  }

  // Conservatively assume we changed something.
  return true;
}
