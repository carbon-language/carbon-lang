//===-- ARM64DeadRegisterDefinitions.cpp - Replace dead defs w/ zero reg --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// When allowed by the instruction, replace a dead definition of a GPR with
// the zero register. This makes the code a bit friendlier towards the
// hardware's register renamer.
//===----------------------------------------------------------------------===//

#include "ARM64.h"
#include "ARM64RegisterInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "arm64-dead-defs"

STATISTIC(NumDeadDefsReplaced, "Number of dead definitions replaced");

namespace {
class ARM64DeadRegisterDefinitions : public MachineFunctionPass {
private:
  const TargetRegisterInfo *TRI;
  bool implicitlyDefinesOverlappingReg(unsigned Reg, const MachineInstr &MI);
  bool processMachineBasicBlock(MachineBasicBlock &MBB);
  bool usesFrameIndex(const MachineInstr &MI);
public:
  static char ID; // Pass identification, replacement for typeid.
  explicit ARM64DeadRegisterDefinitions() : MachineFunctionPass(ID) {}

  virtual bool runOnMachineFunction(MachineFunction &F);

  const char *getPassName() const { return "Dead register definitions"; }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
char ARM64DeadRegisterDefinitions::ID = 0;
} // end anonymous namespace

bool ARM64DeadRegisterDefinitions::implicitlyDefinesOverlappingReg(
    unsigned Reg, const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.implicit_operands())
    if (MO.isReg() && MO.isDef())
      if (TRI->regsOverlap(Reg, MO.getReg()))
        return true;
  return false;
}

bool ARM64DeadRegisterDefinitions::usesFrameIndex(const MachineInstr &MI) {
  for (const MachineOperand &Op : MI.uses())
    if (Op.isFI())
      return true;
  return false;
}

bool
ARM64DeadRegisterDefinitions::processMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineInstr &MI : MBB) {
    if (usesFrameIndex(MI)) {
      // We need to skip this instruction because while it appears to have a
      // dead def it uses a frame index which might expand into a multi
      // instruction sequence during EPI.
      DEBUG(dbgs() << "    Ignoring, operand is frame index\n");
      continue;
    }
    for (int i = 0, e = MI.getDesc().getNumDefs(); i != e; ++i) {
      MachineOperand &MO = MI.getOperand(i);
      if (MO.isReg() && MO.isDead() && MO.isDef()) {
        assert(!MO.isImplicit() && "Unexpected implicit def!");
        DEBUG(dbgs() << "  Dead def operand #" << i << " in:\n    ";
              MI.print(dbgs()));
        // Be careful not to change the register if it's a tied operand.
        if (MI.isRegTiedToUseOperand(i)) {
          DEBUG(dbgs() << "    Ignoring, def is tied operand.\n");
          continue;
        }
        // Don't change the register if there's an implicit def of a subreg or
        // supperreg.
        if (implicitlyDefinesOverlappingReg(MO.getReg(), MI)) {
          DEBUG(dbgs() << "    Ignoring, implicitly defines overlap reg.\n");
          continue;
        }
        // Make sure the instruction take a register class that contains
        // the zero register and replace it if so.
        unsigned NewReg;
        switch (MI.getDesc().OpInfo[i].RegClass) {
        default:
          DEBUG(dbgs() << "    Ignoring, register is not a GPR.\n");
          continue;
        case ARM64::GPR32RegClassID:
          NewReg = ARM64::WZR;
          break;
        case ARM64::GPR64RegClassID:
          NewReg = ARM64::XZR;
          break;
        }
        DEBUG(dbgs() << "    Replacing with zero register. New:\n      ");
        MO.setReg(NewReg);
        DEBUG(MI.print(dbgs()));
        ++NumDeadDefsReplaced;
      }
    }
  }
  return Changed;
}

// Scan the function for instructions that have a dead definition of a
// register. Replace that register with the zero register when possible.
bool ARM64DeadRegisterDefinitions::runOnMachineFunction(MachineFunction &MF) {
  TRI = MF.getTarget().getRegisterInfo();
  bool Changed = false;
  DEBUG(dbgs() << "***** ARM64DeadRegisterDefinitions *****\n");

  for (auto &MBB : MF)
    if (processMachineBasicBlock(MBB))
      Changed = true;
  return Changed;
}

FunctionPass *llvm::createARM64DeadRegisterDefinitions() {
  return new ARM64DeadRegisterDefinitions();
}
