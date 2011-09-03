//===-- X86VZeroUpper.cpp - AVX vzeroupper instruction inserter -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass which inserts x86 AVX vzeroupper instructions
// before calls to SSE encoded functions. This avoids transition latency
// penalty when tranfering control between AVX encoded instructions and old
// SSE encoding mode.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "x86-codegen"
#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/GlobalValue.h"
#include "llvm/Target/TargetInstrInfo.h"
using namespace llvm;

STATISTIC(NumVZU, "Number of vzeroupper instructions inserted");

namespace {
  struct VZeroUpperInserter : public MachineFunctionPass {
    static char ID;
    VZeroUpperInserter() : MachineFunctionPass(ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    bool processBasicBlock(MachineFunction &MF, MachineBasicBlock &MBB);

    virtual const char *getPassName() const { return "X86 vzeroupper inserter";}

  private:
    const TargetInstrInfo *TII; // Machine instruction info.
    MachineBasicBlock *MBB;     // Current basic block
  };
  char VZeroUpperInserter::ID = 0;
}

FunctionPass *llvm::createX86IssueVZeroUpperPass() {
  return new VZeroUpperInserter();
}

/// runOnMachineFunction - Loop over all of the basic blocks, inserting
/// vzero upper instructions before function calls.
bool VZeroUpperInserter::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();
  bool Changed = false;

  // Process any unreachable blocks in arbitrary order now.
  for (MachineFunction::iterator BB = MF.begin(), E = MF.end(); BB != E; ++BB)
    Changed |= processBasicBlock(MF, *BB);

  return Changed;
}

static bool isCallToModuleFn(const MachineInstr *MI) {
  assert(MI->getDesc().isCall() && "Isn't a call instruction");

  for (int i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);

    if (!MO.isGlobal())
      continue;

    const GlobalValue *GV = MO.getGlobal();
    GlobalValue::LinkageTypes LT = GV->getLinkage();
    if (GV->isInternalLinkage(LT) || GV->isPrivateLinkage(LT) ||
        (GV->isExternalLinkage(LT) && !GV->isDeclaration()))
      return true;

    return false;
  }
  return false;
}

/// processBasicBlock - Loop over all of the instructions in the basic block,
/// inserting vzero upper instructions before function calls.
bool VZeroUpperInserter::processBasicBlock(MachineFunction &MF,
                                           MachineBasicBlock &BB) {
  bool Changed = false;
  MBB = &BB;

  for (MachineBasicBlock::iterator I = BB.begin(); I != BB.end(); ++I) {
    MachineInstr *MI = I;
    DebugLoc dl = I->getDebugLoc();

    // Insert a vzeroupper instruction before each control transfer
    // to functions outside this module
    if (MI->getDesc().isCall() && !isCallToModuleFn(MI)) {
      BuildMI(*MBB, I, dl, TII->get(X86::VZEROUPPER));
      ++NumVZU;
    }
  }

  return Changed;
}
