//===-- DelaySlotFiller.cpp - SparcV8 delay slot filler -------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Simple local delay slot filler for SparcV8 machine code
//
//===----------------------------------------------------------------------===//

#include "SparcV8.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "Support/Statistic.h"

using namespace llvm;

namespace {
  Statistic<> FilledSlots ("delayslotfiller", "Num. of delay slots filled");

  struct Filler : public MachineFunctionPass {
    /// Target machine description which we query for reg. names, data
    /// layout, etc.
    ///
    TargetMachine &TM;

    Filler (TargetMachine &tm) : TM (tm) { }

    virtual const char *getPassName () const {
      return "SparcV8 Delay Slot Filler";
    }

    bool runOnMachineBasicBlock (MachineBasicBlock &MBB);
    bool runOnMachineFunction (MachineFunction &F) {
      bool Changed = false;
      for (MachineFunction::iterator FI = F.begin (), FE = F.end ();
           FI != FE; ++FI)
        Changed |= runOnMachineBasicBlock (*FI);
      return Changed;
    }

  };
} // end of anonymous namespace

/// createSparcV8DelaySlotFillerPass - Returns a pass that fills in delay
/// slots in SparcV8 MachineFunctions
///
FunctionPass *llvm::createSparcV8DelaySlotFillerPass (TargetMachine &tm) {
  return new Filler (tm);
}

static bool hasDelaySlot (unsigned Opcode) {
  switch (Opcode) {
    case V8::BA:
    case V8::BCC:
    case V8::BCS:
    case V8::BE:
    case V8::BG:
    case V8::BGE:
    case V8::BGU:
    case V8::BL:
    case V8::BLE:
    case V8::BLEU:
    case V8::BNE:
    case V8::CALL:
    case V8::RETL:
      return true;
    default:
      return false;
  }
}

/// runOnMachineBasicBlock - Fill in delay slots for the given basic block.
/// Currently, we fill delay slots with NOPs. We assume there is only one
/// delay slot per delayed instruction.
///
bool Filler::runOnMachineBasicBlock (MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineBasicBlock::iterator I = MBB.begin (); I != MBB.end (); ++I)
    if (hasDelaySlot (I->getOpcode ())) {
      MachineBasicBlock::iterator J = I;
      ++J;
      BuildMI (MBB, J, V8::NOP, 0);
      ++FilledSlots;
      Changed = true;
    }
  return Changed;
}
