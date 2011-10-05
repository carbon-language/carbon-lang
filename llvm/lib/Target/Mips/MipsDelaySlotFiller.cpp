//===-- DelaySlotFiller.cpp - Mips delay slot filler ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Simple pass to fills delay slots with useful instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "delay-slot-filler"

#include "Mips.h"
#include "MipsTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"

using namespace llvm;

STATISTIC(FilledSlots, "Number of delay slots filled");

static cl::opt<bool> EnableDelaySlotFiller(
  "enable-mips-delay-filler",
  cl::init(false),
  cl::desc("Fill the Mips delay slots useful instructions."),
  cl::Hidden);

namespace {
  struct Filler : public MachineFunctionPass {

    TargetMachine &TM;
    const TargetInstrInfo *TII;

    static char ID;
    Filler(TargetMachine &tm)
      : MachineFunctionPass(ID), TM(tm), TII(tm.getInstrInfo()) { }

    virtual const char *getPassName() const {
      return "Mips Delay Slot Filler";
    }

    bool runOnMachineBasicBlock(MachineBasicBlock &MBB);
    bool runOnMachineFunction(MachineFunction &F) {
      bool Changed = false;
      for (MachineFunction::iterator FI = F.begin(), FE = F.end();
           FI != FE; ++FI)
        Changed |= runOnMachineBasicBlock(*FI);
      return Changed;
    }

    bool isDelayFiller(MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator candidate);

    void insertCallUses(MachineBasicBlock::iterator MI,
                        SmallSet<unsigned, 32>& RegDefs,
                        SmallSet<unsigned, 32>& RegUses);

    void insertDefsUses(MachineBasicBlock::iterator MI,
                        SmallSet<unsigned, 32>& RegDefs,
                        SmallSet<unsigned, 32>& RegUses);

    bool IsRegInSet(SmallSet<unsigned, 32>& RegSet,
                    unsigned Reg);

    bool delayHasHazard(MachineBasicBlock::iterator candidate,
                        bool &sawLoad, bool &sawStore,
                        SmallSet<unsigned, 32> &RegDefs,
                        SmallSet<unsigned, 32> &RegUses);

    MachineBasicBlock::iterator
    findDelayInstr(MachineBasicBlock &MBB, MachineBasicBlock::iterator slot);


  };
  char Filler::ID = 0;
} // end of anonymous namespace

/// runOnMachineBasicBlock - Fill in delay slots for the given basic block.
/// We assume there is only one delay slot per delayed instruction.
bool Filler::
runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I)
    if (I->getDesc().hasDelaySlot()) {
      MachineBasicBlock::iterator D = MBB.end();
      MachineBasicBlock::iterator J = I;

      if (EnableDelaySlotFiller)
        D = findDelayInstr(MBB, I);

      ++FilledSlots;
      Changed = true;

      if (D == MBB.end())
        BuildMI(MBB, ++J, I->getDebugLoc(), TII->get(Mips::NOP));
      else
        MBB.splice(++J, &MBB, D);
     }
  return Changed;

}

/// createMipsDelaySlotFillerPass - Returns a pass that fills in delay
/// slots in Mips MachineFunctions
FunctionPass *llvm::createMipsDelaySlotFillerPass(MipsTargetMachine &tm) {
  return new Filler(tm);
}

MachineBasicBlock::iterator
Filler::findDelayInstr(MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator slot) {
  SmallSet<unsigned, 32> RegDefs;
  SmallSet<unsigned, 32> RegUses;
  bool sawLoad = false;
  bool sawStore = false;

  MachineBasicBlock::iterator I = slot;

  // Call's delay filler can def some of call's uses.
  if (slot->getDesc().isCall())
    insertCallUses(slot, RegDefs, RegUses);
  else
    insertDefsUses(slot, RegDefs, RegUses);

  bool done = false;

  while (!done) {
    done = (I == MBB.begin());

    if (!done)
      --I;

    // skip debug value
    if (I->isDebugValue())
      continue;

    if (I->hasUnmodeledSideEffects()
        || I->isInlineAsm()
        || I->isLabel()
        || isDelayFiller(MBB, I)
        || I->getDesc().isPseudo()
        //
        // Should not allow:
        // ERET, DERET or WAIT, PAUSE. Need to add these to instruction
        // list. TBD.
        )
      break;

    if (delayHasHazard(I, sawLoad, sawStore, RegDefs, RegUses)) {
      insertDefsUses(I, RegDefs, RegUses);
      continue;
    }

    return I;
  }
  return MBB.end();
}

bool Filler::delayHasHazard(MachineBasicBlock::iterator candidate,
                            bool &sawLoad,
                            bool &sawStore,
                            SmallSet<unsigned, 32> &RegDefs,
                            SmallSet<unsigned, 32> &RegUses) {
  if (candidate->isImplicitDef() || candidate->isKill())
    return true;

  // Loads or stores cannot be moved past a store to the delay slot
  // and stores cannot be moved past a load. 
  if (candidate->getDesc().mayLoad()) {
    if (sawStore)
      return true;
    sawLoad = true;
  }

  if (candidate->getDesc().mayStore()) {
    if (sawStore)
      return true;
    sawStore = true;
    if (sawLoad)
      return true;
  }

  for (unsigned i = 0, e = candidate->getNumOperands(); i!= e; ++i) {
    const MachineOperand &MO = candidate->getOperand(i);
    if (!MO.isReg())
      continue; // skip

    unsigned Reg = MO.getReg();

    if (MO.isDef()) {
      // check whether Reg is defined or used before delay slot.
      if (IsRegInSet(RegDefs, Reg) || IsRegInSet(RegUses, Reg))
        return true;
    }
    if (MO.isUse()) {
      // check whether Reg is defined before delay slot.
      if (IsRegInSet(RegDefs, Reg))
        return true;
    }
  }
  return false;
}

void Filler::insertCallUses(MachineBasicBlock::iterator MI,
                            SmallSet<unsigned, 32>& RegDefs,
                            SmallSet<unsigned, 32>& RegUses) {
  switch(MI->getOpcode()) {
  default: llvm_unreachable("Unknown opcode.");
  case Mips::JAL:
    RegDefs.insert(31);
    break;
  case Mips::JALR:
    assert(MI->getNumOperands() >= 1);
    const MachineOperand &Reg = MI->getOperand(0);
    assert(Reg.isReg() && "JALR first operand is not a register.");
    RegUses.insert(Reg.getReg());
    RegDefs.insert(31);
    break;
  }
}

// Insert Defs and Uses of MI into the sets RegDefs and RegUses.
void Filler::insertDefsUses(MachineBasicBlock::iterator MI,
                            SmallSet<unsigned, 32>& RegDefs,
                            SmallSet<unsigned, 32>& RegUses) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;

    unsigned Reg = MO.getReg();
    if (Reg == 0)
      continue;
    if (MO.isDef())
      RegDefs.insert(Reg);
    if (MO.isUse())
      RegUses.insert(Reg);
  }
}

//returns true if the Reg or its alias is in the RegSet.
bool Filler::IsRegInSet(SmallSet<unsigned, 32>& RegSet, unsigned Reg) {
  if (RegSet.count(Reg))
    return true;
  // check Aliased Registers
  for (const unsigned *Alias = TM.getRegisterInfo()->getAliasSet(Reg);
       *Alias; ++Alias)
    if (RegSet.count(*Alias))
      return true;

  return false;
}

// return true if the candidate is a delay filler.
bool Filler::isDelayFiller(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator candidate) {
  if (candidate == MBB.begin())
    return false;
  const MCInstrDesc &prevdesc = (--candidate)->getDesc();
  return prevdesc.hasDelaySlot();
}
