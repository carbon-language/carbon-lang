//===-- DelaySlotFiller.cpp - MBlaze delay slot filler --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A pass that attempts to fill instructions with delay slots. If no
// instructions can be moved into the delay slot then a NOP is placed there.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "delay-slot-filler"

#include "MBlaze.h"
#include "MBlazeTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

STATISTIC(FilledSlots, "Number of delay slots filled");

namespace {
  struct Filler : public MachineFunctionPass {

    TargetMachine &TM;
    const TargetInstrInfo *TII;

    static char ID;
    Filler(TargetMachine &tm) 
      : MachineFunctionPass(ID), TM(tm), TII(tm.getInstrInfo()) { }

    virtual const char *getPassName() const {
      return "MBlaze Delay Slot Filler";
    }

    bool runOnMachineBasicBlock(MachineBasicBlock &MBB);
    bool runOnMachineFunction(MachineFunction &F) {
      bool Changed = false;
      for (MachineFunction::iterator FI = F.begin(), FE = F.end();
           FI != FE; ++FI)
        Changed |= runOnMachineBasicBlock(*FI);
      return Changed;
    }

  };
  char Filler::ID = 0;
} // end of anonymous namespace

static bool hasImmInstruction( MachineBasicBlock::iterator &candidate ) {
    // Any instruction with an immediate mode operand greater than
    // 16-bits requires an implicit IMM instruction.
    unsigned numOper = candidate->getNumOperands();
    for( unsigned op = 0; op < numOper; ++op ) {
        if( candidate->getOperand(op).isImm() &&
            (candidate->getOperand(op).getImm() & 0xFFFFFFFFFFFF0000LL) != 0 )
            return true;

        // FIXME: we could probably check to see if the FP value happens
        //        to not need an IMM instruction. For now we just always
        //        assume that FP values always do.
        if( candidate->getOperand(op).isFPImm() )
            return true;
    }

    return false;
}

static bool delayHasHazard( MachineBasicBlock::iterator &candidate,
                            MachineBasicBlock::iterator &slot ) {

    // Loop over all of the operands in the branch instruction
    // and make sure that none of them are defined by the
    // candidate instruction.
    unsigned numOper = slot->getNumOperands();
    for( unsigned op = 0; op < numOper; ++op ) {
        if( !slot->getOperand(op).isReg() || 
            !slot->getOperand(op).isUse() ||
            slot->getOperand(op).isImplicit() )
            continue;

        unsigned cnumOper = candidate->getNumOperands();
        for( unsigned cop = 0; cop < cnumOper; ++cop ) {
            if( candidate->getOperand(cop).isReg() &&
                candidate->getOperand(cop).isDef() &&
                candidate->getOperand(cop).getReg() == 
                slot->getOperand(op).getReg() )
                return true;
        }
    }

    // There are no hazards between the two instructions
    return false;
}

static bool usedBeforeDelaySlot( MachineBasicBlock::iterator &candidate,
                                 MachineBasicBlock::iterator &slot ) {
  MachineBasicBlock::iterator I = candidate;
  for (++I; I != slot; ++I) {
        unsigned numOper = I->getNumOperands();
        for( unsigned op = 0; op < numOper; ++op ) {
            if( I->getOperand(op).isReg() &&
                I->getOperand(op).isUse() ) {
                unsigned reg = I->getOperand(op).getReg();
                unsigned cops = candidate->getNumOperands();
                for( unsigned cop = 0; cop < cops; ++cop ) {
                    if( candidate->getOperand(cop).isReg() &&
                        candidate->getOperand(cop).isDef() &&
                        candidate->getOperand(cop).getReg() == reg )
                        return true;
                }
            }
        }
  }

  return false;
}

static MachineBasicBlock::iterator
findDelayInstr(MachineBasicBlock &MBB,MachineBasicBlock::iterator &slot) {
  MachineBasicBlock::iterator found = MBB.end();
  for (MachineBasicBlock::iterator I = MBB.begin(); I != slot; ++I) {
      TargetInstrDesc desc = I->getDesc();
      if( desc.hasDelaySlot() || desc.isBranch() || 
          desc.mayLoad() || desc.    mayStore() || 
          hasImmInstruction(I) || delayHasHazard(I,slot) || 
          usedBeforeDelaySlot(I,slot)) continue;

      found = I;
  }

  return found;
}

/// runOnMachineBasicBlock - Fill in delay slots for the given basic block.
/// Currently, we fill delay slots with NOPs. We assume there is only one
/// delay slot per delayed instruction.
bool Filler::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I)
    if (I->getDesc().hasDelaySlot()) {
      MachineBasicBlock::iterator J = I;
      MachineBasicBlock::iterator D = findDelayInstr(MBB,I);

      ++J;
      ++FilledSlots;
      Changed = true;

      if( D == MBB.end() )
        BuildMI(MBB, J, I->getDebugLoc(), TII->get(MBlaze::NOP));
      else
        MBB.splice( J, &MBB, D );
    }
  return Changed;
}

/// createMBlazeDelaySlotFillerPass - Returns a pass that fills in delay
/// slots in MBlaze MachineFunctions
FunctionPass *llvm::createMBlazeDelaySlotFillerPass(MBlazeTargetMachine &tm) {
  return new Filler(tm);
}

