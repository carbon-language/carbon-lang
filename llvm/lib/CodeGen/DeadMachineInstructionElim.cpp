//===- DeadMachineInstructionElim.cpp - Remove dead machine instructions --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is an extremely simple MachineInstr-level dead-code-elimination pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN DeadMachineInstructionElim : 
        public MachineFunctionPass {
    virtual bool runOnMachineFunction(MachineFunction &MF);
    
  public:
    static char ID; // Pass identification, replacement for typeid
    DeadMachineInstructionElim() : MachineFunctionPass(&ID) {}
  };
}
char DeadMachineInstructionElim::ID = 0;

static RegisterPass<DeadMachineInstructionElim>
Y("dead-mi-elimination",
  "Remove dead machine instructions");

FunctionPass *llvm::createDeadMachineInstructionElimPass() {
  return new DeadMachineInstructionElim();
}

bool DeadMachineInstructionElim::runOnMachineFunction(MachineFunction &MF) {
  bool AnyChanges = false;
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();
  bool SawStore = true;

  // Loop over all instructions in all blocks, from bottom to top, so that it's
  // more likely that chains of dependent but ultimately dead instructions will
  // be cleaned up.
  for (MachineFunction::reverse_iterator I = MF.rbegin(), E = MF.rend();
       I != E; ++I) {
    MachineBasicBlock *MBB = &*I;
    for (MachineBasicBlock::reverse_iterator MII = MBB->rbegin(),
         MIE = MBB->rend(); MII != MIE; ) {
      MachineInstr *MI = &*MII;

      // Don't delete instructions with side effects.
      if (MI->isSafeToMove(&TII, SawStore)) {
        // Examine each operand.
        bool AllDefsDead = true;
        for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
          const MachineOperand &MO = MI->getOperand(i);
          if (MO.isRegister() && MO.isDef()) {
            unsigned Reg = MO.getReg();
            if (TargetRegisterInfo::isPhysicalRegister(Reg) ||
                !MRI.use_empty(Reg)) {
              // This def has a use. Don't delete the instruction!
              AllDefsDead = false;
              break;
            }
          }
        }

        // If there are no defs with uses, the instruction is dead.
        if (AllDefsDead) {
          // Clear out the operands to take the registers out of their
          // use chains.
          while (unsigned Num = MI->getNumOperands())
            MI->RemoveOperand(Num-1);

          // Delete the actual instruction.
          AnyChanges = true;
          MI->eraseFromParent();
          MIE = MBB->rend();
          // MII is now pointing to the next instruction to process,
          // so don't increment it.
          continue;
        }
      }
      // We didn't delete the current instruction, so increment MII to
      // the next one.
      ++MII;
    }
  }

  return AnyChanges;
}
