//===-- ARMMul.cpp - Define TargetMachine for A5CRM -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Modify the ARM multiplication instructions so that Rd{Hi,Lo} and Rm are distinct
//
//===----------------------------------------------------------------------===//


#include "ARM.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN FixMul : public MachineFunctionPass {
    virtual bool runOnMachineFunction(MachineFunction &MF);
  };
}

FunctionPass *llvm::createARMFixMulPass() { return new FixMul(); }

bool FixMul::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  for (MachineFunction::iterator BB = MF.begin(), E = MF.end();
       BB != E; ++BB) {
    MachineBasicBlock &MBB = *BB;

    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
         I != E; ++I) {
      MachineInstr *MI = I;

      int Op = MI->getOpcode();
      if (Op == ARM::MUL ||
          Op == ARM::SMULL ||
          Op == ARM::UMULL) {
        MachineOperand &RdOp = MI->getOperand(0);
        MachineOperand &RmOp = MI->getOperand(1);
        MachineOperand &RsOp = MI->getOperand(2);

        unsigned Rd = RdOp.getReg();
        unsigned Rm = RmOp.getReg();
        unsigned Rs = RsOp.getReg();

        if (Rd == Rm) {
          Changed = true;
          if (Rd != Rs) {
	    //Rd and Rm must be distinct, but Rd can be equal to Rs.
	    //Swap Rs and Rm
            RmOp.setReg(Rs);
            RsOp.setReg(Rm);
          } else {
            unsigned scratch = Op == ARM::MUL ? ARM::R12 : ARM::R0;
            BuildMI(MBB, I, MF.getTarget().getInstrInfo()->get(ARM::MOV),
                    scratch).addReg(Rm).addImm(0).addImm(ARMShift::LSL);
            RmOp.setReg(scratch);
          }
        }
      }
    }
  }

  return Changed;
}
