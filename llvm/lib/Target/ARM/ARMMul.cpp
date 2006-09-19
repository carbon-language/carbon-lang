//===-- ARMTargetMachine.cpp - Define TargetMachine for ARM ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//


#include "ARM.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
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

      if (MI->getOpcode() == ARM::MUL) {
	MachineOperand &RdOp = MI->getOperand(0);
	MachineOperand &RmOp = MI->getOperand(1);
	MachineOperand &RsOp = MI->getOperand(2);

	unsigned Rd = RdOp.getReg();
	unsigned Rm = RmOp.getReg();
	unsigned Rs = RsOp.getReg();

	if(Rd == Rm) {
	  Changed = true;
	  if (Rd != Rs) {
	    RmOp.setReg(Rs);
	    RsOp.setReg(Rm);
	  } else {
	    BuildMI(MBB, I, ARM::MOV, 3, ARM::R12).addReg(Rm).addImm(0)
	      .addImm(ARMShift::LSL);
	    RmOp.setReg(ARM::R12);
	  }
	}
      }
    }
  }

  return Changed;
}
