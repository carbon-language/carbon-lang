//===-- NEONPreAllocPass.cpp - Allocate adjacent NEON registers--*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "neon-prealloc"
#include "ARM.h"
#include "ARMInstrInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN NEONPreAllocPass : public MachineFunctionPass {
    const TargetInstrInfo *TII;

  public:
    static char ID;
    NEONPreAllocPass() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "NEON register pre-allocation pass";
    }

  private:
    bool PreAllocNEONRegisters(MachineBasicBlock &MBB);
  };

  char NEONPreAllocPass::ID = 0;
}

static bool isNEONMultiRegOp(int Opcode, unsigned &FirstOpnd,
                             unsigned &NumRegs) {
  switch (Opcode) {
  default:
    break;

  case ARM::VLD2d8:
  case ARM::VLD2d16:
  case ARM::VLD2d32:
  case ARM::VLD2d64:
    FirstOpnd = 0;
    NumRegs = 2;
    return true;

  case ARM::VLD3d8:
  case ARM::VLD3d16:
  case ARM::VLD3d32:
  case ARM::VLD3d64:
    FirstOpnd = 0;
    NumRegs = 3;
    return true;

  case ARM::VLD4d8:
  case ARM::VLD4d16:
  case ARM::VLD4d32:
  case ARM::VLD4d64:
    FirstOpnd = 0;
    NumRegs = 4;
    return true;
  }

  return false;
}

bool NEONPreAllocPass::PreAllocNEONRegisters(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  for (; MBBI != E; ++MBBI) {
    MachineInstr *MI = &*MBBI;
    unsigned FirstOpnd, NumRegs;
    if (!isNEONMultiRegOp(MI->getOpcode(), FirstOpnd, NumRegs))
      continue;

    MachineBasicBlock::iterator NextI = next(MBBI);
    for (unsigned R = 0; R < NumRegs; ++R) {
      MachineOperand &MO = MI->getOperand(FirstOpnd + R);
      assert(MO.isReg() && MO.getSubReg() == 0 && "unexpected operand");
      unsigned VirtReg = MO.getReg();
      assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
             "expected a virtual register");

      // For now, just assign a fixed set of adjacent registers.
      // This leaves plenty of room for future improvements.
      static const unsigned NEONDRegs[] = {
        ARM::D0, ARM::D1, ARM::D2, ARM::D3
      };
      MO.setReg(NEONDRegs[R]);

      if (MO.isUse()) {
        // Insert a copy from VirtReg.
        AddDefaultPred(BuildMI(MBB, MBBI, MI->getDebugLoc(),
                               TII->get(ARM::FCPYD), MO.getReg())
                       .addReg(VirtReg));
        if (MO.isKill()) {
          MachineInstr *CopyMI = prior(MBBI);
          CopyMI->findRegisterUseOperand(VirtReg)->setIsKill();
        }
        MO.setIsKill();
      } else if (MO.isDef() && !MO.isDead()) {
        // Add a copy to VirtReg.
        AddDefaultPred(BuildMI(MBB, NextI, MI->getDebugLoc(),
                               TII->get(ARM::FCPYD), VirtReg)
                       .addReg(MO.getReg()));
      }
    }
  }

  return Modified;
}

bool NEONPreAllocPass::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();

  bool Modified = false;
  for (MachineFunction::iterator MFI = MF.begin(), E = MF.end(); MFI != E;
       ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    Modified |= PreAllocNEONRegisters(MBB);
  }

  return Modified;
}

/// createNEONPreAllocPass - returns an instance of the NEON register
/// pre-allocation pass.
FunctionPass *llvm::createNEONPreAllocPass() {
  return new NEONPreAllocPass();
}
