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
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

namespace {
  class NEONPreAllocPass : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    MachineRegisterInfo *MRI;

  public:
    static char ID;
    NEONPreAllocPass() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "NEON register pre-allocation pass";
    }

  private:
    bool FormsRegSequence(MachineInstr *MI,
                          unsigned FirstOpnd, unsigned NumRegs);
    bool PreAllocNEONRegisters(MachineBasicBlock &MBB);
  };

  char NEONPreAllocPass::ID = 0;
}

static bool isNEONMultiRegOp(int Opcode, unsigned &FirstOpnd, unsigned &NumRegs,
                             unsigned &Offset, unsigned &Stride) {
  // Default to unit stride with no offset.
  Stride = 1;
  Offset = 0;

  switch (Opcode) {
  default:
    break;

  case ARM::VLD1q8:
  case ARM::VLD1q16:
  case ARM::VLD1q32:
  case ARM::VLD1q64:
  case ARM::VLD2d8:
  case ARM::VLD2d16:
  case ARM::VLD2d32:
  case ARM::VLD2LNd8:
  case ARM::VLD2LNd16:
  case ARM::VLD2LNd32:
    FirstOpnd = 0;
    NumRegs = 2;
    return true;

  case ARM::VLD2q8:
  case ARM::VLD2q16:
  case ARM::VLD2q32:
    FirstOpnd = 0;
    NumRegs = 4;
    return true;

  case ARM::VLD2LNq16:
  case ARM::VLD2LNq32:
    FirstOpnd = 0;
    NumRegs = 2;
    Offset = 0;
    Stride = 2;
    return true;

  case ARM::VLD2LNq16odd:
  case ARM::VLD2LNq32odd:
    FirstOpnd = 0;
    NumRegs = 2;
    Offset = 1;
    Stride = 2;
    return true;

  case ARM::VLD3d8:
  case ARM::VLD3d16:
  case ARM::VLD3d32:
  case ARM::VLD1d64T:
  case ARM::VLD3LNd8:
  case ARM::VLD3LNd16:
  case ARM::VLD3LNd32:
    FirstOpnd = 0;
    NumRegs = 3;
    return true;

  case ARM::VLD3q8_UPD:
  case ARM::VLD3q16_UPD:
  case ARM::VLD3q32_UPD:
    FirstOpnd = 0;
    NumRegs = 3;
    Offset = 0;
    Stride = 2;
    return true;

  case ARM::VLD3q8odd_UPD:
  case ARM::VLD3q16odd_UPD:
  case ARM::VLD3q32odd_UPD:
    FirstOpnd = 0;
    NumRegs = 3;
    Offset = 1;
    Stride = 2;
    return true;

  case ARM::VLD3LNq16:
  case ARM::VLD3LNq32:
    FirstOpnd = 0;
    NumRegs = 3;
    Offset = 0;
    Stride = 2;
    return true;

  case ARM::VLD3LNq16odd:
  case ARM::VLD3LNq32odd:
    FirstOpnd = 0;
    NumRegs = 3;
    Offset = 1;
    Stride = 2;
    return true;

  case ARM::VLD4d8:
  case ARM::VLD4d16:
  case ARM::VLD4d32:
  case ARM::VLD1d64Q:
  case ARM::VLD4LNd8:
  case ARM::VLD4LNd16:
  case ARM::VLD4LNd32:
    FirstOpnd = 0;
    NumRegs = 4;
    return true;

  case ARM::VLD4q8_UPD:
  case ARM::VLD4q16_UPD:
  case ARM::VLD4q32_UPD:
    FirstOpnd = 0;
    NumRegs = 4;
    Offset = 0;
    Stride = 2;
    return true;

  case ARM::VLD4q8odd_UPD:
  case ARM::VLD4q16odd_UPD:
  case ARM::VLD4q32odd_UPD:
    FirstOpnd = 0;
    NumRegs = 4;
    Offset = 1;
    Stride = 2;
    return true;

  case ARM::VLD4LNq16:
  case ARM::VLD4LNq32:
    FirstOpnd = 0;
    NumRegs = 4;
    Offset = 0;
    Stride = 2;
    return true;

  case ARM::VLD4LNq16odd:
  case ARM::VLD4LNq32odd:
    FirstOpnd = 0;
    NumRegs = 4;
    Offset = 1;
    Stride = 2;
    return true;

  case ARM::VST1q8:
  case ARM::VST1q16:
  case ARM::VST1q32:
  case ARM::VST1q64:
  case ARM::VST2d8:
  case ARM::VST2d16:
  case ARM::VST2d32:
  case ARM::VST2LNd8:
  case ARM::VST2LNd16:
  case ARM::VST2LNd32:
    FirstOpnd = 2;
    NumRegs = 2;
    return true;

  case ARM::VST2q8:
  case ARM::VST2q16:
  case ARM::VST2q32:
    FirstOpnd = 2;
    NumRegs = 4;
    return true;

  case ARM::VST2LNq16:
  case ARM::VST2LNq32:
    FirstOpnd = 2;
    NumRegs = 2;
    Offset = 0;
    Stride = 2;
    return true;

  case ARM::VST2LNq16odd:
  case ARM::VST2LNq32odd:
    FirstOpnd = 2;
    NumRegs = 2;
    Offset = 1;
    Stride = 2;
    return true;

  case ARM::VST3d8:
  case ARM::VST3d16:
  case ARM::VST3d32:
  case ARM::VST1d64T:
  case ARM::VST3LNd8:
  case ARM::VST3LNd16:
  case ARM::VST3LNd32:
    FirstOpnd = 2;
    NumRegs = 3;
    return true;

  case ARM::VST3q8_UPD:
  case ARM::VST3q16_UPD:
  case ARM::VST3q32_UPD:
    FirstOpnd = 4;
    NumRegs = 3;
    Offset = 0;
    Stride = 2;
    return true;

  case ARM::VST3q8odd_UPD:
  case ARM::VST3q16odd_UPD:
  case ARM::VST3q32odd_UPD:
    FirstOpnd = 4;
    NumRegs = 3;
    Offset = 1;
    Stride = 2;
    return true;

  case ARM::VST3LNq16:
  case ARM::VST3LNq32:
    FirstOpnd = 2;
    NumRegs = 3;
    Offset = 0;
    Stride = 2;
    return true;

  case ARM::VST3LNq16odd:
  case ARM::VST3LNq32odd:
    FirstOpnd = 2;
    NumRegs = 3;
    Offset = 1;
    Stride = 2;
    return true;

  case ARM::VST4d8:
  case ARM::VST4d16:
  case ARM::VST4d32:
  case ARM::VST1d64Q:
  case ARM::VST4LNd8:
  case ARM::VST4LNd16:
  case ARM::VST4LNd32:
    FirstOpnd = 2;
    NumRegs = 4;
    return true;

  case ARM::VST4q8_UPD:
  case ARM::VST4q16_UPD:
  case ARM::VST4q32_UPD:
    FirstOpnd = 4;
    NumRegs = 4;
    Offset = 0;
    Stride = 2;
    return true;

  case ARM::VST4q8odd_UPD:
  case ARM::VST4q16odd_UPD:
  case ARM::VST4q32odd_UPD:
    FirstOpnd = 4;
    NumRegs = 4;
    Offset = 1;
    Stride = 2;
    return true;

  case ARM::VST4LNq16:
  case ARM::VST4LNq32:
    FirstOpnd = 2;
    NumRegs = 4;
    Offset = 0;
    Stride = 2;
    return true;

  case ARM::VST4LNq16odd:
  case ARM::VST4LNq32odd:
    FirstOpnd = 2;
    NumRegs = 4;
    Offset = 1;
    Stride = 2;
    return true;

  case ARM::VTBL2:
    FirstOpnd = 1;
    NumRegs = 2;
    return true;

  case ARM::VTBL3:
    FirstOpnd = 1;
    NumRegs = 3;
    return true;

  case ARM::VTBL4:
    FirstOpnd = 1;
    NumRegs = 4;
    return true;

  case ARM::VTBX2:
    FirstOpnd = 2;
    NumRegs = 2;
    return true;

  case ARM::VTBX3:
    FirstOpnd = 2;
    NumRegs = 3;
    return true;

  case ARM::VTBX4:
    FirstOpnd = 2;
    NumRegs = 4;
    return true;
  }

  return false;
}

bool NEONPreAllocPass::FormsRegSequence(MachineInstr *MI,
                                        unsigned FirstOpnd, unsigned NumRegs) {
  MachineInstr *RegSeq = 0;
  for (unsigned R = 0; R < NumRegs; ++R) {
    MachineOperand &MO = MI->getOperand(FirstOpnd + R);
    assert(MO.isReg() && MO.getSubReg() == 0 && "unexpected operand");
    unsigned VirtReg = MO.getReg();
    assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
           "expected a virtual register");
    if (!MRI->hasOneNonDBGUse(VirtReg))
      return false;
    MachineInstr *UseMI = &*MRI->use_nodbg_begin(VirtReg);
    if (UseMI->getOpcode() != TargetOpcode::REG_SEQUENCE)
      return false;
    if (RegSeq && RegSeq != UseMI)
      return false;
    RegSeq = UseMI;
  }
  return true;
}

bool NEONPreAllocPass::PreAllocNEONRegisters(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  for (; MBBI != E; ++MBBI) {
    MachineInstr *MI = &*MBBI;
    unsigned FirstOpnd, NumRegs, Offset, Stride;
    if (!isNEONMultiRegOp(MI->getOpcode(), FirstOpnd, NumRegs, Offset, Stride))
      continue;
    if (FormsRegSequence(MI, FirstOpnd, NumRegs))
      continue;

    MachineBasicBlock::iterator NextI = llvm::next(MBBI);
    for (unsigned R = 0; R < NumRegs; ++R) {
      MachineOperand &MO = MI->getOperand(FirstOpnd + R);
      assert(MO.isReg() && MO.getSubReg() == 0 && "unexpected operand");
      unsigned VirtReg = MO.getReg();
      assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
             "expected a virtual register");

      // For now, just assign a fixed set of adjacent registers.
      // This leaves plenty of room for future improvements.
      static const unsigned NEONDRegs[] = {
        ARM::D0, ARM::D1, ARM::D2, ARM::D3,
        ARM::D4, ARM::D5, ARM::D6, ARM::D7
      };
      MO.setReg(NEONDRegs[Offset + R * Stride]);

      if (MO.isUse()) {
        // Insert a copy from VirtReg.
        TII->copyRegToReg(MBB, MBBI, MO.getReg(), VirtReg,
                          ARM::DPRRegisterClass, ARM::DPRRegisterClass);
        if (MO.isKill()) {
          MachineInstr *CopyMI = prior(MBBI);
          CopyMI->findRegisterUseOperand(VirtReg)->setIsKill();
        }
        MO.setIsKill();
      } else if (MO.isDef() && !MO.isDead()) {
        // Add a copy to VirtReg.
        TII->copyRegToReg(MBB, NextI, VirtReg, MO.getReg(),
                          ARM::DPRRegisterClass, ARM::DPRRegisterClass);
      }
    }
  }

  return Modified;
}

bool NEONPreAllocPass::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();
  MRI = &MF.getRegInfo();

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
