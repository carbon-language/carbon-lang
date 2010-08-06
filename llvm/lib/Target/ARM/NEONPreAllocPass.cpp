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
                          unsigned FirstOpnd, unsigned NumRegs,
                          unsigned Offset, unsigned Stride) const;
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

bool
NEONPreAllocPass::FormsRegSequence(MachineInstr *MI,
                                   unsigned FirstOpnd, unsigned NumRegs,
                                   unsigned Offset, unsigned Stride) const {
  MachineOperand &FMO = MI->getOperand(FirstOpnd);
  assert(FMO.isReg() && FMO.getSubReg() == 0 && "unexpected operand");
  unsigned VirtReg = FMO.getReg();
  (void)VirtReg;
  assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
         "expected a virtual register");

  unsigned LastSubIdx = 0;
  if (FMO.isDef()) {
    MachineInstr *RegSeq = 0;
    for (unsigned R = 0; R < NumRegs; ++R) {
      const MachineOperand &MO = MI->getOperand(FirstOpnd + R);
      assert(MO.isReg() && MO.getSubReg() == 0 && "unexpected operand");
      unsigned VirtReg = MO.getReg();
      assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
             "expected a virtual register");
      // Feeding into a REG_SEQUENCE.
      if (!MRI->hasOneNonDBGUse(VirtReg))
        return false;
      MachineInstr *UseMI = &*MRI->use_nodbg_begin(VirtReg);
      if (!UseMI->isRegSequence())
        return false;
      if (RegSeq && RegSeq != UseMI)
        return false;
      unsigned OpIdx = 1 + (Offset + R * Stride) * 2;
      if (UseMI->getOperand(OpIdx).getReg() != VirtReg)
        llvm_unreachable("Malformed REG_SEQUENCE instruction!");
      unsigned SubIdx = UseMI->getOperand(OpIdx + 1).getImm();
      if (LastSubIdx) {
        if (LastSubIdx != SubIdx-Stride)
          return false;
      } else {
        // Must start from dsub_0 or qsub_0.
        if (SubIdx != (ARM::dsub_0+Offset) &&
            SubIdx != (ARM::qsub_0+Offset))
          return false;
      }
      RegSeq = UseMI;
      LastSubIdx = SubIdx;
    }

    // In the case of vld3, etc., make sure the trailing operand of
    // REG_SEQUENCE is an undef.
    if (NumRegs == 3) {
      unsigned OpIdx = 1 + (Offset + 3 * Stride) * 2;
      const MachineOperand &MO = RegSeq->getOperand(OpIdx);
      unsigned VirtReg = MO.getReg();
      MachineInstr *DefMI = MRI->getVRegDef(VirtReg);
      if (!DefMI || !DefMI->isImplicitDef())
        return false;
    }
    return true;
  }

  unsigned LastSrcReg = 0;
  SmallVector<unsigned, 4> SubIds;
  for (unsigned R = 0; R < NumRegs; ++R) {
    const MachineOperand &MO = MI->getOperand(FirstOpnd + R);
    assert(MO.isReg() && MO.getSubReg() == 0 && "unexpected operand");
    unsigned VirtReg = MO.getReg();
    assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
           "expected a virtual register");
    // Extracting from a Q or QQ register.
    MachineInstr *DefMI = MRI->getVRegDef(VirtReg);
    if (!DefMI || !DefMI->isCopy() || !DefMI->getOperand(1).getSubReg())
      return false;
    VirtReg = DefMI->getOperand(1).getReg();
    if (LastSrcReg && LastSrcReg != VirtReg)
      return false;
    LastSrcReg = VirtReg;
    const TargetRegisterClass *RC = MRI->getRegClass(VirtReg);
    if (RC != ARM::QPRRegisterClass &&
        RC != ARM::QQPRRegisterClass &&
        RC != ARM::QQQQPRRegisterClass)
      return false;
    unsigned SubIdx = DefMI->getOperand(1).getSubReg();
    if (LastSubIdx) {
      if (LastSubIdx != SubIdx-Stride)
        return false;
    } else {
      // Must start from dsub_0 or qsub_0.
      if (SubIdx != (ARM::dsub_0+Offset) &&
          SubIdx != (ARM::qsub_0+Offset))
        return false;
    }
    SubIds.push_back(SubIdx);
    LastSubIdx = SubIdx;
  }

  // FIXME: Update the uses of EXTRACT_SUBREG from REG_SEQUENCE is
  // currently required for correctness. e.g.
  //  %reg1041<def> = REG_SEQUENCE %reg1040<kill>, 5, %reg1035<kill>, 6
  //  %reg1042<def> = EXTRACT_SUBREG %reg1041, 6
  //  %reg1043<def> = EXTRACT_SUBREG %reg1041, 5
  //  VST1q16 %reg1025<kill>, 0, %reg1043<kill>, %reg1042<kill>,
  // reg1042 and reg1043 should be replaced with reg1041:6 and reg1041:5
  // respectively.
  // We need to change how we model uses of REG_SEQUENCE.
  for (unsigned R = 0; R < NumRegs; ++R) {
    MachineOperand &MO = MI->getOperand(FirstOpnd + R);
    unsigned OldReg = MO.getReg();
    MachineInstr *DefMI = MRI->getVRegDef(OldReg);
    assert(DefMI->isCopy());
    MO.setReg(LastSrcReg);
    MO.setSubReg(SubIds[R]);
    MO.setIsKill(false);
    // Delete the EXTRACT_SUBREG if its result is now dead.
    if (MRI->use_empty(OldReg))
      DefMI->eraseFromParent();
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
    if (FormsRegSequence(MI, FirstOpnd, NumRegs, Offset, Stride))
      continue;
    llvm_unreachable("expected a REG_SEQUENCE");
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
