//===- SystemZInstrInfo.cpp - SystemZ Instruction Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SystemZ implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "SystemZ.h"
#include "SystemZInstrInfo.h"
#include "SystemZMachineFunctionInfo.h"
#include "SystemZTargetMachine.h"
#include "SystemZGenInstrInfo.inc"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"

using namespace llvm;

SystemZInstrInfo::SystemZInstrInfo(SystemZTargetMachine &tm)
  : TargetInstrInfoImpl(SystemZInsts, array_lengthof(SystemZInsts)),
    RI(tm, *this), TM(tm) {}

void SystemZInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MI,
                                    unsigned SrcReg, bool isKill, int FrameIdx,
                                    const TargetRegisterClass *RC) const {
  assert(0 && "Cannot store this register to stack slot!");
}

void SystemZInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                           unsigned DestReg, int FrameIdx,
                                           const TargetRegisterClass *RC) const{
  assert(0 && "Cannot store this register to stack slot!");
}

bool SystemZInstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I,
                                    unsigned DestReg, unsigned SrcReg,
                                    const TargetRegisterClass *DestRC,
                                    const TargetRegisterClass *SrcRC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  // Determine if DstRC and SrcRC have a common superclass.
  const TargetRegisterClass *CommonRC = DestRC;
  if (DestRC == SrcRC)
    /* Same regclass for source and dest */;
  else if (CommonRC->hasSuperClass(SrcRC))
    CommonRC = SrcRC;
  else if (!CommonRC->hasSubClass(SrcRC))
    CommonRC = 0;

  if (CommonRC) {
    unsigned Opc;
    if (CommonRC == &SystemZ::GR64RegClass ||
        CommonRC == &SystemZ::ADDR64RegClass) {
      Opc = SystemZ::MOV64rr;
    } else if (CommonRC == &SystemZ::GR32RegClass ||
               CommonRC == &SystemZ::ADDR32RegClass) {
      Opc = SystemZ::MOV32rr;
    } else {
      return false;
    }

    BuildMI(MBB, I, DL, get(Opc), DestReg).addReg(SrcReg);
    return true;
  }

  if ((SrcRC == &SystemZ::GR64RegClass &&
       DestRC == &SystemZ::ADDR64RegClass) ||
      (DestRC == &SystemZ::GR64RegClass &&
       SrcRC == &SystemZ::ADDR64RegClass)) {
    BuildMI(MBB, I, DL, get(SystemZ::MOV64rr), DestReg).addReg(SrcReg);
    return true;
  } else if ((SrcRC == &SystemZ::GR32RegClass &&
              DestRC == &SystemZ::ADDR32RegClass) ||
             (DestRC == &SystemZ::GR32RegClass &&
              SrcRC == &SystemZ::ADDR32RegClass)) {
    BuildMI(MBB, I, DL, get(SystemZ::MOV32rr), DestReg).addReg(SrcReg);
    return true;
  }

  return false;
}

bool
SystemZInstrInfo::isMoveInstr(const MachineInstr& MI,
                              unsigned &SrcReg, unsigned &DstReg,
                              unsigned &SrcSubIdx, unsigned &DstSubIdx) const {
  SrcSubIdx = DstSubIdx = 0; // No sub-registers yet.

  switch (MI.getOpcode()) {
  default:
    return false;
  case SystemZ::MOV32rr:
  case SystemZ::MOV64rr:
    assert(MI.getNumOperands() >= 2 &&
           MI.getOperand(0).isReg() &&
           MI.getOperand(1).isReg() &&
           "invalid register-register move instruction");
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    return true;
  }
}

bool
SystemZInstrInfo::spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
  if (CSI.empty())
    return false;

  MachineFunction &MF = *MBB.getParent();
  SystemZMachineFunctionInfo *MFI = MF.getInfo<SystemZMachineFunctionInfo>();
  MFI->setCalleeSavedFrameSize(CSI.size() * 8);

  return true;
}

bool
SystemZInstrInfo::restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                             MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
  return true;
}

unsigned
SystemZInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                              MachineBasicBlock *FBB,
                            const SmallVectorImpl<MachineOperand> &Cond) const {
  assert(0 && "Implement branches!");

  return 0;
}
