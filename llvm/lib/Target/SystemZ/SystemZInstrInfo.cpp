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
    RI(tm, *this), TM(tm) {
  // Fill the spill offsets map
  static const unsigned SpillOffsTab[][2] = {
    { SystemZ::R2D,  0x10 },
    { SystemZ::R3D,  0x18 },
    { SystemZ::R4D,  0x20 },
    { SystemZ::R5D,  0x28 },
    { SystemZ::R6D,  0x30 },
    { SystemZ::R7D,  0x38 },
    { SystemZ::R8D,  0x40 },
    { SystemZ::R9D,  0x48 },
    { SystemZ::R10D, 0x50 },
    { SystemZ::R11D, 0x58 },
    { SystemZ::R12D, 0x60 },
    { SystemZ::R13D, 0x68 },
    { SystemZ::R14D, 0x70 },
    { SystemZ::R15D, 0x78 }
  };

  RegSpillOffsets.grow(SystemZ::NUM_TARGET_REGS);

  for (unsigned i = 0, e = array_lengthof(SpillOffsTab); i != e; ++i)
    RegSpillOffsets[SpillOffsTab[i][0]] = SpillOffsTab[i][1];
}

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
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  MachineFunction &MF = *MBB.getParent();
  SystemZMachineFunctionInfo *MFI = MF.getInfo<SystemZMachineFunctionInfo>();
  MFI->setCalleeSavedFrameSize(CSI.size() * 8);

  // Scan the callee-saved and find the bounds of register spill area.
  unsigned LowReg = 0, HighReg = 0, StartOffset = -1U, EndOffset = 0;
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    unsigned Offset = RegSpillOffsets[Reg];
    if (StartOffset > Offset) {
      LowReg = Reg; StartOffset = Offset;
    }
    if (EndOffset < Offset) {
      HighReg = Reg; EndOffset = RegSpillOffsets[Reg];
    }
  }

  // Save information for epilogue inserter.
  MFI->setLowReg(LowReg); MFI->setHighReg(HighReg);

  // Build a store instruction. Use STORE MULTIPLE instruction if there are many
  // registers to store, otherwise - just STORE.
  MachineInstrBuilder MIB =
    BuildMI(MBB, MI, DL, get((LowReg == HighReg ?
                              SystemZ::MOV64mr : SystemZ::MOV64mrm)));

  // Add store operands.
  MIB.addReg(SystemZ::R15D).addImm(StartOffset);
  if (LowReg == HighReg)
    MIB.addReg(0);
  MIB.addReg(LowReg, RegState::Kill);
  if (LowReg != HighReg)
    MIB.addReg(HighReg, RegState::Kill);

  // Do a second scan adding regs as being killed by instruction
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    // Add the callee-saved register as live-in. It's killed at the spill.
    MBB.addLiveIn(Reg);
    if (Reg != LowReg && Reg != HighReg)
      MIB.addReg(Reg, RegState::ImplicitKill);
  }

  return true;
}

bool
SystemZInstrInfo::restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                             MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  MachineFunction &MF = *MBB.getParent();
  const TargetRegisterInfo *RegInfo= MF.getTarget().getRegisterInfo();
  SystemZMachineFunctionInfo *MFI = MF.getInfo<SystemZMachineFunctionInfo>();

  unsigned LowReg = MFI->getLowReg(), HighReg = MFI->getHighReg();
  unsigned StartOffset = RegSpillOffsets[LowReg];

  // Build a load instruction. Use LOAD MULTIPLE instruction if there are many
  // registers to load, otherwise - just LOAD.
  MachineInstrBuilder MIB =
    BuildMI(MBB, MI, DL, get((LowReg == HighReg ?
                              SystemZ::MOV64rm : SystemZ::MOV64rmm)));
  // Add store operands.
  MIB.addReg(LowReg, RegState::Define);
  if (LowReg != HighReg)
    MIB.addReg(HighReg, RegState::Define);

  MIB.addReg((RegInfo->hasFP(MF) ? SystemZ::R11D : SystemZ::R15D));
  MIB.addImm(StartOffset);
  if (LowReg == HighReg)
    MIB.addReg(0);

  // Do a second scan adding regs as being defined by instruction
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    if (Reg != LowReg && Reg != HighReg)
      MIB.addReg(Reg, RegState::ImplicitDefine);
  }

  return true;
}

unsigned
SystemZInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                              MachineBasicBlock *FBB,
                            const SmallVectorImpl<MachineOperand> &Cond) const {
  assert(0 && "Implement branches!");

  return 0;
}

const TargetInstrDesc&
SystemZInstrInfo::getBrCond(SystemZCC::CondCodes CC) const {
  unsigned Opc;
  switch (CC) {
  default:
    assert(0 && "Unknown condition code!");
  case SystemZCC::E:
    Opc = SystemZ::JE;
    break;
  case SystemZCC::NE:
    Opc = SystemZ::JNE;
    break;
  case SystemZCC::H:
    Opc = SystemZ::JH;
    break;
  case SystemZCC::L:
    Opc = SystemZ::JL;
    break;
  case SystemZCC::HE:
    Opc = SystemZ::JHE;
    break;
  case SystemZCC::LE:
    Opc = SystemZ::JLE;
    break;
  }

  return get(Opc);
}
