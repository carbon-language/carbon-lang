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
#include "SystemZInstrBuilder.h"
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
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  unsigned Opc = 0;
  if (RC == &SystemZ::GR32RegClass ||
      RC == &SystemZ::ADDR32RegClass)
    Opc = SystemZ::MOV32mr;
  else if (RC == &SystemZ::GR64RegClass ||
           RC == &SystemZ::ADDR64RegClass) {
    Opc = SystemZ::MOV64mr;
  } else if (RC == &SystemZ::FP32RegClass) {
    Opc = SystemZ::FMOV32mr;
  } else if (RC == &SystemZ::FP64RegClass) {
    Opc = SystemZ::FMOV64mr;
  } else
    assert(0 && "Unsupported regclass to store");

  addFrameReference(BuildMI(MBB, MI, DL, get(Opc)), FrameIdx)
    .addReg(SrcReg, getKillRegState(isKill));
}

void SystemZInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                           unsigned DestReg, int FrameIdx,
                                           const TargetRegisterClass *RC) const{
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  unsigned Opc = 0;
  if (RC == &SystemZ::GR32RegClass ||
      RC == &SystemZ::ADDR32RegClass)
    Opc = SystemZ::MOV32rm;
  else if (RC == &SystemZ::GR64RegClass ||
           RC == &SystemZ::ADDR64RegClass) {
    Opc = SystemZ::MOV64rm;
  } else if (RC == &SystemZ::FP32RegClass) {
    Opc = SystemZ::FMOV32rm;
  } else if (RC == &SystemZ::FP64RegClass) {
    Opc = SystemZ::FMOV64rm;
  } else
    assert(0 && "Unsupported regclass to store");

  addFrameReference(BuildMI(MBB, MI, DL, get(Opc), DestReg), FrameIdx);
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
    if (CommonRC == &SystemZ::GR64RegClass ||
        CommonRC == &SystemZ::ADDR64RegClass) {
      BuildMI(MBB, I, DL, get(SystemZ::MOV64rr), DestReg).addReg(SrcReg);
    } else if (CommonRC == &SystemZ::GR32RegClass ||
               CommonRC == &SystemZ::ADDR32RegClass) {
      BuildMI(MBB, I, DL, get(SystemZ::MOV32rr), DestReg).addReg(SrcReg);
    } else if (CommonRC == &SystemZ::GR64PRegClass) {
      BuildMI(MBB, I, DL, get(SystemZ::MOV64rrP), DestReg).addReg(SrcReg);
    } else if (CommonRC == &SystemZ::GR128RegClass) {
      BuildMI(MBB, I, DL, get(SystemZ::MOV128rr), DestReg).addReg(SrcReg);
    } else if (CommonRC == &SystemZ::FP32RegClass) {
      BuildMI(MBB, I, DL, get(SystemZ::FMOV32rr), DestReg).addReg(SrcReg);
    } else if (CommonRC == &SystemZ::FP64RegClass) {
      BuildMI(MBB, I, DL, get(SystemZ::FMOV64rr), DestReg).addReg(SrcReg);
    } else {
      return false;
    }

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
  switch (MI.getOpcode()) {
  default:
    return false;
  case SystemZ::MOV32rr:
  case SystemZ::MOV64rr:
  case SystemZ::MOV64rrP:
  case SystemZ::MOV128rr:
  case SystemZ::FMOV32rr:
  case SystemZ::FMOV64rr:
    assert(MI.getNumOperands() >= 2 &&
           MI.getOperand(0).isReg() &&
           MI.getOperand(1).isReg() &&
           "invalid register-register move instruction");
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    SrcSubIdx = MI.getOperand(1).getSubReg();
    DstSubIdx = MI.getOperand(0).getSubReg();
    return true;
  }
}

bool
SystemZInstrInfo::spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  MachineFunction &MF = *MBB.getParent();
  SystemZMachineFunctionInfo *MFI = MF.getInfo<SystemZMachineFunctionInfo>();
  unsigned CalleeFrameSize = 0;

  // Scan the callee-saved and find the bounds of register spill area.
  unsigned LowReg = 0, HighReg = 0, StartOffset = -1U, EndOffset = 0;
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    const TargetRegisterClass *RegClass = CSI[i].getRegClass();
    if (RegClass != &SystemZ::FP64RegClass) {
      unsigned Offset = RegSpillOffsets[Reg];
      CalleeFrameSize += 8;
      if (StartOffset > Offset) {
        LowReg = Reg; StartOffset = Offset;
      }
      if (EndOffset < Offset) {
        HighReg = Reg; EndOffset = RegSpillOffsets[Reg];
      }
    }
  }

  // Save information for epilogue inserter.
  MFI->setCalleeSavedFrameSize(CalleeFrameSize);
  MFI->setLowReg(LowReg); MFI->setHighReg(HighReg);

  // Save GPRs
  if (StartOffset) {
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
  }

  // Save FPRs
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    const TargetRegisterClass *RegClass = CSI[i].getRegClass();
    if (RegClass == &SystemZ::FP64RegClass) {
      MBB.addLiveIn(Reg);
      storeRegToStackSlot(MBB, MI, Reg, true, CSI[i].getFrameIdx(), RegClass);
    }
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

  // Restore FP registers
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    const TargetRegisterClass *RegClass = CSI[i].getRegClass();
    if (RegClass == &SystemZ::FP64RegClass)
      loadRegFromStackSlot(MBB, MI, Reg, CSI[i].getFrameIdx(), RegClass);
  }

  // Restore GP registers
  unsigned LowReg = MFI->getLowReg(), HighReg = MFI->getHighReg();
  unsigned StartOffset = RegSpillOffsets[LowReg];

  if (StartOffset) {
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
  }

  return true;
}

unsigned
SystemZInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                               MachineBasicBlock *FBB,
                            const SmallVectorImpl<MachineOperand> &Cond) const {
  // FIXME: this should probably have a DebugLoc operand
  DebugLoc dl = DebugLoc::getUnknownLoc();
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 1 || Cond.size() == 0) &&
         "SystemZ branch conditions have one component!");

  if (Cond.empty()) {
    // Unconditional branch?
    assert(!FBB && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, dl, get(SystemZ::JMP)).addMBB(TBB);
    return 1;
  }

  // Conditional branch.
  unsigned Count = 0;
  SystemZCC::CondCodes CC = (SystemZCC::CondCodes)Cond[0].getImm();
  BuildMI(&MBB, dl, getBrCond(CC)).addMBB(TBB);
  ++Count;

  if (FBB) {
    // Two-way Conditional branch. Insert the second branch.
    BuildMI(&MBB, dl, get(SystemZ::JMP)).addMBB(FBB);
    ++Count;
  }
  return Count;
}

const TargetInstrDesc&
SystemZInstrInfo::getBrCond(SystemZCC::CondCodes CC) const {
  switch (CC) {
  default:
    assert(0 && "Unknown condition code!");
  case SystemZCC::O:   return get(SystemZ::JO);
  case SystemZCC::H:   return get(SystemZ::JH);
  case SystemZCC::NLE: return get(SystemZ::JNLE);
  case SystemZCC::L:   return get(SystemZ::JL);
  case SystemZCC::NHE: return get(SystemZ::JNHE);
  case SystemZCC::LH:  return get(SystemZ::JLH);
  case SystemZCC::NE:  return get(SystemZ::JNE);
  case SystemZCC::E:   return get(SystemZ::JE);
  case SystemZCC::NLH: return get(SystemZ::JNLH);
  case SystemZCC::HE:  return get(SystemZ::JHE);
  case SystemZCC::NL:  return get(SystemZ::JNL);
  case SystemZCC::LE:  return get(SystemZ::JLE);
  case SystemZCC::NH:  return get(SystemZ::JNH);
  case SystemZCC::NO:  return get(SystemZ::JNO);
  }
}

const TargetInstrDesc&
SystemZInstrInfo::getLongDispOpc(unsigned Opc) const {
  switch (Opc) {
  case SystemZ::MOV32mr:   return get(SystemZ::MOV32mry);
  case SystemZ::MOV32rm:   return get(SystemZ::MOV32rmy);
  case SystemZ::MOVSX32rm16: return get(SystemZ::MOVSX32rm16y);
  case SystemZ::MOV32m8r:  return get(SystemZ::MOV32m8ry);
  case SystemZ::MOV32m16r: return get(SystemZ::MOV32m16ry);
  case SystemZ::MOV64m8r:  return get(SystemZ::MOV64m8ry);
  case SystemZ::MOV64m16r: return get(SystemZ::MOV64m16ry);
  case SystemZ::MOV64m32r: return get(SystemZ::MOV64m32ry);
  case SystemZ::MOV8mi:    return get(SystemZ::MOV8miy);
  case SystemZ::MUL32rm:   return get(SystemZ::MUL32rmy);
  case SystemZ::CMP32rm:   return get(SystemZ::CMP32rmy);
  case SystemZ::UCMP32rm:  return get(SystemZ::UCMP32rmy);
  case SystemZ::FMOV32mr:  return get(SystemZ::FMOV32mry);
  case SystemZ::FMOV64mr:  return get(SystemZ::FMOV64mry);
  default: return get(Opc);
  }
}

