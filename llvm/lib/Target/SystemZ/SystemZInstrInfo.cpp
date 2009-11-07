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
#include "llvm/Support/ErrorHandling.h"
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

/// isGVStub - Return true if the GV requires an extra load to get the
/// real address.
static inline bool isGVStub(GlobalValue *GV, SystemZTargetMachine &TM) {
  return TM.getSubtarget<SystemZSubtarget>().GVRequiresExtraLoad(GV, TM, false);
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
  } else if (RC == &SystemZ::GR64PRegClass) {
    Opc = SystemZ::MOV64Pmr;
  } else if (RC == &SystemZ::GR128RegClass) {
    Opc = SystemZ::MOV128mr;
  } else
    llvm_unreachable("Unsupported regclass to store");

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
  } else if (RC == &SystemZ::GR64PRegClass) {
    Opc = SystemZ::MOV64Prm;
  } else if (RC == &SystemZ::GR128RegClass) {
    Opc = SystemZ::MOV128rm;
  } else
    llvm_unreachable("Unsupported regclass to load");

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

unsigned SystemZInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                               int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case SystemZ::MOV32rm:
  case SystemZ::MOV32rmy:
  case SystemZ::MOV64rm:
  case SystemZ::MOVSX32rm8:
  case SystemZ::MOVSX32rm16y:
  case SystemZ::MOVSX64rm8:
  case SystemZ::MOVSX64rm16:
  case SystemZ::MOVSX64rm32:
  case SystemZ::MOVZX32rm8:
  case SystemZ::MOVZX32rm16:
  case SystemZ::MOVZX64rm8:
  case SystemZ::MOVZX64rm16:
  case SystemZ::MOVZX64rm32:
  case SystemZ::FMOV32rm:
  case SystemZ::FMOV32rmy:
  case SystemZ::FMOV64rm:
  case SystemZ::FMOV64rmy:
  case SystemZ::MOV64Prm:
  case SystemZ::MOV64Prmy:
  case SystemZ::MOV128rm:
    if (MI->getOperand(1).isFI() &&
        MI->getOperand(2).isImm() && MI->getOperand(3).isReg() &&
        MI->getOperand(2).getImm() == 0 && MI->getOperand(3).getReg() == 0) {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned SystemZInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                              int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case SystemZ::MOV32mr:
  case SystemZ::MOV32mry:
  case SystemZ::MOV64mr:
  case SystemZ::MOV32m8r:
  case SystemZ::MOV32m8ry:
  case SystemZ::MOV32m16r:
  case SystemZ::MOV32m16ry:
  case SystemZ::MOV64m8r:
  case SystemZ::MOV64m8ry:
  case SystemZ::MOV64m16r:
  case SystemZ::MOV64m16ry:
  case SystemZ::MOV64m32r:
  case SystemZ::MOV64m32ry:
  case SystemZ::FMOV32mr:
  case SystemZ::FMOV32mry:
  case SystemZ::FMOV64mr:
  case SystemZ::FMOV64mry:
  case SystemZ::MOV64Pmr:
  case SystemZ::MOV64Pmry:
  case SystemZ::MOV128mr:
    if (MI->getOperand(0).isFI() &&
        MI->getOperand(1).isImm() && MI->getOperand(2).isReg() &&
        MI->getOperand(1).getImm() == 0 && MI->getOperand(2).getReg() == 0) {
      FrameIndex = MI->getOperand(0).getIndex();
      return MI->getOperand(3).getReg();
    }
    break;
  }
  return 0;
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

bool SystemZInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 1 && "Invalid Xbranch condition!");

  SystemZCC::CondCodes CC = static_cast<SystemZCC::CondCodes>(Cond[0].getImm());
  Cond[0].setImm(getOppositeCondition(CC));
  return false;
}

bool SystemZInstrInfo::BlockHasNoFallThrough(const MachineBasicBlock &MBB)const{
  if (MBB.empty()) return false;

  switch (MBB.back().getOpcode()) {
  case SystemZ::RET:   // Return.
  case SystemZ::JMP:   // Uncond branch.
  case SystemZ::JMPr:  // Indirect branch.
    return true;
  default: return false;
  }
}

bool SystemZInstrInfo::isUnpredicatedTerminator(const MachineInstr *MI) const {
  const TargetInstrDesc &TID = MI->getDesc();
  if (!TID.isTerminator()) return false;

  // Conditional branch is a special case.
  if (TID.isBranch() && !TID.isBarrier())
    return true;
  if (!TID.isPredicable())
    return true;
  return !isPredicated(MI);
}

bool SystemZInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                     MachineBasicBlock *&TBB,
                                     MachineBasicBlock *&FBB,
                                     SmallVectorImpl<MachineOperand> &Cond,
                                     bool AllowModify) const {
  // Start from the bottom of the block and work up, examining the
  // terminator instructions.
  MachineBasicBlock::iterator I = MBB.end();
  while (I != MBB.begin()) {
    --I;
    // Working from the bottom, when we see a non-terminator
    // instruction, we're done.
    if (!isUnpredicatedTerminator(I))
      break;

    // A terminator that isn't a branch can't easily be handled
    // by this analysis.
    if (!I->getDesc().isBranch())
      return true;

    // Handle unconditional branches.
    if (I->getOpcode() == SystemZ::JMP) {
      if (!AllowModify) {
        TBB = I->getOperand(0).getMBB();
        continue;
      }

      // If the block has any instructions after a JMP, delete them.
      while (next(I) != MBB.end())
        next(I)->eraseFromParent();
      Cond.clear();
      FBB = 0;

      // Delete the JMP if it's equivalent to a fall-through.
      if (MBB.isLayoutSuccessor(I->getOperand(0).getMBB())) {
        TBB = 0;
        I->eraseFromParent();
        I = MBB.end();
        continue;
      }

      // TBB is used to indicate the unconditinal destination.
      TBB = I->getOperand(0).getMBB();
      continue;
    }

    // Handle conditional branches.
    SystemZCC::CondCodes BranchCode = getCondFromBranchOpc(I->getOpcode());
    if (BranchCode == SystemZCC::INVALID)
      return true;  // Can't handle indirect branch.

    // Working from the bottom, handle the first conditional branch.
    if (Cond.empty()) {
      FBB = TBB;
      TBB = I->getOperand(0).getMBB();
      Cond.push_back(MachineOperand::CreateImm(BranchCode));
      continue;
    }

    // Handle subsequent conditional branches. Only handle the case where all
    // conditional branches branch to the same destination.
    assert(Cond.size() == 1);
    assert(TBB);

    // Only handle the case where all conditional branches branch to
    // the same destination.
    if (TBB != I->getOperand(0).getMBB())
      return true;

    SystemZCC::CondCodes OldBranchCode = (SystemZCC::CondCodes)Cond[0].getImm();
    // If the conditions are the same, we can leave them alone.
    if (OldBranchCode == BranchCode)
      continue;

    return true;
  }

  return false;
}

unsigned SystemZInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;

  while (I != MBB.begin()) {
    --I;
    if (I->getOpcode() != SystemZ::JMP &&
        getCondFromBranchOpc(I->getOpcode()) == SystemZCC::INVALID)
      break;
    // Remove the branch.
    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }

  return Count;
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
   llvm_unreachable("Unknown condition code!");
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

SystemZCC::CondCodes
SystemZInstrInfo::getCondFromBranchOpc(unsigned Opc) const {
  switch (Opc) {
  default:            return SystemZCC::INVALID;
  case SystemZ::JO:   return SystemZCC::O;
  case SystemZ::JH:   return SystemZCC::H;
  case SystemZ::JNLE: return SystemZCC::NLE;
  case SystemZ::JL:   return SystemZCC::L;
  case SystemZ::JNHE: return SystemZCC::NHE;
  case SystemZ::JLH:  return SystemZCC::LH;
  case SystemZ::JNE:  return SystemZCC::NE;
  case SystemZ::JE:   return SystemZCC::E;
  case SystemZ::JNLH: return SystemZCC::NLH;
  case SystemZ::JHE:  return SystemZCC::HE;
  case SystemZ::JNL:  return SystemZCC::NL;
  case SystemZ::JLE:  return SystemZCC::LE;
  case SystemZ::JNH:  return SystemZCC::NH;
  case SystemZ::JNO:  return SystemZCC::NO;
  }
}

SystemZCC::CondCodes
SystemZInstrInfo::getOppositeCondition(SystemZCC::CondCodes CC) const {
  switch (CC) {
  default:
    llvm_unreachable("Invalid condition!");
  case SystemZCC::O:   return SystemZCC::NO;
  case SystemZCC::H:   return SystemZCC::NH;
  case SystemZCC::NLE: return SystemZCC::LE;
  case SystemZCC::L:   return SystemZCC::NL;
  case SystemZCC::NHE: return SystemZCC::HE;
  case SystemZCC::LH:  return SystemZCC::NLH;
  case SystemZCC::NE:  return SystemZCC::E;
  case SystemZCC::E:   return SystemZCC::NE;
  case SystemZCC::NLH: return SystemZCC::LH;
  case SystemZCC::HE:  return SystemZCC::NHE;
  case SystemZCC::NL:  return SystemZCC::L;
  case SystemZCC::LE:  return SystemZCC::NLE;
  case SystemZCC::NH:  return SystemZCC::H;
  case SystemZCC::NO:  return SystemZCC::O;
  }
}

const TargetInstrDesc&
SystemZInstrInfo::getLongDispOpc(unsigned Opc) const {
  switch (Opc) {
  default:
    llvm_unreachable("Don't have long disp version of this instruction");
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
  case SystemZ::FMOV32rm:  return get(SystemZ::FMOV32rmy);
  case SystemZ::FMOV64rm:  return get(SystemZ::FMOV64rmy);
  case SystemZ::MOV64Pmr:  return get(SystemZ::MOV64Pmry);
  case SystemZ::MOV64Prm:  return get(SystemZ::MOV64Prmy);
  }
}
