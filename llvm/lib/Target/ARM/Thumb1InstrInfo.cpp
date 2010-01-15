//===- Thumb1InstrInfo.cpp - Thumb-1 Instruction Information ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-1 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "Thumb1InstrInfo.h"
#include "ARM.h"
#include "ARMGenInstrInfo.inc"
#include "ARMMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/ADT/SmallVector.h"
#include "Thumb1InstrInfo.h"

using namespace llvm;

Thumb1InstrInfo::Thumb1InstrInfo(const ARMSubtarget &STI)
  : ARMBaseInstrInfo(STI), RI(*this, STI) {
}

unsigned Thumb1InstrInfo::getUnindexedOpcode(unsigned Opc) const {
  return 0;
}

bool Thumb1InstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I,
                                   unsigned DestReg, unsigned SrcReg,
                                   const TargetRegisterClass *DestRC,
                                   const TargetRegisterClass *SrcRC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  if (DestRC == ARM::GPRRegisterClass) {
    if (SrcRC == ARM::GPRRegisterClass) {
      BuildMI(MBB, I, DL, get(ARM::tMOVgpr2gpr), DestReg).addReg(SrcReg);
      return true;
    } else if (SrcRC == ARM::tGPRRegisterClass) {
      BuildMI(MBB, I, DL, get(ARM::tMOVtgpr2gpr), DestReg).addReg(SrcReg);
      return true;
    }
  } else if (DestRC == ARM::tGPRRegisterClass) {
    if (SrcRC == ARM::GPRRegisterClass) {
      BuildMI(MBB, I, DL, get(ARM::tMOVgpr2tgpr), DestReg).addReg(SrcReg);
      return true;
    } else if (SrcRC == ARM::tGPRRegisterClass) {
      BuildMI(MBB, I, DL, get(ARM::tMOVr), DestReg).addReg(SrcReg);
      return true;
    }
  }

  return false;
}

bool Thumb1InstrInfo::
canFoldMemoryOperand(const MachineInstr *MI,
                     const SmallVectorImpl<unsigned> &Ops) const {
  if (Ops.size() != 1) return false;

  unsigned OpNum = Ops[0];
  unsigned Opc = MI->getOpcode();
  switch (Opc) {
  default: break;
  case ARM::tMOVr:
  case ARM::tMOVtgpr2gpr:
  case ARM::tMOVgpr2tgpr:
  case ARM::tMOVgpr2gpr: {
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      if (TargetRegisterInfo::isPhysicalRegister(SrcReg) &&
          !isARMLowRegister(SrcReg))
        // tSpill cannot take a high register operand.
        return false;
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      if (TargetRegisterInfo::isPhysicalRegister(DstReg) &&
          !isARMLowRegister(DstReg))
        // tRestore cannot target a high register operand.
        return false;
    }
    return true;
  }
  }

  return false;
}

void Thumb1InstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  assert((RC == ARM::tGPRRegisterClass ||
          (TargetRegisterInfo::isPhysicalRegister(SrcReg) &&
           isARMLowRegister(SrcReg))) && "Unknown regclass!");

  if (RC == ARM::tGPRRegisterClass ||
      (TargetRegisterInfo::isPhysicalRegister(SrcReg) &&
       isARMLowRegister(SrcReg))) {
    MachineFunction &MF = *MBB.getParent();
    MachineFrameInfo &MFI = *MF.getFrameInfo();
    MachineMemOperand *MMO =
      MF.getMachineMemOperand(PseudoSourceValue::getFixedStack(FI),
                              MachineMemOperand::MOStore, 0,
                              MFI.getObjectSize(FI),
                              MFI.getObjectAlignment(FI));
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::tSpill))
                   .addReg(SrcReg, getKillRegState(isKill))
                   .addFrameIndex(FI).addImm(0).addMemOperand(MMO));
  }
}

void Thumb1InstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const {
  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (I != MBB.end()) DL = I->getDebugLoc();

  assert((RC == ARM::tGPRRegisterClass ||
          (TargetRegisterInfo::isPhysicalRegister(DestReg) &&
           isARMLowRegister(DestReg))) && "Unknown regclass!");

  if (RC == ARM::tGPRRegisterClass ||
      (TargetRegisterInfo::isPhysicalRegister(DestReg) &&
       isARMLowRegister(DestReg))) {
    MachineFunction &MF = *MBB.getParent();
    MachineFrameInfo &MFI = *MF.getFrameInfo();
    MachineMemOperand *MMO =
      MF.getMachineMemOperand(PseudoSourceValue::getFixedStack(FI),
                              MachineMemOperand::MOLoad, 0,
                              MFI.getObjectSize(FI),
                              MFI.getObjectAlignment(FI));
    AddDefaultPred(BuildMI(MBB, I, DL, get(ARM::tRestore), DestReg)
                   .addFrameIndex(FI).addImm(0).addMemOperand(MMO));
  }
}

bool Thumb1InstrInfo::
spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MI,
                          const std::vector<CalleeSavedInfo> &CSI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL = DebugLoc::getUnknownLoc();
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, get(ARM::tPUSH));
  AddDefaultPred(MIB);
  MIB.addReg(0); // No write back.
  for (unsigned i = CSI.size(); i != 0; --i) {
    unsigned Reg = CSI[i-1].getReg();
    // Add the callee-saved register as live-in. It's killed at the spill.
    MBB.addLiveIn(Reg);
    MIB.addReg(Reg, RegState::Kill);
  }
  return true;
}

bool Thumb1InstrInfo::
restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            const std::vector<CalleeSavedInfo> &CSI) const {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  if (CSI.empty())
    return false;

  bool isVarArg = AFI->getVarArgsRegSaveSize() > 0;
  DebugLoc DL = MI->getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(ARM::tPOP));
  AddDefaultPred(MIB);
  MIB.addReg(0); // No write back.

  bool NumRegs = false;
  for (unsigned i = CSI.size(); i != 0; --i) {
    unsigned Reg = CSI[i-1].getReg();
    if (Reg == ARM::LR) {
      // Special epilogue for vararg functions. See emitEpilogue
      if (isVarArg)
        continue;
      Reg = ARM::PC;
      (*MIB).setDesc(get(ARM::tPOP_RET));
      MI = MBB.erase(MI);
    }
    MIB.addReg(Reg, getDefRegState(true));
    NumRegs = true;
  }

  // It's illegal to emit pop instruction without operands.
  if (NumRegs)
    MBB.insert(MI, &*MIB);

  return true;
}

MachineInstr *Thumb1InstrInfo::
foldMemoryOperandImpl(MachineFunction &MF, MachineInstr *MI,
                      const SmallVectorImpl<unsigned> &Ops, int FI) const {
  if (Ops.size() != 1) return NULL;

  unsigned OpNum = Ops[0];
  unsigned Opc = MI->getOpcode();
  MachineInstr *NewMI = NULL;
  switch (Opc) {
  default: break;
  case ARM::tMOVr:
  case ARM::tMOVtgpr2gpr:
  case ARM::tMOVgpr2tgpr:
  case ARM::tMOVgpr2gpr: {
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      bool isKill = MI->getOperand(1).isKill();
      if (TargetRegisterInfo::isPhysicalRegister(SrcReg) &&
          !isARMLowRegister(SrcReg))
        // tSpill cannot take a high register operand.
        break;
      NewMI = AddDefaultPred(BuildMI(MF, MI->getDebugLoc(), get(ARM::tSpill))
                             .addReg(SrcReg, getKillRegState(isKill))
                             .addFrameIndex(FI).addImm(0));
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      if (TargetRegisterInfo::isPhysicalRegister(DstReg) &&
          !isARMLowRegister(DstReg))
        // tRestore cannot target a high register operand.
        break;
      bool isDead = MI->getOperand(0).isDead();
      NewMI = AddDefaultPred(BuildMI(MF, MI->getDebugLoc(), get(ARM::tRestore))
                             .addReg(DstReg,
                                     RegState::Define | getDeadRegState(isDead))
                             .addFrameIndex(FI).addImm(0));
    }
    break;
  }
  }

  return NewMI;
}
