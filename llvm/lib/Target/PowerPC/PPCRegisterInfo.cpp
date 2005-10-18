//===- PPCRegisterInfo.cpp - PowerPC Register Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reginfo"
#include "PPC.h"
#include "PPCInstrBuilder.h"
#include "PPCRegisterInfo.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdlib>
#include <iostream>
using namespace llvm;

PPCRegisterInfo::PPCRegisterInfo()
  : PPCGenRegisterInfo(PPC::ADJCALLSTACKDOWN, PPC::ADJCALLSTACKUP) {
  ImmToIdxMap[PPC::LD]   = PPC::LDX;    ImmToIdxMap[PPC::STD]  = PPC::STDX;
  ImmToIdxMap[PPC::LBZ]  = PPC::LBZX;   ImmToIdxMap[PPC::STB]  = PPC::STBX;
  ImmToIdxMap[PPC::LHZ]  = PPC::LHZX;   ImmToIdxMap[PPC::LHA]  = PPC::LHAX;
  ImmToIdxMap[PPC::LWZ]  = PPC::LWZX;   ImmToIdxMap[PPC::LWA]  = PPC::LWAX;
  ImmToIdxMap[PPC::LFS]  = PPC::LFSX;   ImmToIdxMap[PPC::LFD]  = PPC::LFDX;
  ImmToIdxMap[PPC::STH]  = PPC::STHX;   ImmToIdxMap[PPC::STW]  = PPC::STWX;
  ImmToIdxMap[PPC::STFS] = PPC::STFSX;  ImmToIdxMap[PPC::STFD] = PPC::STFDX;
  ImmToIdxMap[PPC::ADDI] = PPC::ADD4;
}

void
PPCRegisterInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MI,
                                     unsigned SrcReg, int FrameIdx,
                                     const TargetRegisterClass *RC) const {
  if (SrcReg == PPC::LR) {
    BuildMI(MBB, MI, PPC::MFLR, 1, PPC::R11);
    addFrameReference(BuildMI(MBB, MI, PPC::STW, 3).addReg(PPC::R11), FrameIdx);
  } else if (RC == PPC::CRRCRegisterClass) {
    BuildMI(MBB, MI, PPC::MFCR, 0, PPC::R11);
    addFrameReference(BuildMI(MBB, MI, PPC::STW, 3).addReg(PPC::R11), FrameIdx);
  } else if (RC == PPC::GPRCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::STW, 3).addReg(SrcReg),FrameIdx);
  } else if (RC == PPC::G8RCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::STD, 3).addReg(SrcReg),FrameIdx);
  } else if (RC == PPC::F8RCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::STFD, 3).addReg(SrcReg),FrameIdx);
  } else if (RC == PPC::F4RCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::STFS, 3).addReg(SrcReg),FrameIdx);
  } else {
    assert(0 && "Unknown regclass!");
    abort();
  }
}

void
PPCRegisterInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        unsigned DestReg, int FrameIdx,
                                        const TargetRegisterClass *RC) const {
  if (DestReg == PPC::LR) {
    addFrameReference(BuildMI(MBB, MI, PPC::LWZ, 2, PPC::R11), FrameIdx);
    BuildMI(MBB, MI, PPC::MTLR, 1).addReg(PPC::R11);
  } else if (RC == PPC::CRRCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::LWZ, 2, PPC::R11), FrameIdx);
    BuildMI(MBB, MI, PPC::MTCRF, 1, DestReg).addReg(PPC::R11);
  } else if (RC == PPC::GPRCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::LWZ, 2, DestReg), FrameIdx);
  } else if (RC == PPC::G8RCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::LD, 2, DestReg), FrameIdx);
  } else if (RC == PPC::F8RCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::LFD, 2, DestReg), FrameIdx);
  } else if (RC == PPC::F4RCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::LFS, 2, DestReg), FrameIdx);
  } else {
    assert(0 && "Unknown regclass!");
    abort();
  }
}

void PPCRegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned DestReg, unsigned SrcReg,
                                   const TargetRegisterClass *RC) const {
  MachineInstr *I;

  if (RC == PPC::GPRCRegisterClass) {
    BuildMI(MBB, MI, PPC::OR4, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
  } else if (RC == PPC::G8RCRegisterClass) {
    BuildMI(MBB, MI, PPC::OR8, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
  } else if (RC == PPC::F4RCRegisterClass) {
    BuildMI(MBB, MI, PPC::FMRS, 1, DestReg).addReg(SrcReg);
  } else if (RC == PPC::F8RCRegisterClass) {
    BuildMI(MBB, MI, PPC::FMRD, 1, DestReg).addReg(SrcReg);
  } else if (RC == PPC::CRRCRegisterClass) {
    BuildMI(MBB, MI, PPC::MCRF, 1, DestReg).addReg(SrcReg);
  } else {
    std::cerr << "Attempt to copy register that is not GPR or FPR";
    abort();
  }
}

unsigned PPCRegisterInfo::isLoadFromStackSlot(MachineInstr *MI, 
                                              int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case PPC::LD:
  case PPC::LWZ:
  case PPC::LFS:
  case PPC::LFD:
    if (MI->getOperand(1).isImmediate() && !MI->getOperand(1).getImmedValue() &&
        MI->getOperand(2).isFrameIndex()) {
      FrameIndex = MI->getOperand(2).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

/// foldMemoryOperand - PowerPC (like most RISC's) can only fold spills into
/// copy instructions, turning them into load/store instructions.
MachineInstr *PPCRegisterInfo::foldMemoryOperand(MachineInstr *MI,
                                                 unsigned OpNum,
                                                 int FrameIndex) const {
  // Make sure this is a reg-reg copy.  Note that we can't handle MCRF, because
  // it takes more than one instruction to store it.
  unsigned Opc = MI->getOpcode();
  
  if ((Opc == PPC::OR4 &&
       MI->getOperand(1).getReg() == MI->getOperand(2).getReg())) {
    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      return addFrameReference(BuildMI(PPC::STW,
                                       3).addReg(InReg), FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      return addFrameReference(BuildMI(PPC::LWZ, 2, OutReg), FrameIndex);
    }
  } else if ((Opc == PPC::OR8 &&
              MI->getOperand(1).getReg() == MI->getOperand(2).getReg())) {
    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      return addFrameReference(BuildMI(PPC::STD,
                                       3).addReg(InReg), FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      return addFrameReference(BuildMI(PPC::LD, 2, OutReg), FrameIndex);
    }
  } else if (Opc == PPC::FMRD) {
    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      return addFrameReference(BuildMI(PPC::STFD,
                                       3).addReg(InReg), FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      return addFrameReference(BuildMI(PPC::LFD, 2, OutReg), FrameIndex);
    }
  } else if (Opc == PPC::FMRS) {
    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      return addFrameReference(BuildMI(PPC::STFS,
                                       3).addReg(InReg), FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      return addFrameReference(BuildMI(PPC::LFS, 2, OutReg), FrameIndex);
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
//
static bool hasFP(MachineFunction &MF) {
  return NoFramePointerElim || MF.getFrameInfo()->hasVarSizedObjects();
}

void PPCRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (hasFP(MF)) {
    // If we have a frame pointer, convert as follows:
    // ADJCALLSTACKDOWN -> addi, r1, r1, -amount
    // ADJCALLSTACKUP   -> addi, r1, r1, amount
    MachineInstr *Old = I;
    unsigned Amount = Old->getOperand(0).getImmedValue();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      // Replace the pseudo instruction with a new instruction...
      if (Old->getOpcode() == PPC::ADJCALLSTACKDOWN) {
        MBB.insert(I, BuildMI(PPC::ADDI, 2, PPC::R1).addReg(PPC::R1)
                .addSImm(-Amount));
      } else {
        assert(Old->getOpcode() == PPC::ADJCALLSTACKUP);
        MBB.insert(I, BuildMI(PPC::ADDI, 2, PPC::R1).addReg(PPC::R1)
                .addSImm(Amount));
      }
    }
  }
  MBB.erase(I);
}

void
PPCRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II) const {
  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();

  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getFrameIndex();

  // Replace the FrameIndex with base register with GPR1 (SP) or GPR31 (FP).
  MI.SetMachineOperandReg(i, hasFP(MF) ? PPC::R31 : PPC::R1);

  // Take into account whether it's an add or mem instruction
  unsigned OffIdx = (i == 2) ? 1 : 2;

  // Now add the frame object offset to the offset from r1.
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
               MI.getOperand(OffIdx).getImmedValue();

  // If we're not using a Frame Pointer that has been set to the value of the
  // SP before having the stack size subtracted from it, then add the stack size
  // to Offset to get the correct offset.
  Offset += MF.getFrameInfo()->getStackSize();

  if (Offset > 32767 || Offset < -32768) {
    // Insert a set of r0 with the full offset value before the ld, st, or add
    MachineBasicBlock *MBB = MI.getParent();
    MBB->insert(II, BuildMI(PPC::LIS, 1, PPC::R0).addSImm(Offset >> 16));
    MBB->insert(II, BuildMI(PPC::ORI, 2, PPC::R0).addReg(PPC::R0)
      .addImm(Offset));
    // convert into indexed form of the instruction
    // sth 0:rA, 1:imm 2:(rB) ==> sthx 0:rA, 2:rB, 1:r0
    // addi 0:rA 1:rB, 2, imm ==> add 0:rA, 1:rB, 2:r0
    assert(ImmToIdxMap.count(MI.getOpcode()) &&
           "No indexed form of load or store available!");
    unsigned NewOpcode = ImmToIdxMap.find(MI.getOpcode())->second;
    MI.setOpcode(NewOpcode);
    MI.SetMachineOperandReg(1, MI.getOperand(i).getReg());
    MI.SetMachineOperandReg(2, PPC::R0);
  } else {
    MI.SetMachineOperandConst(OffIdx, MachineOperand::MO_SignExtendedImmed,
                              Offset);
  }
}


void PPCRegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineInstr *MI;

  // Get the number of bytes to allocate from the FrameInfo
  unsigned NumBytes = MFI->getStackSize();

  // If we have calls, we cannot use the red zone to store callee save registers
  // and we must set up a stack frame, so calculate the necessary size here.
  if (MFI->hasCalls()) {
    // We reserve argument space for call sites in the function immediately on
    // entry to the current function.  This eliminates the need for add/sub
    // brackets around call sites.
    NumBytes += MFI->getMaxCallFrameSize();
  }

  // If we are a leaf function, and use up to 224 bytes of stack space,
  // and don't have a frame pointer, then we do not need to adjust the stack
  // pointer (we fit in the Red Zone).
  if ((NumBytes == 0) || (NumBytes <= 224 && !hasFP(MF) && !MFI->hasCalls())) {
    MFI->setStackSize(0);
    return;
  }

  // Add the size of R1 to  NumBytes size for the store of R1 to the bottom
  // of the stack and round the size to a multiple of the alignment.
  unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
  unsigned GPRSize = 4;
  unsigned Size = hasFP(MF) ? GPRSize + GPRSize : GPRSize;
  NumBytes = (NumBytes+Size+Align-1)/Align*Align;

  // Update frame info to pretend that this is part of the stack...
  MFI->setStackSize(NumBytes);

  // Adjust stack pointer: r1 -= numbytes.
  if (NumBytes <= 32768) {
    MI=BuildMI(PPC::STWU,3).addReg(PPC::R1).addSImm(-NumBytes).addReg(PPC::R1);
    MBB.insert(MBBI, MI);
  } else {
    int NegNumbytes = -NumBytes;
    MI = BuildMI(PPC::LIS, 1, PPC::R0).addSImm(NegNumbytes >> 16);
    MBB.insert(MBBI, MI);
    MI = BuildMI(PPC::ORI, 2, PPC::R0).addReg(PPC::R0)
      .addImm(NegNumbytes & 0xFFFF);
    MBB.insert(MBBI, MI);
    MI = BuildMI(PPC::STWUX, 3).addReg(PPC::R1).addReg(PPC::R1).addReg(PPC::R0);
    MBB.insert(MBBI, MI);
  }

  if (hasFP(MF)) {
    MI = BuildMI(PPC::STW, 3).addReg(PPC::R31).addSImm(GPRSize).addReg(PPC::R1);
    MBB.insert(MBBI, MI);
    MI = BuildMI(PPC::OR4, 2, PPC::R31).addReg(PPC::R1).addReg(PPC::R1);
    MBB.insert(MBBI, MI);
  }
}

void PPCRegisterInfo::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  MachineInstr *MI;
  assert(MBBI->getOpcode() == PPC::BLR &&
         "Can only insert epilog into returning blocks");

  // Get the number of bytes allocated from the FrameInfo...
  unsigned NumBytes = MFI->getStackSize();
  unsigned GPRSize = 4;

  if (NumBytes != 0) {
    if (hasFP(MF)) {
      MI = BuildMI(PPC::LWZ, 2, PPC::R31).addSImm(GPRSize).addReg(PPC::R31);
      MBB.insert(MBBI, MI);
    }
    MI = BuildMI(PPC::LWZ, 2, PPC::R1).addSImm(0).addReg(PPC::R1);
    MBB.insert(MBBI, MI);
  }
}

#include "PPCGenRegisterInfo.inc"

