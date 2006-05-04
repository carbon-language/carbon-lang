//===- SparcRegisterInfo.cpp - SPARC Register Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SPARC implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "Sparc.h"
#include "SparcRegisterInfo.h"
#include "SparcSubtarget.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Type.h"
#include "llvm/ADT/STLExtras.h"
#include <iostream>
using namespace llvm;

SparcRegisterInfo::SparcRegisterInfo(SparcSubtarget &st)
  : SparcGenRegisterInfo(SP::ADJCALLSTACKDOWN, SP::ADJCALLSTACKUP),
    Subtarget(st) {
}

void SparcRegisterInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, int FI,
                    const TargetRegisterClass *RC) const {
  // On the order of operands here: think "[FrameIdx + 0] = SrcReg".
  if (RC == SP::IntRegsRegisterClass)
    BuildMI(MBB, I, SP::STri, 3).addFrameIndex(FI).addImm(0).addReg(SrcReg);
  else if (RC == SP::FPRegsRegisterClass)
    BuildMI(MBB, I, SP::STFri, 3).addFrameIndex(FI).addImm(0).addReg(SrcReg);
  else if (RC == SP::DFPRegsRegisterClass)
    BuildMI(MBB, I, SP::STDFri, 3).addFrameIndex(FI).addImm(0).addReg(SrcReg);
  else
    assert(0 && "Can't store this register to stack slot");
}

void SparcRegisterInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const {
  if (RC == SP::IntRegsRegisterClass)
    BuildMI(MBB, I, SP::LDri, 2, DestReg).addFrameIndex(FI).addImm(0);
  else if (RC == SP::FPRegsRegisterClass)
    BuildMI(MBB, I, SP::LDFri, 2, DestReg).addFrameIndex(FI).addImm (0);
  else if (RC == SP::DFPRegsRegisterClass)
    BuildMI(MBB, I, SP::LDDFri, 2, DestReg).addFrameIndex(FI).addImm(0);
  else
    assert(0 && "Can't load this register from stack slot");
}

void SparcRegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I,
                                     unsigned DestReg, unsigned SrcReg,
                                     const TargetRegisterClass *RC) const {
  if (RC == SP::IntRegsRegisterClass)
    BuildMI(MBB, I, SP::ORrr, 2, DestReg).addReg(SP::G0).addReg(SrcReg);
  else if (RC == SP::FPRegsRegisterClass)
    BuildMI(MBB, I, SP::FMOVS, 1, DestReg).addReg(SrcReg);
  else if (RC == SP::DFPRegsRegisterClass)
    BuildMI(MBB, I, Subtarget.isV9() ? SP::FMOVD : SP::FpMOVD,
            1, DestReg).addReg(SrcReg);
  else
    assert (0 && "Can't copy this register");
}

MachineInstr *SparcRegisterInfo::foldMemoryOperand(MachineInstr* MI,
                                                   unsigned OpNum,
                                                   int FI) const {
  bool isFloat = false;
  switch (MI->getOpcode()) {
  case SP::ORrr:
    if (MI->getOperand(1).isRegister() && MI->getOperand(1).getReg() == SP::G0&&
        MI->getOperand(0).isRegister() && MI->getOperand(2).isRegister()) {
      if (OpNum == 0)    // COPY -> STORE
        return BuildMI(SP::STri, 3).addFrameIndex(FI).addImm(0)
                                   .addReg(MI->getOperand(2).getReg());
      else               // COPY -> LOAD
        return BuildMI(SP::LDri, 2, MI->getOperand(0).getReg())
                      .addFrameIndex(FI).addImm(0);
    }
    break;
  case SP::FMOVS:
    isFloat = true;
    // FALLTHROUGH
  case SP::FMOVD:
    if (OpNum == 0)  // COPY -> STORE
      return BuildMI(isFloat ? SP::STFri : SP::STDFri, 3)
               .addFrameIndex(FI).addImm(0).addReg(MI->getOperand(1).getReg());
    else             // COPY -> LOAD
      return BuildMI(isFloat ? SP::LDFri : SP::LDDFri, 2, 
                     MI->getOperand(0).getReg()).addFrameIndex(FI).addImm(0);
    break;
  }
  return 0;
}

void SparcRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  MachineInstr &MI = *I;
  int Size = MI.getOperand(0).getImmedValue();
  if (MI.getOpcode() == SP::ADJCALLSTACKDOWN)
    Size = -Size;
  if (Size)
    BuildMI(MBB, I, SP::ADDri, 2, SP::O6).addReg(SP::O6).addImm(Size);
  MBB.erase(I);
}

void
SparcRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II) const {
  unsigned i = 0;
  MachineInstr &MI = *II;
  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getFrameIndex();

  // Addressable stack objects are accessed using neg. offsets from %fp
  MachineFunction &MF = *MI.getParent()->getParent();
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
               MI.getOperand(i+1).getImmedValue();

  // Replace frame index with a frame pointer reference.
  if (Offset >= -4096 && Offset <= 4095) {
    // If the offset is small enough to fit in the immediate field, directly
    // encode it.
    MI.SetMachineOperandReg(i, SP::I6);
    MI.SetMachineOperandConst(i+1, MachineOperand::MO_Immediate, Offset);
  } else {
    // Otherwise, emit a G1 = SETHI %hi(offset).  FIXME: it would be better to 
    // scavenge a register here instead of reserving G1 all of the time.
    unsigned OffHi = (unsigned)Offset >> 10U;
    BuildMI(*MI.getParent(), II, SP::SETHIi, 1, SP::G1).addImm(OffHi);
    // Emit G1 = G1 + I6
    BuildMI(*MI.getParent(), II, SP::ADDrr, 2, 
            SP::G1).addReg(SP::G1).addReg(SP::I6);
    // Insert: G1+%lo(offset) into the user.
    MI.SetMachineOperandReg(i, SP::G1);
    MI.SetMachineOperandConst(i+1, MachineOperand::MO_Immediate,
                              Offset & ((1 << 10)-1));
  }
}

void SparcRegisterInfo::
processFunctionBeforeFrameFinalized(MachineFunction &MF) const {}

void SparcRegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  // Get the number of bytes to allocate from the FrameInfo
  int NumBytes = (int) MFI->getStackSize();

  // Emit the correct save instruction based on the number of bytes in
  // the frame. Minimum stack frame size according to V8 ABI is:
  //   16 words for register window spill
  //    1 word for address of returned aggregate-value
  // +  6 words for passing parameters on the stack
  // ----------
  //   23 words * 4 bytes per word = 92 bytes
  NumBytes += 92;
  // Round up to next doubleword boundary -- a double-word boundary
  // is required by the ABI.
  NumBytes = (NumBytes + 7) & ~7;
  NumBytes = -NumBytes;
  
  if (NumBytes >= -4096) {
    BuildMI(MBB, MBB.begin(), SP::SAVEri, 2,
            SP::O6).addImm(NumBytes).addReg(SP::O6);
  } else {
    MachineBasicBlock::iterator InsertPt = MBB.begin();
    // Emit this the hard way.  This clobbers G1 which we always know is 
    // available here.
    unsigned OffHi = (unsigned)NumBytes >> 10U;
    BuildMI(MBB, InsertPt, SP::SETHIi, 1, SP::G1).addImm(OffHi);
    // Emit G1 = G1 + I6
    BuildMI(MBB, InsertPt, SP::ORri, 2, SP::G1)
      .addReg(SP::G1).addImm(NumBytes & ((1 << 10)-1));
    BuildMI(MBB, InsertPt, SP::SAVErr, 2,
            SP::O6).addReg(SP::O6).addReg(SP::G1);
  }
}

void SparcRegisterInfo::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  assert(MBBI->getOpcode() == SP::RETL &&
         "Can only put epilog before 'retl' instruction!");
  BuildMI(MBB, MBBI, SP::RESTORErr, 2, SP::G0).addReg(SP::G0).addReg(SP::G0);
}

unsigned SparcRegisterInfo::getRARegister() const {
  assert(0 && "What is the return address register");
  return 0;
}

unsigned SparcRegisterInfo::getFrameRegister(MachineFunction &MF) const {
  assert(0 && "What is the frame register");
  return SP::G1;
}

#include "SparcGenRegisterInfo.inc"

