//===- SparcV8RegisterInfo.cpp - SparcV8 Register Information ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SparcV8 implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "SparcV8.h"
#include "SparcV8RegisterInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Type.h"
#include "llvm/ADT/STLExtras.h"
#include <iostream>
using namespace llvm;

SparcV8RegisterInfo::SparcV8RegisterInfo()
  : SparcV8GenRegisterInfo(V8::ADJCALLSTACKDOWN,
                           V8::ADJCALLSTACKUP) {}

void SparcV8RegisterInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, int FrameIdx,
                    const TargetRegisterClass *RC) const {
  // On the order of operands here: think "[FrameIdx + 0] = SrcReg".
  if (RC == V8::IntRegsRegisterClass)
    BuildMI (MBB, I, V8::STri, 3).addFrameIndex (FrameIdx).addSImm (0)
      .addReg (SrcReg);
  else if (RC == V8::FPRegsRegisterClass)
    BuildMI (MBB, I, V8::STFri, 3).addFrameIndex (FrameIdx).addSImm (0)
      .addReg (SrcReg);
  else if (RC == V8::DFPRegsRegisterClass)
    BuildMI (MBB, I, V8::STDFri, 3).addFrameIndex (FrameIdx).addSImm (0)
      .addReg (SrcReg);
  else
    assert (0 && "Can't store this register to stack slot");
}

void SparcV8RegisterInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FrameIdx,
                     const TargetRegisterClass *RC) const {
  if (RC == V8::IntRegsRegisterClass)
    BuildMI (MBB, I, V8::LDri, 2, DestReg).addFrameIndex (FrameIdx).addSImm (0);
  else if (RC == V8::FPRegsRegisterClass)
    BuildMI (MBB, I, V8::LDFri, 2, DestReg).addFrameIndex (FrameIdx)
      .addSImm (0);
  else if (RC == V8::DFPRegsRegisterClass)
    BuildMI (MBB, I, V8::LDDFri, 2, DestReg).addFrameIndex (FrameIdx)
      .addSImm (0);
  else
    assert(0 && "Can't load this register from stack slot");
}

void SparcV8RegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       unsigned DestReg, unsigned SrcReg,
                                       const TargetRegisterClass *RC) const {
  if (RC == V8::IntRegsRegisterClass)
    BuildMI (MBB, I, V8::ORrr, 2, DestReg).addReg (V8::G0).addReg (SrcReg);
  else if (RC == V8::FPRegsRegisterClass)
    BuildMI (MBB, I, V8::FMOVS, 1, DestReg).addReg (SrcReg);
  else if (RC == V8::DFPRegsRegisterClass)
    BuildMI (MBB, I, V8::FpMOVD, 1, DestReg).addReg (SrcReg);
  else
    assert (0 && "Can't copy this register");
}

void SparcV8RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  MachineInstr &MI = *I;
  int Size = MI.getOperand(0).getImmedValue();
  if (MI.getOpcode() == V8::ADJCALLSTACKDOWN)
    Size = -Size;
  if (Size)
    BuildMI(MBB, I, V8::ADDri, 2, V8::O6).addReg(V8::O6).addSImm(Size);
  MBB.erase(I);
}

void
SparcV8RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II) const {
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
    MI.SetMachineOperandReg(i, V8::I6);
    MI.SetMachineOperandConst(i+1, MachineOperand::MO_SignExtendedImmed,Offset);
  } else {
    // Otherwise, emit a G1 = SETHI %hi(offset).  FIXME: it would be better to 
    // scavenge a register here instead of reserving G1 all of the time.
    unsigned OffHi = (unsigned)Offset >> 10U;
    BuildMI(*MI.getParent(), II, V8::SETHIi, 1, V8::G1).addImm(OffHi);
    // Emit G1 = G1 + I6
    BuildMI(*MI.getParent(), II, V8::ADDrr, 2, 
            V8::G1).addReg(V8::G1).addReg(V8::I6);
    // Insert: G1+%lo(offset) into the user.
    MI.SetMachineOperandReg(i, V8::I1);
    MI.SetMachineOperandConst(i+1, MachineOperand::MO_SignExtendedImmed,
                              Offset & ((1 << 10)-1));
  }
}

void SparcV8RegisterInfo::
processFunctionBeforeFrameFinalized(MachineFunction &MF) const {}

void SparcV8RegisterInfo::emitPrologue(MachineFunction &MF) const {
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
    BuildMI(MBB, MBB.begin(), V8::SAVEri, 2,
            V8::O6).addImm(NumBytes).addReg(V8::O6);
  } else {
    MachineBasicBlock::iterator InsertPt = MBB.begin();
    // Emit this the hard way.  This clobbers G1 which we always know is 
    // available here.
    unsigned OffHi = (unsigned)NumBytes >> 10U;
    BuildMI(MBB, InsertPt, V8::SETHIi, 1, V8::G1).addImm(OffHi);
    // Emit G1 = G1 + I6
    BuildMI(MBB, InsertPt, V8::ORri, 2, V8::G1)
      .addReg(V8::G1).addImm(NumBytes & ((1 << 10)-1));
    BuildMI(MBB, InsertPt, V8::SAVErr, 2,
            V8::O6).addReg(V8::O6).addReg(V8::G1);
  }
}

void SparcV8RegisterInfo::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  // FIXME: RETVOID should be removed. See SparcV8InstrInfo.td
  assert((MBBI->getOpcode() == V8::RETL || MBBI->getOpcode() == V8::RETVOID) &&
         "Can only put epilog before 'retl' instruction!");
  BuildMI(MBB, MBBI, V8::RESTORErr, 2, V8::G0).addReg(V8::G0).addReg(V8::G0);
}

#include "SparcV8GenRegisterInfo.inc"

const TargetRegisterClass*
SparcV8RegisterInfo::getRegClassForType(const Type* Ty) const {
  switch (Ty->getTypeID()) {
  case Type::FloatTyID:  return V8::FPRegsRegisterClass;
  case Type::DoubleTyID: return V8::DFPRegsRegisterClass;
  case Type::LongTyID:
  case Type::ULongTyID:  assert(0 && "Long values do not fit in registers!");
  default:               assert(0 && "Invalid type to getClass!");
  case Type::BoolTyID:
  case Type::SByteTyID:
  case Type::UByteTyID:
  case Type::ShortTyID:
  case Type::UShortTyID:
  case Type::IntTyID:
  case Type::UIntTyID:
  case Type::PointerTyID: return V8::IntRegsRegisterClass;
  }
}

