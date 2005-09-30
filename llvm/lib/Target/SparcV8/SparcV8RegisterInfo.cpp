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

static const TargetRegisterClass *getClass(unsigned SrcReg) {
  if (V8::IntRegsRegisterClass->contains(SrcReg))
    return V8::IntRegsRegisterClass;
  else if (V8::FPRegsRegisterClass->contains(SrcReg))
    return V8::FPRegsRegisterClass;
  else if (V8::DFPRegsRegisterClass->contains(SrcReg))
    return V8::DFPRegsRegisterClass;
  else {
    std::cerr << "Error: register of unknown class found: " << SrcReg << "\n";
    abort ();
  }
}

void SparcV8RegisterInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, int FrameIdx,
                    const TargetRegisterClass *rc) const {
  const TargetRegisterClass *RC = getClass(SrcReg);

  // On the order of operands here: think "[FrameIdx + 0] = SrcReg".
  if (RC == V8::IntRegsRegisterClass)
    BuildMI (MBB, I, V8::ST, 3).addFrameIndex (FrameIdx).addSImm (0)
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
                     const TargetRegisterClass *rc) const {
  const TargetRegisterClass *RC = getClass(DestReg);
  if (RC == V8::IntRegsRegisterClass)
    BuildMI (MBB, I, V8::LD, 2, DestReg).addFrameIndex (FrameIdx).addSImm (0);
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
  int size = MI.getOperand (0).getImmedValue ();
  if (MI.getOpcode () == V8::ADJCALLSTACKDOWN)
    size = -size;
  BuildMI (MBB, I, V8::ADDri, 2, V8::SP).addReg (V8::SP).addSImm (size);
  MBB.erase (I);
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

  // Replace frame index with a frame pointer reference
  MI.SetMachineOperandReg (i, V8::FP);

  // Addressable stack objects are accessed using neg. offsets from %fp
  MachineFunction &MF = *MI.getParent()->getParent();
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
               MI.getOperand(i+1).getImmedValue();
  // note: Offset < 0
  MI.SetMachineOperandConst (i+1, MachineOperand::MO_SignExtendedImmed, Offset);
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
  BuildMI(MBB, MBB.begin(), V8::SAVEri, 2,
          V8::SP).addImm(-NumBytes).addReg(V8::SP);
}

void SparcV8RegisterInfo::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  assert(MBBI->getOpcode() == V8::RETL &&
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

