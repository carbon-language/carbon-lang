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
#include "Support/STLExtras.h"
using namespace llvm;

SparcV8RegisterInfo::SparcV8RegisterInfo()
  : SparcV8GenRegisterInfo(V8::ADJCALLSTACKDOWN,
                           V8::ADJCALLSTACKUP) {}

int SparcV8RegisterInfo::storeRegToStackSlot(
  MachineBasicBlock &MBB,
  MachineBasicBlock::iterator MBBI,
  unsigned SrcReg, int FrameIdx,
  const TargetRegisterClass *RC) const
{
  assert (RC == SparcV8::IntRegsRegisterClass
          && "Can only store 32-bit values to stack slots");
  MachineInstr *I =
    BuildMI (V8::STrm, 3).addFrameIndex (FrameIdx).addSImm (0).addReg (SrcReg);
  MBB.insert(MBBI, I);
  return 1;
}

int SparcV8RegisterInfo::loadRegFromStackSlot(
  MachineBasicBlock &MBB,
  MachineBasicBlock::iterator I,
  unsigned DestReg, int FrameIdx,
  const TargetRegisterClass *RC) const
{
  assert (RC == SparcV8::IntRegsRegisterClass
          && "Can only load 32-bit registers from stack slots");
  BuildMI (MBB, I, V8::LDmr, 2, DestReg).addFrameIndex (FrameIdx).addSImm (0);
  return 1;
}

int SparcV8RegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator I,
                                      unsigned DestReg, unsigned SrcReg,
                                      const TargetRegisterClass *RC) const {
  assert (RC == SparcV8::IntRegsRegisterClass
          && "Can only copy 32-bit registers");
  BuildMI (MBB, I, V8::ORrr, 2, DestReg).addReg (V8::G0).addReg (SrcReg);
  return -1;
}

void SparcV8RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  std::cerr
    << "Sorry, I don't know how to eliminate call frame pseudo instrs yet, in\n"
    << __FUNCTION__ << " at " << __FILE__ << ":" << __LINE__ << "\n";
  abort();
}

void
SparcV8RegisterInfo::eliminateFrameIndex(MachineFunction &MF,
                                         MachineBasicBlock::iterator II) const {
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

  // Emit the correct save instruction based on the number of bytes in the frame.
  // Minimum stack frame size according to V8 ABI is:
  //   16 words for register window spill
  //    1 word for address of returned aggregate-value
  // +  6 words for passing parameters on the stack
  // ----------
  //   23 words * 4 bytes per word = 92 bytes
  NumBytes += 92;
  NumBytes = (NumBytes + 7) & ~7;  // Round up to next doubleword boundary
   // (Technically, a word boundary should be sufficient, but SPARC as complains)
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
  switch (Ty->getPrimitiveID()) {
  case Type::FloatTyID:  return &FPRegsInstance;
  case Type::DoubleTyID: return &DFPRegsInstance;
  case Type::LongTyID:
  case Type::ULongTyID: assert(0 && "Long values can't fit in registers!");
  default:              assert(0 && "Invalid type to getClass!");
  case Type::BoolTyID:
  case Type::SByteTyID:
  case Type::UByteTyID:
  case Type::ShortTyID:
  case Type::UShortTyID:
  case Type::IntTyID:
  case Type::UIntTyID:
  case Type::PointerTyID: return &IntRegsInstance;
  }
}

