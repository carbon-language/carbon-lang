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
#include "llvm/Type.h"
using namespace llvm;

SparcV8RegisterInfo::SparcV8RegisterInfo()
  : SparcV8GenRegisterInfo(SparcV8::ADJCALLSTACKDOWN,
                           SparcV8::ADJCALLSTACKUP) {}

int SparcV8RegisterInfo::storeRegToStackSlot(
  MachineBasicBlock &MBB,
  MachineBasicBlock::iterator MBBI,
  unsigned SrcReg, int FrameIdx,
  const TargetRegisterClass *RC) const
{
  abort();
  return -1;
}

int SparcV8RegisterInfo::loadRegFromStackSlot(
  MachineBasicBlock &MBB,
  MachineBasicBlock::iterator MBBI,
  unsigned DestReg, int FrameIdx,
  const TargetRegisterClass *RC) const
{
  abort();
  return -1;
}

int SparcV8RegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI,
                                      unsigned DestReg, unsigned SrcReg,
                                      const TargetRegisterClass *RC) const {
  abort();
  return -1;
}

void SparcV8RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  abort();
}

void
SparcV8RegisterInfo::eliminateFrameIndex(MachineFunction &MF,
                                         MachineBasicBlock::iterator II) const {
  abort();
}

void SparcV8RegisterInfo::processFunctionBeforeFrameFinalized(
    MachineFunction &MF) const {
  abort();
}

void SparcV8RegisterInfo::emitPrologue(MachineFunction &MF) const {
  abort();
}

void SparcV8RegisterInfo::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  abort();
}


#include "SparcV8GenRegisterInfo.inc"

const TargetRegisterClass*
SparcV8RegisterInfo::getRegClassForType(const Type* Ty) const {
  switch (Ty->getPrimitiveID()) {
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
  case Type::PointerTyID: return &GPRCInstance;
    
  case Type::FloatTyID:
  case Type::DoubleTyID: return &FPRCInstance;
  }
}

