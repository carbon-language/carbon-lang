//===- SkeletonRegisterInfo.cpp - Skeleton Register Information -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Skeleton implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "Skeleton.h"
#include "SkeletonRegisterInfo.h"
#include "llvm/Type.h"
using namespace llvm;

SkeletonRegisterInfo::SkeletonRegisterInfo()
  : SkeletonGenRegisterInfo(Skeleton::ADJCALLSTACKDOWN,
                           Skeleton::ADJCALLSTACKUP) {}

void SkeletonRegisterInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                    unsigned SrcReg, int FrameIdx) const {
  abort();
}

void SkeletonRegisterInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                     unsigned DestReg, int FrameIdx) const {
  abort();
}

void SkeletonRegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        unsigned DestReg, unsigned SrcReg,
                                        const TargetRegisterClass *RC) const {
  abort();
}

void SkeletonRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  abort();
}

void SkeletonRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II)
  const {
  abort();
}

void SkeletonRegisterInfo::
processFunctionBeforeFrameFinalized(MachineFunction &MF) const {
  abort();
}

void SkeletonRegisterInfo::emitPrologue(MachineFunction &MF) const {
  abort();
}

void SkeletonRegisterInfo::emitEpilogue(MachineFunction &MF,
                                        MachineBasicBlock &MBB) const {
  abort();
}


#include "SkeletonGenRegisterInfo.inc"

const TargetRegisterClass*
SkeletonRegisterInfo::getRegClassForType(const Type* Ty) const {
  switch (Ty->getTypeID()) {
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

