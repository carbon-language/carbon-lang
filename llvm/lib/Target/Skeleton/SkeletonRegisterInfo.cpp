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

