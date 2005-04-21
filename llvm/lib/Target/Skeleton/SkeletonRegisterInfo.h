//===- SkeletonRegisterInfo.h - Skeleton Register Information Impl -*- C++ -*-==//
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

#ifndef SKELETON_REGISTERINFO_H
#define SKELETON_REGISTERINFO_H

#include "SkeletonGenRegisterInfo.h.inc"

namespace llvm {
  class Type;

  struct SkeletonRegisterInfo : public SkeletonGenRegisterInfo {
    SkeletonRegisterInfo();
    const TargetRegisterClass* getRegClassForType(const Type* Ty) const;

    void storeRegToStackSlot(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MBBI,
                             unsigned SrcReg, int FrameIndex) const;

    void loadRegFromStackSlot(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              unsigned DestReg, int FrameIndex) const;

    void copyRegToReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                      unsigned DestReg, unsigned SrcReg,
                      const TargetRegisterClass *RC) const;

    void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                       MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I) const;

    void eliminateFrameIndex(MachineBasicBlock::iterator II) const;

    void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;

    void emitPrologue(MachineFunction &MF) const;
    void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
  };
} // end namespace llvm

#endif
