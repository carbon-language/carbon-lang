//===-- MipsSERegisterInfo.h - Mips32/64 Register Information ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips32/64 implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSSEREGISTERINFO_H
#define MIPSSEREGISTERINFO_H

#include "MipsRegisterInfo.h"

namespace llvm {

class MipsSERegisterInfo : public MipsRegisterInfo {
public:
  MipsSERegisterInfo(const MipsSubtarget &Subtarget,
                     const TargetInstrInfo &TII);

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

private:
  virtual void eliminateFI(MachineBasicBlock::iterator II, unsigned OpNo,
                           int FrameIndex, uint64_t StackSize,
                           int64_t SPOffset) const;
};

} // end namespace llvm

#endif
