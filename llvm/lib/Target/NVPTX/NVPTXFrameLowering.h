//===--- NVPTXFrameLowering.h - Define frame lowering for NVPTX -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef NVPTX_FRAMELOWERING_H
#define NVPTX_FRAMELOWERING_H

#include "NVPTXSubtarget.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {

class NVPTXFrameLowering : public TargetFrameLowering {
  bool is64bit;

public:
  explicit NVPTXFrameLowering(NVPTXSubtarget &STI)
      : TargetFrameLowering(TargetFrameLowering::StackGrowsUp, 8, 0),
        is64bit(STI.is64Bit()) {}

  bool hasFP(const MachineFunction &MF) const override;
  void emitPrologue(MachineFunction &MF) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                  MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I) const override;
};

} // End llvm namespace

#endif
