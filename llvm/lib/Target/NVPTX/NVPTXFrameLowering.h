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

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXFRAMELOWERING_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXFRAMELOWERING_H

#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
class NVPTXSubtarget;
class NVPTXFrameLowering : public TargetFrameLowering {
  bool is64bit;

public:
  explicit NVPTXFrameLowering(NVPTXSubtarget &STI);

  bool hasFP(const MachineFunction &MF) const override;
  void emitPrologue(MachineFunction &MF) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                  MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I) const override;
};

} // End llvm namespace

#endif
