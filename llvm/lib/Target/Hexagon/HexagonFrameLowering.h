//=- HexagonFrameLowering.h - Define frame lowering for Hexagon --*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_FRAMEINFO_H
#define HEXAGON_FRAMEINFO_H

#include "Hexagon.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {

class HexagonFrameLowering : public TargetFrameLowering {
private:
  void determineFrameLayout(MachineFunction &MF) const;

public:
  explicit HexagonFrameLowering() : TargetFrameLowering(StackGrowsDown, 8, 0) {}

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI,
                                 const TargetRegisterInfo *TRI) const override;

  void
  eliminateCallFramePseudoInstr(MachineFunction &MF,
                                MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I) const override;

  bool
  restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI,
                              const std::vector<CalleeSavedInfo> &CSI,
                              const TargetRegisterInfo *TRI) const override;
  int getFrameIndexOffset(const MachineFunction &MF, int FI) const override;
  bool hasFP(const MachineFunction &MF) const override;
  bool hasTailCall(MachineBasicBlock &MBB) const;
};

} // End llvm namespace

#endif
