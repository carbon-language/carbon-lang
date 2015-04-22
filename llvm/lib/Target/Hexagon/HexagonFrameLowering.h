//=- HexagonFrameLowering.h - Define frame lowering for Hexagon --*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONFRAMELOWERING_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONFRAMELOWERING_H

#include "Hexagon.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {

class HexagonInstrInfo;

class HexagonFrameLowering : public TargetFrameLowering {
private:
  void expandAlloca(MachineInstr *AI, const HexagonInstrInfo &TII,
                    unsigned SP, unsigned CF) const;

public:
  explicit HexagonFrameLowering()
      : TargetFrameLowering(StackGrowsDown, 8, 0, 1, true) {}

  void emitPrologue(MachineFunction &MF) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  bool targetHandlesStackFrameRounding() const override {
    return true;
  }
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
  void processFunctionBeforeFrameFinalized(MachineFunction &MF,
        RegScavenger *RS = NULL) const override;
  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                            RegScavenger *RS) const override;
  int getFrameIndexOffset(const MachineFunction &MF, int FI) const override;
  bool hasFP(const MachineFunction &MF) const override;
  bool hasTailCall(MachineBasicBlock &MBB) const;
  void adjustForCalleeSavedRegsSpillCall(MachineFunction &MF) const;
  bool replacePredRegPseudoSpillCode(MachineFunction &MF) const;

  const SpillSlot *getCalleeSavedSpillSlots(unsigned &NumEntries)
        const override {
    static const SpillSlot Offsets[] = {
      { Hexagon::R17, -4 }, { Hexagon::R16, -8 }, { Hexagon::D8, -8 },
      { Hexagon::R19, -12 }, { Hexagon::R18, -16 }, { Hexagon::D9, -16 },
      { Hexagon::R21, -20 }, { Hexagon::R20, -24 }, { Hexagon::D10, -24 },
      { Hexagon::R23, -28 }, { Hexagon::R22, -32 }, { Hexagon::D11, -32 },
      { Hexagon::R25, -36 }, { Hexagon::R24, -40 }, { Hexagon::D12, -40 },
      { Hexagon::R27, -44 }, { Hexagon::R26, -48 }, { Hexagon::D13, -48 }
    };

    NumEntries = array_lengthof(Offsets);
    return Offsets;
  }

  bool assignCalleeSavedSpillSlots(MachineFunction &MF,
        const TargetRegisterInfo *TRI,
        std::vector<CalleeSavedInfo> &CSI) const override;

  bool needsAligna(const MachineFunction &MF) const;
  MachineInstr *getAlignaInstr(MachineFunction &MF) const;
};

} // End llvm namespace

#endif
