//===-- X86TargetFrameLowering.h - Define frame lowering for X86 -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements X86-specific bits of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#ifndef X86_FRAMELOWERING_H
#define X86_FRAMELOWERING_H

#include "X86Subtarget.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {

namespace CU {

  /// Compact unwind encoding values.
  enum CompactUnwindEncodings {
    /// [RE]BP based frame where [RE]BP is pused on the stack immediately after
    /// the return address, then [RE]SP is moved to [RE]BP.
    UNWIND_MODE_BP_FRAME                   = 0x01000000,

    /// A frameless function with a small constant stack size.
    UNWIND_MODE_STACK_IMMD                 = 0x02000000,

    /// A frameless function with a large constant stack size.
    UNWIND_MODE_STACK_IND                  = 0x03000000,

    /// No compact unwind encoding is available.
    UNWIND_MODE_DWARF                      = 0x04000000,

    /// Mask for encoding the frame registers.
    UNWIND_BP_FRAME_REGISTERS              = 0x00007FFF,

    /// Mask for encoding the frameless registers.
    UNWIND_FRAMELESS_STACK_REG_PERMUTATION = 0x000003FF
  };

} // end CU namespace

class MCSymbol;
class X86TargetMachine;

class X86FrameLowering : public TargetFrameLowering {
  const X86TargetMachine &TM;
  const X86Subtarget &STI;
public:
  explicit X86FrameLowering(const X86TargetMachine &tm, const X86Subtarget &sti)
    : TargetFrameLowering(StackGrowsDown,
                          sti.getStackAlignment(),
                          (sti.is64Bit() ? -8 : -4)),
      TM(tm), STI(sti) {
  }

  void emitCalleeSavedFrameMoves(MachineFunction &MF, MCSymbol *Label,
                                 unsigned FramePtr) const;

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  void adjustForSegmentedStacks(MachineFunction &MF) const;

  void adjustForHiPEPrologue(MachineFunction &MF) const;

  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                            RegScavenger *RS = NULL) const;

  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI,
                                 const TargetRegisterInfo *TRI) const;

  bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   const std::vector<CalleeSavedInfo> &CSI,
                                   const TargetRegisterInfo *TRI) const;

  bool hasFP(const MachineFunction &MF) const;
  bool hasReservedCallFrame(const MachineFunction &MF) const;

  int getFrameIndexOffset(const MachineFunction &MF, int FI) const;
  int getFrameIndexReference(const MachineFunction &MF, int FI,
                             unsigned &FrameReg) const;
  uint32_t getCompactUnwindEncoding(MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MI) const;
};

} // End llvm namespace

#endif
