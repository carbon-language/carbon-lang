//===-- X86TargetFrameInfo.h - Define TargetFrameInfo for X86 ---*- C++ -*-===//
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

#ifndef X86_FRAMEINFO_H
#define X86_FRAMEINFO_H

#include "X86Subtarget.h"
#include "llvm/Target/TargetFrameInfo.h"

namespace llvm {
  class MCSymbol;
  class X86TargetMachine;

class X86FrameInfo : public TargetFrameInfo {
  const X86TargetMachine &TM;
  const X86Subtarget &STI;
public:
  explicit X86FrameInfo(const X86TargetMachine &tm, const X86Subtarget &sti)
    : TargetFrameInfo(StackGrowsDown,
                      sti.getStackAlignment(),
                      (sti.isTargetWin64() ? -40 : (sti.is64Bit() ? -8 : -4))),
      TM(tm), STI(sti) {
  }

  void emitCalleeSavedFrameMoves(MachineFunction &MF, MCSymbol *Label,
                                 unsigned FramePtr) const;

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

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

  void getInitialFrameState(std::vector<MachineMove> &Moves) const;
  int getFrameIndexOffset(const MachineFunction &MF, int FI) const;
};

} // End llvm namespace

#endif
