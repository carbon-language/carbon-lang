//===-- PPCFrameLowering.h - Define frame lowering for PowerPC --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_PPCFRAMELOWERING_H
#define LLVM_LIB_TARGET_POWERPC_PPCFRAMELOWERING_H

#include "PPC.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class PPCSubtarget;

class PPCFrameLowering: public TargetFrameLowering {
  const PPCSubtarget &Subtarget;

public:
  PPCFrameLowering(const PPCSubtarget &STI);

  unsigned determineFrameLayout(MachineFunction &MF,
                                bool UpdateMF = true,
                                bool UseEstimate = false) const;

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  bool hasFP(const MachineFunction &MF) const override;
  bool needsFP(const MachineFunction &MF) const;
  void replaceFPWithRealFP(MachineFunction &MF) const;

  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                     RegScavenger *RS = nullptr) const override;
  void processFunctionBeforeFrameFinalized(MachineFunction &MF,
                                     RegScavenger *RS = nullptr) const override;
  void addScavengingSpillSlot(MachineFunction &MF, RegScavenger *RS) const;

  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI,
                                 const TargetRegisterInfo *TRI) const override;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                  MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I) const override;

  bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MI,
                                  const std::vector<CalleeSavedInfo> &CSI,
                                  const TargetRegisterInfo *TRI) const override;

  /// targetHandlesStackFrameRounding - Returns true if the target is
  /// responsible for rounding up the stack frame (probably at emitPrologue
  /// time).
  bool targetHandlesStackFrameRounding() const override { return true; }

  /// getReturnSaveOffset - Return the previous frame offset to save the
  /// return address.
  static unsigned getReturnSaveOffset(bool isPPC64, bool isDarwinABI) {
    if (isDarwinABI)
      return isPPC64 ? 16 : 8;
    // SVR4 ABI:
    return isPPC64 ? 16 : 4;
  }

  /// getTOCSaveOffset - Return the previous frame offset to save the
  /// TOC register -- 64-bit SVR4 ABI only.
  static unsigned getTOCSaveOffset(bool isELFv2ABI) {
    return isELFv2ABI ? 24 : 40;
  }

  /// getFramePointerSaveOffset - Return the previous frame offset to save the
  /// frame pointer.
  static unsigned getFramePointerSaveOffset(bool isPPC64, bool isDarwinABI) {
    // For the Darwin ABI:
    // We cannot use the TOC save slot (offset +20) in the PowerPC linkage area
    // for saving the frame pointer (if needed.)  While the published ABI has
    // not used this slot since at least MacOSX 10.2, there is older code
    // around that does use it, and that needs to continue to work.
    if (isDarwinABI)
      return isPPC64 ? -8U : -4U;

    // SVR4 ABI: First slot in the general register save area.
    return isPPC64 ? -8U : -4U;
  }

  /// getBasePointerSaveOffset - Return the previous frame offset to save the
  /// base pointer.
  static unsigned getBasePointerSaveOffset(bool isPPC64,
                                           bool isDarwinABI,
                                           bool isPIC) {
    if (isDarwinABI)
      return isPPC64 ? -16U : -8U;

    // SVR4 ABI: First slot in the general register save area.
    return isPPC64 ? -16U : isPIC ? -12U : -8U;
  }

  /// getLinkageSize - Return the size of the PowerPC ABI linkage area.
  ///
  static unsigned getLinkageSize(bool isPPC64, bool isDarwinABI,
                                 bool isELFv2ABI) {
    if (isDarwinABI || isPPC64)
      return (isELFv2ABI ? 4 : 6) * (isPPC64 ? 8 : 4);

    // SVR4 ABI:
    return 8;
  }

  const SpillSlot *
  getCalleeSavedSpillSlots(unsigned &NumEntries) const override;
};
} // End llvm namespace

#endif
