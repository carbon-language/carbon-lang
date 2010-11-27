//==- SystemZFrameInfo.h - Define TargetFrameInfo for z/System --*- C++ -*--==//
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

#ifndef SYSTEMZ_FRAMEINFO_H
#define SYSTEMZ_FRAMEINFO_H

#include "SystemZ.h"
#include "SystemZSubtarget.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/ADT/IndexedMap.h"

namespace llvm {
  class SystemZSubtarget;

class SystemZFrameInfo : public TargetFrameInfo {
  IndexedMap<unsigned> RegSpillOffsets;
protected:
  const SystemZSubtarget &STI;

public:
  explicit SystemZFrameInfo(const SystemZSubtarget &sti);

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI,
                                 const TargetRegisterInfo *TRI) const;
  bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   const std::vector<CalleeSavedInfo> &CSI,
                                   const TargetRegisterInfo *TRI) const;

  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                            RegScavenger *RS) const;

  bool hasReservedCallFrame(const MachineFunction &MF) const { return true; }
  bool hasFP(const MachineFunction &MF) const;
  int getFrameIndexOffset(const MachineFunction &MF, int FI) const;
};

} // End llvm namespace

#endif
