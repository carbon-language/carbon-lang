//==- SystemZMachineFuctionInfo.h - SystemZ machine function info -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares SystemZ-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEMZMACHINEFUNCTIONINFO_H
#define SYSTEMZMACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

/// SystemZMachineFunctionInfo - This class is derived from MachineFunction and
/// contains private SystemZ target-specific information for each MachineFunction.
class SystemZMachineFunctionInfo : public MachineFunctionInfo {
  /// CalleeSavedFrameSize - Size of the callee-saved register portion of the
  /// stack frame in bytes.
  unsigned CalleeSavedFrameSize;

  /// LowReg - Low register of range of callee-saved registers to store.
  unsigned LowReg;

  /// HighReg - High register of range of callee-saved registers to store.
  unsigned HighReg;
public:
  SystemZMachineFunctionInfo() : CalleeSavedFrameSize(0) {}

  SystemZMachineFunctionInfo(MachineFunction &MF) : CalleeSavedFrameSize(0) {}

  unsigned getCalleeSavedFrameSize() const { return CalleeSavedFrameSize; }
  void setCalleeSavedFrameSize(unsigned bytes) { CalleeSavedFrameSize = bytes; }

  unsigned getLowReg() const { return LowReg; }
  void setLowReg(unsigned Reg) { LowReg = Reg; }

  unsigned getHighReg() const { return HighReg; }
  void setHighReg(unsigned Reg) { HighReg = Reg; }
};

} // End llvm namespace

#endif
