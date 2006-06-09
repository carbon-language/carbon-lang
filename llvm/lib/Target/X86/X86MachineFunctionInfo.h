//====- X86MachineFuctionInfo.h - X86 machine function info -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file declares X86-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef X86MACHINEFUNCTIONINFO_H
#define X86MACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

/// X86FunctionInfo - This class is derived from MachineFunction private
/// X86 target-specific information for each MachineFunction.
class X86FunctionInfo : public MachineFunctionInfo {
  // ForceFramePointer - True if the function is required to use of frame
  // pointer for reasons other than it containing dynamic allocation or 
  // that FP eliminatation is turned off. For example, Cygwin main function
  // contains stack pointer re-alignment code which requires FP.
  bool ForceFramePointer;
public:
  X86FunctionInfo(MachineFunction& MF) : ForceFramePointer(false) {}
  bool getForceFramePointer() const { return ForceFramePointer;} 
  void setForceFramePointer(bool forceFP) { ForceFramePointer = forceFP; }
};
} // End llvm namespace

#endif
