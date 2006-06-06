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

class X86FunctionInfo : public MachineFunctionInfo {
  bool ForceFramePointer;  // Function requires use of frame pointer.
public:
  X86FunctionInfo(MachineFunction& MF) : ForceFramePointer(false) {}
  bool getForceFramePointer() const { return ForceFramePointer;} 
  void setForceFramePointer(bool forceFP) { ForceFramePointer = forceFP; }
};
} // End llvm namespace

#endif
