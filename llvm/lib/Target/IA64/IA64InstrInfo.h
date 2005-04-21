//===- IA64InstrInfo.h - IA64 Instruction Information ----------*- C++ -*- ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Duraid Madina and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the IA64 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef IA64INSTRUCTIONINFO_H
#define IA64INSTRUCTIONINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "IA64RegisterInfo.h"

namespace llvm {

/// IA64II - This namespace holds all of the target specific flags that
/// instruction info tracks.
/// FIXME: now gone!

  class IA64InstrInfo : public TargetInstrInfo {
  const IA64RegisterInfo RI;
public:
  IA64InstrInfo();

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const { return RI; }

  //
  // Return true if the instruction is a register to register move and
  // leave the source and dest operands in the passed parameters.
  //
  virtual bool isMoveInstr(const MachineInstr& MI,
                           unsigned& sourceReg,
                           unsigned& destReg) const;

};

} // End llvm namespace

#endif

