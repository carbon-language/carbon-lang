//===- SparcV8InstrInfo.h - SparcV8 Instruction Information -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SparcV8 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCV8INSTRUCTIONINFO_H
#define SPARCV8INSTRUCTIONINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "SparcV8RegisterInfo.h"

namespace llvm {

/// V8II - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace V8II {
  enum {
    Pseudo = (1<<0),
    Load = (1<<1),
    Store = (1<<2),
    DelaySlot = (1<<3)
  };
};

class SparcV8InstrInfo : public TargetInstrInfo {
  const SparcV8RegisterInfo RI;
public:
  SparcV8InstrInfo();

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const { return RI; }

  /// Return true if the instruction is a register to register move and
  /// leave the source and dest operands in the passed parameters.
  ///
  virtual bool isMoveInstr(const MachineInstr &MI,
                           unsigned &SrcReg, unsigned &DstReg) const;
};

}

#endif
