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

class SparcV8InstrInfo : public TargetInstrInfo {
  const SparcV8RegisterInfo RI;
public:
  SparcV8InstrInfo();

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const { return RI; }
};

}

#endif
