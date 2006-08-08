//===- ARMInstrInfo.h - ARM Instruction Information --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMINSTRUCTIONINFO_H
#define ARMINSTRUCTIONINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "ARMRegisterInfo.h"

namespace llvm {

class ARMInstrInfo : public TargetInstrInfo {
  const ARMRegisterInfo RI;
public:
  ARMInstrInfo();

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const { return RI; }

  /// getPointerRegClass - Return the register class to use to hold pointers.
  /// This is used for addressing modes.
  virtual const TargetRegisterClass *getPointerRegClass() const;

  /// Return true if the instruction is a register to register move and
  /// leave the source and dest operands in the passed parameters.
  ///
  virtual bool isMoveInstr(const MachineInstr &MI,
                           unsigned &SrcReg, unsigned &DstReg) const;
};

}

#endif
