//===- SkeletonInstrInfo.h - Instruction Information ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is where the target-specific implementation of the TargetInstrInfo
// class goes.
//
//===----------------------------------------------------------------------===//

#ifndef SKELETON_INSTRUCTIONINFO_H
#define SKELETON_INSTRUCTIONINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "SkeletonRegisterInfo.h"

namespace llvm {

  class SkeletonInstrInfo : public TargetInstrInfo {
    const SkeletonRegisterInfo RI;
  public:
    SkeletonInstrInfo();

    /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
    /// such, whenever a client has an instance of instruction info, it should
    /// always be able to get register info as well (through this method).
    ///
    virtual const MRegisterInfo &getRegisterInfo() const { return RI; }
  };
}

#endif
