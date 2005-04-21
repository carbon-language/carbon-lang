//===-- PPC64TargetMachine.h - Define TargetMachine for PowerPC64 -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PowerPC specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC64_TARGETMACHINE_H
#define POWERPC64_TARGETMACHINE_H

#include "PowerPCTargetMachine.h"
#include "PPC64InstrInfo.h"
#include "llvm/PassManager.h"

namespace llvm {

class IntrinsicLowering;

class PPC64TargetMachine : public PowerPCTargetMachine {
  PPC64InstrInfo InstrInfo;

public:
  PPC64TargetMachine(const Module &M, IntrinsicLowering *IL);
  virtual const PPC64InstrInfo   *getInstrInfo() const { return &InstrInfo; }
  virtual const MRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  static unsigned getModuleMatchQuality(const Module &M);

  bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                  MachineCodeEmitter &MCE);
};

} // end namespace llvm

#endif
