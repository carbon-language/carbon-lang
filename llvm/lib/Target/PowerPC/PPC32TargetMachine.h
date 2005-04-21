//===-- PPC32TargetMachine.h - Define TargetMachine for PowerPC -*- C++ -*-=//
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

#ifndef POWERPC32_TARGETMACHINE_H
#define POWERPC32_TARGETMACHINE_H

#include "PowerPCTargetMachine.h"
#include "PPC32JITInfo.h"
#include "PPC32InstrInfo.h"
#include "llvm/PassManager.h"

namespace llvm {

class IntrinsicLowering;

class PPC32TargetMachine : public PowerPCTargetMachine {
  PPC32InstrInfo InstrInfo;
  PPC32JITInfo JITInfo;

public:
  PPC32TargetMachine(const Module &M, IntrinsicLowering *IL);
  virtual const PPC32InstrInfo   *getInstrInfo() const { return &InstrInfo; }
  virtual const MRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual TargetJITInfo *getJITInfo() {
    return &JITInfo;
  }

  static unsigned getJITMatchQuality();

  static unsigned getModuleMatchQuality(const Module &M);

  bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                  MachineCodeEmitter &MCE);
};

} // end namespace llvm

#endif
