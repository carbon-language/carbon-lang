//===-- SparcV8TargetMachine.h - Define TargetMachine for SparcV8 -*- C++ -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file declares the SparcV8 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCV8TARGETMACHINE_H
#define SPARCV8TARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/PassManager.h"
#include "SparcV8InstrInfo.h"
#include "SparcV8JITInfo.h"

namespace llvm {

class IntrinsicLowering;

class SparcV8TargetMachine : public TargetMachine {
  SparcV8InstrInfo InstrInfo;
  TargetFrameInfo FrameInfo;
  SparcV8JITInfo JITInfo;
public:
  SparcV8TargetMachine(const Module &M, IntrinsicLowering *IL);

  virtual const SparcV8InstrInfo     &getInstrInfo() const { return InstrInfo; }
  virtual const TargetFrameInfo  &getFrameInfo() const { return FrameInfo; }
  virtual const MRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual TargetJITInfo *getJITInfo() {
    return &JITInfo;
  }

  virtual const TargetSchedInfo &getSchedInfo()  const { abort(); }
  virtual const TargetRegInfo   &getRegInfo()    const { abort(); }
  virtual const TargetCacheInfo  &getCacheInfo() const { abort(); }

  /// addPassesToEmitMachineCode - Add passes to the specified pass manager to
  /// get machine code emitted.  This uses a MachineCodeEmitter object to handle
  /// actually outputting the machine code and resolving things like the address
  /// of functions.  This method should returns true if machine code emission is
  /// not supported.
  ///
  virtual bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                          MachineCodeEmitter &MCE);
  
  virtual bool addPassesToEmitAssembly(PassManager &PM, std::ostream &Out);
};

} // end namespace llvm

#endif
