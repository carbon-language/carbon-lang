//===-- PowerPCTargetMachine.h - Define TargetMachine for PowerPC -*- C++ -*-=//
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

#ifndef POWERPCTARGETMACHINE_H
#define POWERPCTARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/PassManager.h"
#include "PowerPCInstrInfo.h"
#include "PowerPCJITInfo.h"

namespace llvm {

class IntrinsicLowering;

class PowerPCTargetMachine : public TargetMachine {
  PowerPCInstrInfo InstrInfo;
  TargetFrameInfo FrameInfo;
  PowerPCJITInfo JITInfo;
public:
  PowerPCTargetMachine(const Module &M, IntrinsicLowering *IL);

  virtual const PowerPCInstrInfo *getInstrInfo() const { return &InstrInfo; }
  virtual const TargetFrameInfo  *getFrameInfo() const { return &FrameInfo; }
  virtual const MRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual TargetJITInfo *getJITInfo() {
    return &JITInfo;
  }

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
