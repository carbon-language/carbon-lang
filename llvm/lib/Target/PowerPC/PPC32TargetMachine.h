//===-- PPC32TargetMachine.h - PowerPC/Darwin TargetMachine ---*- C++ -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file declares the PowerPC/Darwin specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC_DARWIN_TARGETMACHINE_H
#define POWERPC_DARWIN_TARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/PassManager.h"
#include "PowerPCTargetMachine.h"

namespace llvm {

class IntrinsicLowering;

class PPC32TargetMachine : public PowerPCTargetMachine {
public:
  PPC32TargetMachine(const Module &M, IntrinsicLowering *IL);

  /// addPassesToEmitMachineCode - Add passes to the specified pass manager to
  /// get machine code emitted.  This uses a MachineCodeEmitter object to handle
  /// actually outputting the machine code and resolving things like the address
  /// of functions.  This method should returns true if machine code emission is
  /// not supported.
  ///
  virtual bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                          MachineCodeEmitter &MCE);
  
  static unsigned getModuleMatchQuality(const Module &M);
};

} // end namespace llvm

#endif
