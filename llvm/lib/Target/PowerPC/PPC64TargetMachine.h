//===-- PPC64TargetMachine.h - Define AIX/PowerPC TargetMachine --*- C++ -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file declares the PowerPC/AIX specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC_AIX_TARGETMACHINE_H
#define POWERPC_AIX_TARGETMACHINE_H

#include "PowerPCTargetMachine.h"

namespace llvm {

class PPC64TargetMachine : public PowerPCTargetMachine {
public:
  PPC64TargetMachine(const Module &M, IntrinsicLowering *IL);

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
