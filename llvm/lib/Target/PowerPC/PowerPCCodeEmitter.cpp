//===-- PowerPCCodeEmitter.cpp - JIT Code Emitter for PowerPC -----*- C++ -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
//
//===----------------------------------------------------------------------===//

#include "PowerPCTargetMachine.h"

namespace llvm {

/// addPassesToEmitMachineCode - Add passes to the specified pass manager to get
/// machine code emitted.  This uses a MachineCodeEmitter object to handle
/// actually outputting the machine code and resolving things like the address
/// of functions.  This method should returns true if machine code emission is
/// not supported.
///
bool PowerPCTargetMachine::addPassesToEmitMachineCode(FunctionPassManager &PM,
                                                      MachineCodeEmitter &MCE) {
  return true;
  // It should go something like this:
  // PM.add(new Emitter(MCE));  // Machine code emitter pass for PowerPC
  // Delete machine code for this function after emitting it:
  // PM.add(createMachineCodeDeleter());
}

void *PowerPCJITInfo::getJITStubForFunction(Function *F,
                                            MachineCodeEmitter &MCE) {
  assert (0 && "PowerPCJITInfo::getJITStubForFunction not implemented");
  return 0;
}

void PowerPCJITInfo::replaceMachineCodeForFunction (void *Old, void *New) {
  assert (0 && "PowerPCJITInfo::replaceMachineCodeForFunction not implemented");
}

} // end llvm namespace

