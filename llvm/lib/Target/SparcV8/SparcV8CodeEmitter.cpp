//===-- SparcV8CodeEmitter.cpp - JIT Code Emitter for SparcV8 -----*- C++ -*-=//
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

#include "SparcV8TargetMachine.h"

namespace llvm {

/// addPassesToEmitMachineCode - Add passes to the specified pass manager to get
/// machine code emitted.  This uses a MachineCodeEmitter object to handle
/// actually outputting the machine code and resolving things like the address
/// of functions.  This method should returns true if machine code emission is
/// not supported.
///
bool SparcV8TargetMachine::addPassesToEmitMachineCode(FunctionPassManager &PM,
                                                      MachineCodeEmitter &MCE) {
  return true;
  // It should go something like this:
  // PM.add(new Emitter(MCE));  // Machine code emitter pass for SparcV8
  // Delete machine code for this function after emitting it:
  // PM.add(createMachineCodeDeleter());
}

void *SparcV8JITInfo::getJITStubForFunction(Function *F,
                                            MachineCodeEmitter &MCE) {
  assert (0 && "SparcV8JITInfo::getJITStubForFunction not implemented");
  return 0;
}

void SparcV8JITInfo::replaceMachineCodeForFunction (void *Old, void *New) {
  assert (0 && "SparcV8JITInfo::replaceMachineCodeForFunction not implemented");
}

} // end llvm namespace

