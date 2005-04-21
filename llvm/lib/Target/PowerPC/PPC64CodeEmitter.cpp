//===-- PPC64CodeEmitter.cpp - JIT Code Emitter for PPC64 -----*- C++ -*-=//
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

#include "PPC64JITInfo.h"
#include "PPC64TargetMachine.h"
using namespace llvm;

/// addPassesToEmitMachineCode - Add passes to the specified pass manager to get
/// machine code emitted.  This uses a MachineCodeEmitter object to handle
/// actually outputting the machine code and resolving things like the address
/// of functions.  This method should returns true if machine code emission is
/// not supported.
///
bool PPC64TargetMachine::addPassesToEmitMachineCode(FunctionPassManager &PM,
                                                     MachineCodeEmitter &MCE) {
  return true;
  // It should go something like this:
  // PM.add(new Emitter(MCE));  // Machine code emitter pass for PPC64
  // Delete machine code for this function after emitting it:
  // PM.add(createMachineCodeDeleter());
}

void PPC64JITInfo::replaceMachineCodeForFunction (void *Old, void *New) {
  assert (0 && "PPC64JITInfo::replaceMachineCodeForFunction not implemented");
}

