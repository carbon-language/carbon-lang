//===-- SkeletonCodeEmitter.cpp - JIT Code Emitter --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a stub for a JIT code generator, which is obviously not implemented.
//
//===----------------------------------------------------------------------===//

#include "SkeletonTargetMachine.h"
using namespace llvm;

/// addPassesToEmitMachineCode - Add passes to the specified pass manager to get
/// machine code emitted.  This uses a MachineCodeEmitter object to handle
/// actually outputting the machine code and resolving things like the address
/// of functions.  This method should returns true if machine code emission is
/// not supported.
///
bool SkeletonTargetMachine::addPassesToEmitMachineCode(FunctionPassManager &PM,
                                                       MachineCodeEmitter &MCE){
  return true;  // Not implemented yet!
}

void SkeletonJITInfo::replaceMachineCodeForFunction (void *Old, void *New) {
  assert (0 && "replaceMachineCodeForFunction not implemented");
}

