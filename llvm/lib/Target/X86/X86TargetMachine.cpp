//===-- X86TargetMachine.cpp - Define TargetMachine for the X86 -----------===//
// 
// This file defines the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "X86TargetMachine.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "llvm/PassManager.h"
#include "X86.h"
#include <iostream>

// allocateX86TargetMachine - Allocate and return a subclass of TargetMachine
// that implements the X86 backend.
//
TargetMachine *allocateX86TargetMachine() { return new X86TargetMachine(); }


/// X86TargetMachine ctor - Create an ILP32 architecture model
///
X86TargetMachine::X86TargetMachine() : TargetMachine("X86", 1, 4, 4, 4) {
}


/// addPassesToJITCompile - Add passes to the specified pass manager to
/// implement a fast dynamic compiler for this target.  Return true if this is
/// not supported for this target.
///
bool X86TargetMachine::addPassesToJITCompile(PassManager &PM) {
  PM.add(createSimpleX86InstructionSelector(*this));

  // TODO: optional optimizations go here

  // Perform register allocation to convert to a concrete x86 representation
  //PM.add(createSimpleX86RegisterAllocator(*this));

  PM.add(createX86CodePrinterPass(*this, std::cerr));

  //PM.add(createEmitX86CodeToMemory(*this));

  return false; // success!
}

