//===-- X86TargetMachine.cpp - Define TargetMachine for the X86 -----------===//
// 
// This file defines the X86 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "X86TargetMachine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/PassManager.h"
#include "X86.h"
#include "Support/CommandLine.h"
#include "Support/Statistic.h"
#include <iostream>

namespace {
  cl::opt<bool> UseLocalRA("local-ra",
                           cl::desc("Use Local RegAlloc instead of Simple RA"));
}

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
  // For the moment we have decided that malloc and free will be
  // taken care of by converting them to calls, using the existing
  // LLVM scalar transforms pass to do this.
  PM.add(createLowerAllocationsPass());

  PM.add(createSimpleX86InstructionSelector(*this));

  // TODO: optional optimizations go here

  // Print the instruction selected machine code...
  DEBUG(PM.add(createMachineFunctionPrinterPass()));

  // Perform register allocation to convert to a concrete x86 representation
  if (UseLocalRA)
    PM.add(createLocalRegisterAllocator(*this));
  else
    PM.add(createSimpleRegisterAllocator(*this));

  // Print the instruction selected machine code...
  // PM.add(createMachineFunctionPrinterPass());

  // Print the register-allocated code
  DEBUG(PM.add(createX86CodePrinterPass(*this, std::cerr)));

  return false; // success!
}

