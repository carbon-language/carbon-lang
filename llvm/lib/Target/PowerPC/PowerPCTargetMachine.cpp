//===-- PowerPCTargetMachine.cpp - Define TargetMachine for PowerPC -------===//
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
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"

namespace llvm {

// allocatePowerPCTargetMachine - Allocate and return a subclass of 
// TargetMachine that implements the PowerPC backend.
//
TargetMachine *allocatePowerPCTargetMachine(const Module &M,
                                            IntrinsicLowering *IL) {
  return new PowerPCTargetMachine(M, IL);
}

/// PowerPCTargetMachine ctor - Create an ILP32 architecture model
///
PowerPCTargetMachine::PowerPCTargetMachine(const Module &M,
                                           IntrinsicLowering *IL)
  : TargetMachine("PowerPC", IL, true, 4, 4, 4, 4, 4),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, 4) {
}

// addPassesToEmitAssembly - We currently use all of the same passes as the JIT
// does to emit statically compiled machine code.
bool PowerPCTargetMachine::addPassesToEmitAssembly(PassManager &PM,
					       std::ostream &Out) {
  return true;
}

/// addPassesToJITCompile - Add passes to the specified pass manager to
/// implement a fast dynamic compiler for this target.
///
void PowerPCJITInfo::addPassesToJITCompile(FunctionPassManager &PM) {
}

} // end namespace llvm
