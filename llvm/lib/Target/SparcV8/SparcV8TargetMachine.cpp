//===-- SparcV8TargetMachine.cpp - Define TargetMachine for SparcV8 -------===//
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
#include "SparcV8.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"

namespace llvm {

// allocateSparcV8TargetMachine - Allocate and return a subclass of 
// TargetMachine that implements the SparcV8 backend.
//
TargetMachine *allocateSparcV8TargetMachine(const Module &M,
                                            IntrinsicLowering *IL) {
  return new SparcV8TargetMachine(M, IL);
}

/// SparcV8TargetMachine ctor - Create an ILP32 architecture model
///
SparcV8TargetMachine::SparcV8TargetMachine(const Module &M,
                                           IntrinsicLowering *IL)
  : TargetMachine("SparcV8", IL, true, 4, 4, 4, 4, 4),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, 4), JITInfo(*this) {
}

/// addPassesToEmitAssembly - Add passes to the specified pass manager
/// to implement a static compiler for this target.
///
bool SparcV8TargetMachine::addPassesToEmitAssembly(PassManager &PM,
					       std::ostream &Out) {
  // <insert instruction selector passes here>
  PM.add(createRegisterAllocator());
  PM.add(createPrologEpilogCodeInserter());
  // <insert assembly code output passes here>
  PM.add(createMachineCodeDeleter());
  return true; // change to `return false' when this actually works.
}

/// addPassesToJITCompile - Add passes to the specified pass manager to
/// implement a fast dynamic compiler for this target.
///
void SparcV8JITInfo::addPassesToJITCompile(FunctionPassManager &PM) {
  // <insert instruction selector passes here>
  PM.add(createRegisterAllocator());
  PM.add(createPrologEpilogCodeInserter());
}

} // end namespace llvm
