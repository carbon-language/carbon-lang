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
#include "PowerPC.h"
#include "llvm/IntrinsicLowering.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineImpls.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

// allocatePowerPCTargetMachine - Allocate and return a subclass of 
// TargetMachine that implements the PowerPC backend.
//
TargetMachine *llvm::allocatePowerPCTargetMachine(const Module &M,
                                                  IntrinsicLowering *IL) {
  return new PowerPCTargetMachine(M, IL);
}

/// PowerPCTargetMachine ctor - Create an ILP32 architecture model
///
/// FIXME: Should double alignment be 8 bytes?  Then we get a PtrAl != DoubleAl abort
PowerPCTargetMachine::PowerPCTargetMachine(const Module &M,
                                           IntrinsicLowering *IL)
  : TargetMachine("PowerPC", IL, false, 4, 4, 4, 4, 4, 4, 4, 4),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 16, -4), JITInfo(*this) {
}

/// addPassesToEmitAssembly - Add passes to the specified pass manager
/// to implement a static compiler for this target.
///
bool PowerPCTargetMachine::addPassesToEmitAssembly(PassManager &PM,
					       std::ostream &Out) {
  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: The code generator does not properly handle functions with
  // unreachable basic blocks.
  PM.add(createCFGSimplificationPass());

  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  PM.add(createPPCSimpleInstructionSelector(*this));
  PM.add(createRegisterAllocator());
  PM.add(createPrologEpilogCodeInserter());
  PM.add(createPPCCodePrinterPass(Out, *this));
  PM.add(createMachineCodeDeleter());
  return false;
}

/// addPassesToJITCompile - Add passes to the specified pass manager to
/// implement a fast dynamic compiler for this target.
///
void PowerPCJITInfo::addPassesToJITCompile(FunctionPassManager &PM) {
  // FIXME: Implement efficient support for garbage collection intrinsics.
  PM.add(createLowerGCPass());

  // FIXME: Implement the invoke/unwind instructions!
  PM.add(createLowerInvokePass());

  // FIXME: The code generator does not properly handle functions with
  // unreachable basic blocks.
  PM.add(createCFGSimplificationPass());

  // FIXME: Implement the switch instruction in the instruction selector!
  PM.add(createLowerSwitchPass());

  PM.add(createPPCSimpleInstructionSelector(TM));
  PM.add(createRegisterAllocator());
  PM.add(createPrologEpilogCodeInserter());
}

