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
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include <iostream>
using namespace llvm;

PowerPCTargetMachine::PowerPCTargetMachine(const std::string &name,
                                           IntrinsicLowering *IL,
                                           const TargetData &TD,
                                           const TargetFrameInfo &TFI,
                                           const PowerPCJITInfo &TJI) 
  : TargetMachine(name, IL, TD), FrameInfo(TFI), JITInfo(TJI) {}

unsigned PowerPCTargetMachine::getJITMatchQuality() {
#if defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)
  return 10;
#else
  return 0;
#endif
}

void PowerPCJITInfo::addPassesToJITCompile(FunctionPassManager &PM) {
  assert(0 && "Cannot execute PowerPCJITInfo::addPassesToJITCompile()");
}

void PowerPCJITInfo::replaceMachineCodeForFunction(void *Old, void *New) {
  assert(0 && "Cannot execute PowerPCJITInfo::replaceMachineCodeForFunction()");
}

void *PowerPCJITInfo::getJITStubForFunction(Function *F, 
                                            MachineCodeEmitter &MCE) {
  assert(0 && "Cannot execute PowerPCJITInfo::getJITStubForFunction()");
  return 0;
}
