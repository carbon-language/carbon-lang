//===-- IA64TargetMachine.cpp - Define TargetMachine for IA64 -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IA64 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "IA64TargetAsmInfo.h"
#include "IA64TargetMachine.h"
#include "IA64.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

// Register the target
static RegisterTarget<IA64TargetMachine> X(TheIA64Target, "ia64",
                                           "IA-64 (Itanium) [experimental]");

// Force static initialization.
extern "C" void LLVMInitializeIA64Target() { }

const TargetAsmInfo *IA64TargetMachine::createTargetAsmInfo() const {
  return new IA64TargetAsmInfo(*this);
}

/// IA64TargetMachine ctor - Create an LP64 architecture model
///
IA64TargetMachine::IA64TargetMachine(const Target &T, const Module &M, 
                                     const std::string &FS)
  : LLVMTargetMachine(T),
    DataLayout("e-f80:128:128"),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 16, 0),
    TLInfo(*this) { // FIXME? check this stuff
}


//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool IA64TargetMachine::addInstSelector(PassManagerBase &PM,
                                        CodeGenOpt::Level OptLevel) {
  PM.add(createIA64DAGToDAGInstructionSelector(*this));
  return false;
}

bool IA64TargetMachine::addPreEmitPass(PassManagerBase &PM,
                                       CodeGenOpt::Level OptLevel) {
  // Make sure everything is bundled happily
  PM.add(createIA64BundlingPass(*this));
  return true;
}
