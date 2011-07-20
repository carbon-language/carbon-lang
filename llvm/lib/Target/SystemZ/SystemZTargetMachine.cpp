//===-- SystemZTargetMachine.cpp - Define TargetMachine for SystemZ -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SystemZTargetMachine.h"
#include "SystemZ.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeSystemZTarget() {
  // Register the target.
  RegisterTargetMachine<SystemZTargetMachine> X(TheSystemZTarget);
}

/// SystemZTargetMachine ctor - Create an ILP64 architecture model
///
SystemZTargetMachine::SystemZTargetMachine(const Target &T,
                                           StringRef TT, StringRef CPU,
                                           StringRef FS, Reloc::Model RM,
                                           CodeModel::Model CM)
  : LLVMTargetMachine(T, TT, CPU, FS, RM, CM),
    Subtarget(TT, CPU, FS),
    DataLayout("E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32"
               "-f64:64:64-f128:128:128-a0:16:16-n32:64"),
    InstrInfo(*this), TLInfo(*this), TSInfo(*this),
    FrameLowering(Subtarget) {
}

bool SystemZTargetMachine::addInstSelector(PassManagerBase &PM,
                                          CodeGenOpt::Level OptLevel) {
  // Install an instruction selector.
  PM.add(createSystemZISelDag(*this, OptLevel));
  return false;
}
