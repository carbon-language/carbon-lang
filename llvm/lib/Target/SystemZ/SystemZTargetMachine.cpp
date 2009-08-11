//===-- SystemZTargetMachine.cpp - Define TargetMachine for SystemZ -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "SystemZTargetAsmInfo.h"
#include "SystemZTargetMachine.h"
#include "SystemZ.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeSystemZTarget() {
  // Register the target.
  RegisterTargetMachine<SystemZTargetMachine> X(TheSystemZTarget);
}

const TargetAsmInfo *SystemZTargetMachine::createTargetAsmInfo() const {
  return new SystemZTargetAsmInfo();
}

/// SystemZTargetMachine ctor - Create an ILP64 architecture model
///
SystemZTargetMachine::SystemZTargetMachine(const Target &T,
                                           const std::string &TT,
                                           const std::string &FS)
  : LLVMTargetMachine(T, TT),
    Subtarget(TT, FS),
    DataLayout("E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32"
               "-f64:64:64-f128:128:128-a0:16:16"),
    InstrInfo(*this), TLInfo(*this),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, -160) {

  if (getRelocationModel() == Reloc::Default)
    setRelocationModel(Reloc::Static);
}

bool SystemZTargetMachine::addInstSelector(PassManagerBase &PM,
                                          CodeGenOpt::Level OptLevel) {
  // Install an instruction selector.
  PM.add(createSystemZISelDag(*this, OptLevel));
  return false;
}
