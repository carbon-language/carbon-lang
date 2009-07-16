//===-- SystemZTargetMachine.cpp - Define TargetMachine for SystemZ -----------===//
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
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

extern Target TheSystemZTarget;
namespace {
  // Register the target.
  RegisterTarget<SystemZTargetMachine> X(TheSystemZTarget,
                                         "systemz",
                                         "SystemZ [experimental]");
}

// Force static initialization.
extern "C" void LLVMInitializeSystemZTarget() {

}

const TargetAsmInfo *SystemZTargetMachine::createTargetAsmInfo() const {
  // FIXME: Handle Solaris subtarget someday :)
  return new SystemZTargetAsmInfo(*this);
}

/// SystemZTargetMachine ctor - Create an ILP64 architecture model
///
SystemZTargetMachine::SystemZTargetMachine(const Target &T,
                                           const Module &M,
                                           const std::string &FS)
  : LLVMTargetMachine(T),
    Subtarget(*this, M, FS),
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

unsigned SystemZTargetMachine::getModuleMatchQuality(const Module &M) {
  std::string TT = M.getTargetTriple();

  // We strongly match s390x
  if (TT.size() >= 5 && TT[0] == 's' && TT[1] == '3' && TT[2] == '9' &&
      TT[3] == '0' &&  TT[4] == 'x')
    return 20;

  return 0;
}
