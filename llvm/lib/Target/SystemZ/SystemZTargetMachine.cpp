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

/// SystemZTargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int SystemZTargetMachineModule;
int SystemZTargetMachineModule = 0;

// Register the target.
static RegisterTarget<SystemZTargetMachine>
X("systemz", "SystemZ [experimental]");

const TargetAsmInfo *SystemZTargetMachine::createTargetAsmInfo() const {
  // FIXME: Handle Solaris subtarget someday :)
  return new SystemZTargetAsmInfo(*this);
}

/// SystemZTargetMachine ctor - Create an ILP64 architecture model
///
SystemZTargetMachine::SystemZTargetMachine(const Module &M, const std::string &FS)
  : Subtarget(*this, M, FS),
    DataLayout("E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128"),
    InstrInfo(*this), TLInfo(*this),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, 0) {
}

bool SystemZTargetMachine::addInstSelector(PassManagerBase &PM,
                                          CodeGenOpt::Level OptLevel) {
  // Install an instruction selector.
  PM.add(createSystemZISelDag(*this, OptLevel));
  return false;
}

bool SystemZTargetMachine::addAssemblyEmitter(PassManagerBase &PM,
                                             CodeGenOpt::Level OptLevel,
                                             bool Verbose,
                                             raw_ostream &Out) {
  // Output assembly language.
  PM.add(createSystemZCodePrinterPass(Out, *this, OptLevel, Verbose));
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
