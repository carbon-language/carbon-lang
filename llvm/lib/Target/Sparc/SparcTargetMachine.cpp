//===-- SparcTargetMachine.cpp - Define TargetMachine for Sparc -----------===//
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

#include "Sparc.h"
#include "SparcMCAsmInfo.h"
#include "SparcTargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeSparcTarget() {
  // Register the target.
  RegisterTargetMachine<SparcV8TargetMachine> X(TheSparcTarget);
  RegisterTargetMachine<SparcV9TargetMachine> Y(TheSparcV9Target);

  RegisterAsmInfo<SparcELFMCAsmInfo> A(TheSparcTarget);
  RegisterAsmInfo<SparcELFMCAsmInfo> B(TheSparcV9Target);

}

/// SparcTargetMachine ctor - Create an ILP32 architecture model
///
SparcTargetMachine::SparcTargetMachine(const Target &T, const std::string &TT, 
                                       const std::string &FS, bool is64bit)
  : LLVMTargetMachine(T, TT),
    Subtarget(TT, FS, is64bit),
    DataLayout(Subtarget.getDataLayout()),
    TLInfo(*this), TSInfo(*this), InstrInfo(Subtarget),
    FrameLowering(Subtarget) {
}

bool SparcTargetMachine::addInstSelector(PassManagerBase &PM,
                                         CodeGenOpt::Level OptLevel) {
  PM.add(createSparcISelDag(*this));
  return false;
}

/// addPreEmitPass - This pass may be implemented by targets that want to run
/// passes immediately before machine code is emitted.  This should return
/// true if -print-machineinstrs should print out the code after the passes.
bool SparcTargetMachine::addPreEmitPass(PassManagerBase &PM,
                                        CodeGenOpt::Level OptLevel){
  PM.add(createSparcFPMoverPass(*this));
  PM.add(createSparcDelaySlotFillerPass(*this));
  return true;
}

SparcV8TargetMachine::SparcV8TargetMachine(const Target &T,
                                           const std::string &TT, 
                                           const std::string &FS)
  : SparcTargetMachine(T, TT, FS, false) {
}

SparcV9TargetMachine::SparcV9TargetMachine(const Target &T, 
                                           const std::string &TT, 
                                           const std::string &FS)
  : SparcTargetMachine(T, TT, FS, true) {
}
