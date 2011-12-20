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
#include "SparcTargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeSparcTarget() {
  // Register the target.
  RegisterTargetMachine<SparcV8TargetMachine> X(TheSparcTarget);
  RegisterTargetMachine<SparcV9TargetMachine> Y(TheSparcV9Target);
}

/// SparcTargetMachine ctor - Create an ILP32 architecture model
///
SparcTargetMachine::SparcTargetMachine(const Target &T, StringRef TT, 
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL,
                                       bool is64bit)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    Subtarget(TT, CPU, FS, is64bit),
    DataLayout(Subtarget.getDataLayout()),
    TLInfo(*this), TSInfo(*this), InstrInfo(Subtarget),
    FrameLowering(Subtarget) {
}

bool SparcTargetMachine::addInstSelector(PassManagerBase &PM) {
  PM.add(createSparcISelDag(*this));
  return false;
}

/// addPreEmitPass - This pass may be implemented by targets that want to run
/// passes immediately before machine code is emitted.  This should return
/// true if -print-machineinstrs should print out the code after the passes.
bool SparcTargetMachine::addPreEmitPass(PassManagerBase &PM){
  PM.add(createSparcFPMoverPass(*this));
  PM.add(createSparcDelaySlotFillerPass(*this));
  return true;
}

void SparcV8TargetMachine::anchor() { }

SparcV8TargetMachine::SparcV8TargetMachine(const Target &T,
                                           StringRef TT, StringRef CPU,
                                           StringRef FS,
                                           const TargetOptions &Options,
                                           Reloc::Model RM,
                                           CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
  : SparcTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {
}

void SparcV9TargetMachine::anchor() { }

SparcV9TargetMachine::SparcV9TargetMachine(const Target &T, 
                                           StringRef TT,  StringRef CPU,
                                           StringRef FS,
                                           const TargetOptions &Options,
                                           Reloc::Model RM,
                                           CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
  : SparcTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {
}
