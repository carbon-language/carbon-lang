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

#include "SparcTargetAsmInfo.h"
#include "SparcTargetMachine.h"
#include "Sparc.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeSparcTarget() {
  // Register the target.
  RegisterTargetMachine<SparcTargetMachine> X(TheSparcTarget);
  RegisterAsmInfo<SparcELFTargetAsmInfo> Y(TheSparcTarget);

}

/// SparcTargetMachine ctor - Create an ILP32 architecture model
///
SparcTargetMachine::SparcTargetMachine(const Target &T, const std::string &TT, 
                                       const std::string &FS)
  : LLVMTargetMachine(T, TT),
    DataLayout("E-p:32:32-f128:128:128"),
    Subtarget(TT, FS), TLInfo(*this), InstrInfo(Subtarget),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 8, 0) {
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
