//===-- AlphaTargetMachine.cpp - Define TargetMachine for Alpha -----------===//
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

#include "Alpha.h"
#include "AlphaJITInfo.h"
#include "AlphaMCAsmInfo.h"
#include "AlphaTargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeAlphaTarget() { 
  // Register the target.
  RegisterTargetMachine<AlphaTargetMachine> X(TheAlphaTarget);
  RegisterAsmInfo<AlphaMCAsmInfo> Y(TheAlphaTarget);
}

AlphaTargetMachine::AlphaTargetMachine(const Target &T, const std::string &TT,
                                       const std::string &FS)
  : LLVMTargetMachine(T, TT),
    DataLayout("e-f128:128:128"),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 16, 0),
    JITInfo(*this),
    Subtarget(TT, FS),
    TLInfo(*this) {
  setRelocationModel(Reloc::PIC_);
}


//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool AlphaTargetMachine::addInstSelector(PassManagerBase &PM,
                                         CodeGenOpt::Level OptLevel) {
  PM.add(createAlphaISelDag(*this));
  return false;
}
bool AlphaTargetMachine::addPreEmitPass(PassManagerBase &PM,
                                        CodeGenOpt::Level OptLevel) {
  // Must run branch selection immediately preceding the asm printer
  PM.add(createAlphaBranchSelectionPass());
  PM.add(createAlphaLLRPPass(*this));
  return false;
}
bool AlphaTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                        CodeGenOpt::Level OptLevel,
                                        MachineCodeEmitter &MCE) {
  PM.add(createAlphaCodeEmitterPass(*this, MCE));
  return false;
}
bool AlphaTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                        CodeGenOpt::Level OptLevel,
                                        JITCodeEmitter &JCE) {
  PM.add(createAlphaJITCodeEmitterPass(*this, JCE));
  return false;
}
bool AlphaTargetMachine::addCodeEmitter(PassManagerBase &PM,
                                        CodeGenOpt::Level OptLevel,
                                        ObjectCodeEmitter &OCE) {
  PM.add(createAlphaObjectCodeEmitterPass(*this, OCE));
  return false;
}
bool AlphaTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM,
                                              CodeGenOpt::Level OptLevel,
                                              MachineCodeEmitter &MCE) {
  return addCodeEmitter(PM, OptLevel, MCE);
}
bool AlphaTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM,
                                              CodeGenOpt::Level OptLevel,
                                              JITCodeEmitter &JCE) {
  return addCodeEmitter(PM, OptLevel, JCE);
}
bool AlphaTargetMachine::addSimpleCodeEmitter(PassManagerBase &PM,
                                              CodeGenOpt::Level OptLevel,
                                              ObjectCodeEmitter &OCE) {
  return addCodeEmitter(PM, OptLevel, OCE);
}

