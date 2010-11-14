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
    DataLayout("e-f128:128:128-n64"),
    FrameInfo(TargetFrameInfo::StackGrowsDown, 16, 0),
    Subtarget(TT, FS),
    TLInfo(*this),
    TSInfo(*this) {
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
