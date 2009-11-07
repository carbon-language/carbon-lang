//===-- MSP430TargetMachine.cpp - Define TargetMachine for MSP430 ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the MSP430 target.
//
//===----------------------------------------------------------------------===//

#include "MSP430.h"
#include "MSP430MCAsmInfo.h"
#include "MSP430TargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeMSP430Target() {
  // Register the target.
  RegisterTargetMachine<MSP430TargetMachine> X(TheMSP430Target);
  RegisterAsmInfo<MSP430MCAsmInfo> Z(TheMSP430Target);
}

MSP430TargetMachine::MSP430TargetMachine(const Target &T,
                                         const std::string &TT,
                                         const std::string &FS) :
  LLVMTargetMachine(T, TT),
  Subtarget(TT, FS),
  // FIXME: Check TargetData string.
  DataLayout("e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"),
  InstrInfo(*this), TLInfo(*this),
  FrameInfo(TargetFrameInfo::StackGrowsDown, 2, -2) { }


bool MSP430TargetMachine::addInstSelector(PassManagerBase &PM,
                                          CodeGenOpt::Level OptLevel) {
  // Install an instruction selector.
  PM.add(createMSP430ISelDag(*this, OptLevel));
  return false;
}

