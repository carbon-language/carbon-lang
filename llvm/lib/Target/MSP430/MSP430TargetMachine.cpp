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
#include "MSP430TargetAsmInfo.h"
#include "MSP430TargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetAsmInfo.h"
using namespace llvm;

MSP430TargetMachine::MSP430TargetMachine(const Target &T,
                                         const std::string &TT,
                                         const std::string &FS) :
  LLVMTargetMachine(T, TT),
  Subtarget(TT, FS),
  // FIXME: Check TargetData string.
  DataLayout("e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"),
  InstrInfo(*this), TLInfo(*this),
  FrameInfo(TargetFrameInfo::StackGrowsDown, 2, -2) { }

const TargetAsmInfo *MSP430TargetMachine::createTargetAsmInfo() const {
  return new MSP430TargetAsmInfo();
}

bool MSP430TargetMachine::addInstSelector(PassManagerBase &PM,
                                          CodeGenOpt::Level OptLevel) {
  // Install an instruction selector.
  PM.add(createMSP430ISelDag(*this, OptLevel));
  return false;
}

