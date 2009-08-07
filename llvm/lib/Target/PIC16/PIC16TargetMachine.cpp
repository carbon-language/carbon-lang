//===-- PIC16TargetMachine.cpp - Define TargetMachine for PIC16 -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PIC16 target.
//
//===----------------------------------------------------------------------===//

#include "PIC16.h"
#include "PIC16TargetAsmInfo.h"
#include "PIC16TargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetAsmInfo.h"

using namespace llvm;

// PIC16TargetMachine - Traditional PIC16 Machine.
PIC16TargetMachine::PIC16TargetMachine(const Target &T, const std::string &TT,
                                       const std::string &FS, bool Cooper)
: LLVMTargetMachine(T),
  Subtarget(TT, FS, Cooper),
  DataLayout("e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"), 
  InstrInfo(*this), TLInfo(*this),
  FrameInfo(TargetFrameInfo::StackGrowsUp, 8, 0) { }

// CooperTargetMachine - Uses the same PIC16TargetMachine, but makes IsCooper
// as true.
CooperTargetMachine::CooperTargetMachine(const Target &T, const std::string &TT, 
                                         const std::string &FS)
  : PIC16TargetMachine(T, TT, FS, true) {}


const TargetAsmInfo *PIC16TargetMachine::createTargetAsmInfo() const {
  return new PIC16TargetAsmInfo();
}

bool PIC16TargetMachine::addInstSelector(PassManagerBase &PM,
                                         CodeGenOpt::Level OptLevel) {
  // Install an instruction selector.
  PM.add(createPIC16ISelDag(*this));
  return false;
}

bool PIC16TargetMachine::addPreEmitPass(PassManagerBase &PM, 
                                         CodeGenOpt::Level OptLevel) {
  PM.add(createPIC16MemSelOptimizerPass());
  return true;  // -print-machineinstr should print after this.
}


