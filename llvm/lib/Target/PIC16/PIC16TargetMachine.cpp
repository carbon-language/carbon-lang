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
#include "PIC16MCAsmInfo.h"
#include "PIC16TargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetRegistry.h"

using namespace llvm;

extern "C" void LLVMInitializePIC16Target() {
  // Register the target. Curretnly the codegen works for
  // enhanced pic16 mid-range.
  RegisterTargetMachine<PIC16TargetMachine> X(ThePIC16Target);
  RegisterAsmInfo<PIC16MCAsmInfo> A(ThePIC16Target);
}


// PIC16TargetMachine - Enhanced PIC16 mid-range Machine. May also represent
// a Traditional Machine if 'Trad' is true.
PIC16TargetMachine::PIC16TargetMachine(const Target &T, const std::string &TT,
                                       const std::string &FS, bool Trad)
: LLVMTargetMachine(T, TT),
  Subtarget(TT, FS, Trad),
  DataLayout("e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"), 
  InstrInfo(*this), TLInfo(*this),
  FrameInfo(TargetFrameInfo::StackGrowsUp, 8, 0) { }


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


