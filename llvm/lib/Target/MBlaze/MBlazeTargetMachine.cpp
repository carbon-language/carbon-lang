//===-- MBlazeTargetMachine.cpp - Define TargetMachine for MBlaze ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about MBlaze target spec.
//
//===----------------------------------------------------------------------===//

#include "MBlaze.h"
#include "MBlazeMCAsmInfo.h"
#include "MBlazeTargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

extern "C" void LLVMInitializeMBlazeTarget() {
  // Register the target.
  RegisterTargetMachine<MBlazeTargetMachine> X(TheMBlazeTarget);
  RegisterAsmInfo<MBlazeMCAsmInfo> A(TheMBlazeTarget);
}

// DataLayout --> Big-endian, 32-bit pointer/ABI/alignment
// The stack is always 8 byte aligned
// On function prologue, the stack is created by decrementing
// its pointer. Once decremented, all references are done with positive
// offset from the stack/frame pointer, using StackGrowsUp enables
// an easier handling.
MBlazeTargetMachine::
MBlazeTargetMachine(const Target &T, const std::string &TT,
                    const std::string &FS):
  LLVMTargetMachine(T, TT),
  Subtarget(TT, FS),
  DataLayout("E-p:32:32-i8:8:8-i16:16:16-i64:32:32-"
             "f64:32:32-v64:32:32-v128:32:32-n32"),
  InstrInfo(*this),
  FrameInfo(TargetFrameInfo::StackGrowsUp, 8, 0),
  TLInfo(*this), TSInfo(*this) {
  if (getRelocationModel() == Reloc::Default) {
      setRelocationModel(Reloc::Static);
  }

  if (getCodeModel() == CodeModel::Default)
    setCodeModel(CodeModel::Small);
}

// Install an instruction selector pass using
// the ISelDag to gen MBlaze code.
bool MBlazeTargetMachine::
addInstSelector(PassManagerBase &PM, CodeGenOpt::Level OptLevel) {
  PM.add(createMBlazeISelDag(*this));
  return false;
}

// Implemented by targets that want to run passes immediately before
// machine code is emitted. return true if -print-machineinstrs should
// print out the code after the passes.
bool MBlazeTargetMachine::
addPreEmitPass(PassManagerBase &PM, CodeGenOpt::Level OptLevel) {
  PM.add(createMBlazeDelaySlotFillerPass(*this));
  return true;
}
