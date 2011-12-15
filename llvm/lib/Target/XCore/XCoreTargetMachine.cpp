//===-- XCoreTargetMachine.cpp - Define TargetMachine for XCore -----------===//
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

#include "XCoreTargetMachine.h"
#include "XCore.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

/// XCoreTargetMachine ctor - Create an ILP32 architecture model
///
XCoreTargetMachine::XCoreTargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
  : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
    Subtarget(TT, CPU, FS),
    DataLayout("e-p:32:32:32-a0:0:32-f32:32:32-f64:32:32-i1:8:32-i8:8:32-"
               "i16:16:32-i32:32:32-i64:32:32-n32"),
    InstrInfo(),
    FrameLowering(Subtarget),
    TLInfo(*this),
    TSInfo(*this) {
}

bool XCoreTargetMachine::addInstSelector(PassManagerBase &PM) {
  PM.add(createXCoreISelDag(*this, getOptLevel()));
  return false;
}

// Force static initialization.
extern "C" void LLVMInitializeXCoreTarget() {
  RegisterTargetMachine<XCoreTargetMachine> X(TheXCoreTarget);
}
