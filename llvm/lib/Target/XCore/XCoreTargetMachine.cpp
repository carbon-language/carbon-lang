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

#include "XCoreMCAsmInfo.h"
#include "XCoreTargetMachine.h"
#include "XCore.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

/// XCoreTargetMachine ctor - Create an ILP32 architecture model
///
XCoreTargetMachine::XCoreTargetMachine(const Target &T, const std::string &TT,
                                       const std::string &FS)
  : LLVMTargetMachine(T, TT),
    Subtarget(TT, FS),
    DataLayout("e-p:32:32:32-a0:0:32-f32:32:32-f64:32:32-i1:8:32-i8:8:32-"
               "i16:16:32-i32:32:32-i64:32:32"),
    InstrInfo(),
    FrameInfo(*this),
    TLInfo(*this) {
}

bool XCoreTargetMachine::addInstSelector(PassManagerBase &PM,
                                         CodeGenOpt::Level OptLevel) {
  PM.add(createXCoreISelDag(*this));
  return false;
}

// Force static initialization.
extern "C" void LLVMInitializeXCoreTarget() {
  RegisterTargetMachine<XCoreTargetMachine> X(TheXCoreTarget);
  RegisterAsmInfo<XCoreMCAsmInfo> Y(TheXCoreTarget);
}
