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
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Module.h"
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
    DL("e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:32-f64:32-a:0:32-n32"),
    InstrInfo(),
    FrameLowering(Subtarget),
    TLInfo(*this),
    TSInfo(*this) {
  initAsmInfo();
}

namespace {
/// XCore Code Generator Pass Configuration Options.
class XCorePassConfig : public TargetPassConfig {
public:
  XCorePassConfig(XCoreTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  XCoreTargetMachine &getXCoreTargetMachine() const {
    return getTM<XCoreTargetMachine>();
  }

  virtual bool addPreISel();
  virtual bool addInstSelector();
  virtual bool addPreEmitPass();
};
} // namespace

TargetPassConfig *XCoreTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new XCorePassConfig(this, PM);
}

bool XCorePassConfig::addPreISel() {
  addPass(createXCoreLowerThreadLocalPass());
  return false;
}

bool XCorePassConfig::addInstSelector() {
  addPass(createXCoreISelDag(getXCoreTargetMachine(), getOptLevel()));
  return false;
}

bool XCorePassConfig::addPreEmitPass() {
  addPass(createXCoreFrameToArgsOffsetEliminationPass());
  return false;
}

// Force static initialization.
extern "C" void LLVMInitializeXCoreTarget() {
  RegisterTargetMachine<XCoreTargetMachine> X(TheXCoreTarget);
}

void XCoreTargetMachine::addAnalysisPasses(PassManagerBase &PM) {
  // Add first the target-independent BasicTTI pass, then our XCore pass. This
  // allows the XCore pass to delegate to the target independent layer when
  // appropriate.
  PM.add(createBasicTargetTransformInfoPass(this));
  PM.add(createXCoreTargetTransformInfoPass(this));
}
